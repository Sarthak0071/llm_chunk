"""
verify_storage.py -- Stage 0: is what got saved what should have been saved?

Mechanical, not statistical. There is no small-sample problem here, so this
gives REAL pass/fail verdicts on 10 documents and scales to 925 unchanged.

INDEPENDENT OF THE GENERATOR, on purpose. A verifier that shares code with the
thing it checks cannot catch that thing's blind spots. So this module:
  - reads only data/*.json and output/chunk/*.json,
  - NEVER reads the *_review.json sidecars (that would be trusting the
    generator's own account of itself),
  - re-derives case_no and the Turkish-language check from scratch rather than
    importing chunk_generate,
  - and adds the two checks the generator structurally cannot do: corpus-wide
    chunk_id uniqueness, and cross-source schema comparison.

ONE DELIBERATE EXCEPTION. Two pure functions ARE imported from the generator:
make_canonical_id and normalise_law_short. The check "stored key == rebuilt
key" cannot exist without the canonical key builder, and a second copy here
would drift from the real one while looking independent. What is still NOT
imported is any of the generator's judgements, flags or sidecars.

FAIL = the stored data is wrong.       -> exit code 1
WARN = a cross-check disagrees and a human should look. -> exit code 0

Usage:  python verify_storage.py [--json]
"""

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import corpus as C

# Only the two key builders, nothing else from the generator (see module doc).
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "llm_chunk"))
from chunk_generate import make_canonical_id, normalise_law_short  # noqa: E402

MAX_CHUNK_CHARS = 2000
RULING_ROLES = {"conclusion", "outcome"}
DISSENT_ROLES = {"dissent"}
LEGISLATION_TYPES = {"statute", "decree_law", "constitution", "regulation", "directive"}
CONTENT_TYPES = {"reasoning", "ruling"}
REASONING_STAGES = {"background", "analysis", "outcome"}   # docs 14.1, universal

# Per-source role vocabularies, stated independently of the generator's prompt.
ROLE_VOCAB = {
    "aym": {"facts", "issue", "rule", "application", "conclusion", "dissent", "unknown"},
    "bam": {"facts", "issue", "rule_application", "conclusion", "dissent", "other"},
    "danistay": {"facts", "issue", "rule_application", "conclusion", "dissent", "other"},
    "first_degree": {"facts", "issue", "rule_application", "conclusion", "dissent", "other"},
    "kvkk": {"background", "analysis", "outcome"},
}

OUTCOME_PATTERNS = {
    "no_violation": [r"İHLÂL\s+EDİLMEDİĞİNE", r"İHLAL\s+EDİLMEDİĞİNE"],
    "violation": [r"İHLÂL\s+EDİLDİĞİNE", r"İHLAL\s+EDİLDİĞİNE"],
    "denied": [r"REDDİNE"],
    "affirmed": [r"ONANMASINA"],
}

# The outcome LABEL is written by the model in Turkish ("ihlal_yok"), while the
# pattern keys above are English. Comparing them directly reports a
# disagreement that is only a language difference, so map each verdict to the
# label fragments that legitimately express it. Checked most-specific first:
# "ihlal_yok" contains "ihlal", so no_violation must be tested before violation.
OUTCOME_LABEL_FORMS = {
    "no_violation": ("ihlal_yok", "ihlal_olmad", "ihlal_edilmedi", "no_violation",
                     "ihlal_bulunmad"),
    "violation": ("ihlal", "violation"),
    "denied": ("red", "denied", "dismiss", "kabul_edilemez"),
    "affirmed": ("onan", "onama", "affirm", "onandi"),
}

# Case-number lines in the document's own header, e.g. "ESAS NO : 2016/1359".
# Redacted headers ("DOSYA NO : ...") simply do not match and are skipped.
HEADER_CASE_RE = re.compile(
    r"(?:ESAS|DOSYA)\s*(?:NO)?\s*[:\.]?\s*(\d{4}\s*/\s*\d+)", re.IGNORECASE)

ENGLISH_STOPWORDS = {"the", "and", "of", "was", "were", "that", "this", "court",
                     "applicant", "with", "which", "from", "have", "been"}
TURKISH_CHARS = set("çğıöşüÇĞİÖŞÜ")
TURKISH_MARKERS = {
    "ve", "ile", "bu", "bir", "icin", "için", "karar", "karari", "kararı",
    "dava", "davanin", "davanın", "mahkeme", "mahkemesi", "hakki", "hakkı",
    "basvuru", "başvuru", "ihlal", "reddine", "kabul", "edilmis", "edilmiş",
    "verilmistir", "verilmiştir", "olmadigina", "olmadığına", "kanun", "kanunun",
    "madde", "maddesi", "sayili", "sayılı", "idari", "para", "cezasi", "cezası",
    "kurul", "nedeniyle", "uyarinca", "uyarınca", "gore", "göre",
}


def tr_lower(s):
    return (s or "").replace("İ", "i").replace("I", "ı").lower()


def tr_upper(s):
    return (s or "").replace("i", "İ").replace("ı", "I").upper()


def looks_turkish(text):
    """Re-derived here rather than imported, so a bug in the generator's own
    language check cannot hide behind the verifier agreeing with it."""
    letters = [c for c in text if c.isalpha()]
    words = re.findall(r"[a-zçğıöşü]+", tr_lower(text))
    if not letters or not words:
        return False
    if sum(1 for w in words if w in ENGLISH_STOPWORDS) / len(words) >= 0.08:
        return False
    if sum(1 for c in letters if c in TURKISH_CHARS) / len(letters) >= 0.01:
        return True
    return any(w in TURKISH_MARKERS for w in words)


def expected_case_no(record, source):
    """Independent re-derivation from the raw record's own metadata."""
    if source == "kvkk":
        try:
            meta = json.loads(record.get("metadata") or "{}")
        except (ValueError, TypeError):
            meta = {}
        raw = (meta.get("data") or {}).get("decision_number_raw") or ""
        case_no = raw.strip().lstrip(":-").strip()
        if not re.fullmatch(r"\d{4}/\d+", case_no):
            if record.get("karar_year") and record.get("karar_no"):
                return f'{record["karar_year"]}/{record["karar_no"]}'
        return case_no or None
    if record.get("esas_year") and record.get("esas_no"):
        return f'{record["esas_year"]}/{record["esas_no"]}'
    return None


class Results:
    def __init__(self):
        self.checks = []

    def add(self, name, status, detail=""):
        self.checks.append({"check": name, "status": status, "detail": detail})

    def report(self, name, failures, warn=False, note=""):
        label = "WARN" if warn else "FAIL"
        if failures:
            self.add(name, label, f"{len(failures)} item(s)")
            print(f"  [{label}] {name}  ({len(failures)})")
            for f in failures[:8]:
                print(f"         {f}")
            if len(failures) > 8:
                print(f"         ... {len(failures) - 8} more")
        else:
            self.add(name, "PASS", note)
            print(f"  [PASS] {name}{('  ' + note) if note else ''}")


def run(as_json=False):
    c = C.Corpus()
    if not c.chunks:
        print("No generated output found in output/chunk/. Run chunk_generate.py first.")
        return 1

    raw = C.load_raw_documents(wanted_doc_ids=set(c.documents))
    res = Results()
    s = c.summary()
    print(f"Corpus: {s['documents']} documents, {s['chunks']} chunks, "
          f"{s['capsules']} capsules")
    print(f"Raw source documents located: {len(raw)}/{len(c.documents)}\n")
    print("STORAGE VERIFICATION")

    # ---- 1. schema conformance -------------------------------------------
    bad = []
    for ch in c.chunks:
        src = ch.get("source_type")
        expect = set(C.CHUNK_BASE_FIELDS)
        role_field = C.ROLE_FIELD.get(src)
        if role_field != "firac_role":
            expect.add(role_field)
        missing, extra = expect - set(ch), set(ch) - expect
        if missing or extra:
            bad.append(f"{ch.get('chunk_label')}: missing={sorted(missing)} extra={sorted(extra)}")
        if src == "aym" and ch.get("firac_role") is None:
            bad.append(f"{ch.get('chunk_label')}: aym chunk has null firac_role")
        if src != "aym" and ch.get("firac_role") is not None:
            bad.append(f"{ch.get('chunk_label')}: non-aym chunk has firac_role set")
        if src != "aym" and ch.get("rights") is not None:
            bad.append(f"{ch.get('chunk_label')}: non-aym chunk has rights set")
    res.report("chunk schema conformance", bad, note=f"{len(c.chunks)} chunks")

    bad = [f"{cap.get('source_type')}/{cap.get('case_no')}: "
           f"missing={sorted(set(C.CAPSULE_FIELDS) - set(cap))} "
           f"extra={sorted(set(cap) - set(C.CAPSULE_FIELDS))}"
           for cap in c.capsules
           if set(cap) != set(C.CAPSULE_FIELDS)]
    res.report("capsule schema conformance", bad, note=f"{len(c.capsules)} capsules")

    # ---- 2. grounding -----------------------------------------------------
    # Two DIFFERENT things, deliberately separated. "text is not a contiguous
    # substring of the document" conflates invented text with a gap in
    # paragraph_refs, and those need different responses. Since text is
    # assembled from the referenced paragraphs, the real grounding question is
    # whether it is built only from real source paragraphs.
    ungrounded, mentions, gaps = [], [], []
    for ch in c.chunks:
        key = (ch["source_type"], C.doc_id_of(ch))
        doc = raw.get(key)
        if not doc:
            ungrounded.append(f"{ch.get('chunk_label')}: raw document not found")
            continue
        paras = doc["paragraphs"]
        nums = sorted(int(r[1:]) for r in (ch.get("source_paragraph_ids") or [])
                      if C.PARAGRAPH_RE.match(str(r)))
        if not nums:
            ungrounded.append(f"{ch.get('chunk_label')}: no usable source_paragraph_ids")
            continue
        if max(nums) > len(paras):
            ungrounded.append(f"{ch.get('chunk_label')}: references p{max(nums)} but the "
                              f"document has {len(paras)} paragraphs")
            continue
        own = C.norm_ws("\n".join(paras[i - 1] for i in nums))
        if C.norm_ws(ch.get("text")) not in own:
            ungrounded.append(f"{ch.get('chunk_label')}: text is not built from its own "
                              f"referenced paragraphs")
        # A gap means the model skipped a paragraph inside the range it claimed,
        # so that content is silently absent from the stored chunk.
        missing = sorted(set(range(min(nums), max(nums) + 1)) - set(nums))
        if missing:
            gaps.append(f"{ch.get('chunk_label')}: refs span p{min(nums)}-p{max(nums)} "
                        f"but skip {['p%d' % m for m in missing]} "
                        f"({sum(len(paras[m-1]) for m in missing)} chars dropped)")
        for g in ch.get("cited_legislations") or []:
            vm = C.norm_ws(g.get("verbatim_mention"))
            if vm and vm not in C.norm_ws(ch.get("text")):
                mentions.append(f"{ch.get('chunk_label')}: verbatim_mention not in chunk text")
    res.report("chunk text built only from real source paragraphs", ungrounded,
               note=f"{len(c.chunks)} chunks")
    res.report("verbatim_mention grounded in chunk text", mentions)
    res.report("paragraph_refs have no internal gaps", gaps, warn=True)

    # ---- 3. uniqueness and arithmetic -------------------------------------
    dupes = [f"{cid} x{n}" for cid, n in Counter(ch["chunk_id"] for ch in c.chunks).items() if n > 1]
    res.report("chunk_id unique corpus-wide", dupes, note=f"{len(c.chunk_by_id)} ids")

    bad = [f"{ch.get('chunk_label')}: char_length={ch.get('char_length')} len(text)={len(ch.get('text') or '')}"
           for ch in c.chunks if ch.get("char_length") != len(ch.get("text") or "")]
    res.report("char_length == len(text)", bad)

    over = [f"{ch.get('chunk_label')}: {ch.get('char_length')}" for ch in c.chunks
            if (ch.get("char_length") or 0) > MAX_CHUNK_CHARS]
    res.report(f"char_length <= {MAX_CHUNK_CHARS}", over)

    # ---- 4. internal consistency ------------------------------------------
    # Enum vocabulary is a storage question, not a judgement call: a value
    # outside the documented set is stored-wrong, so it FAILS rather than warns.
    bad = []
    for ch in c.chunks:
        if ch.get("content_type") not in CONTENT_TYPES:
            bad.append(f"{ch.get('chunk_label')}: content_type={ch.get('content_type')!r}")
        if ch.get("reasoning_stage") not in REASONING_STAGES:
            bad.append(f"{ch.get('chunk_label')}: reasoning_stage="
                       f"{ch.get('reasoning_stage')!r} not in {sorted(REASONING_STAGES)}")
        role = ch.get(C.ROLE_FIELD.get(ch["source_type"]))
        if role is None:
            bad.append(f"{ch.get('chunk_label')}: "
                       f"{C.ROLE_FIELD.get(ch['source_type'])} is null")
        elif role not in ROLE_VOCAB[ch["source_type"]]:
            bad.append(f"{ch.get('chunk_label')}: role={role!r} not in "
                       f"{sorted(ROLE_VOCAB[ch['source_type']])}")
    res.report("enum fields within documented vocabulary", bad)

    bad = [f"{ch.get('chunk_label')}: content_type=ruling but role="
           f"{ch.get(C.ROLE_FIELD.get(ch['source_type']))!r}"
           for ch in c.chunks
           if ch.get("content_type") == "ruling"
           and ch.get(C.ROLE_FIELD.get(ch["source_type"])) not in RULING_ROLES]
    res.report("content_type consistent with role", bad, warn=True)

    bad = []
    for cap in c.capsules:
        opinion = cap.get("opinion_type") or ""
        roles = {c.chunk_by_id[cid].get(C.ROLE_FIELD.get(c.chunk_by_id[cid]["source_type"]))
                 for cid in cap.get("supporting_chunk_ids") or [] if cid in c.chunk_by_id}
        if opinion.startswith("dissent") != bool(roles & DISSENT_ROLES):
            bad.append(f"{cap.get('source_type')}/{cap.get('case_no')}: "
                       f"opinion_type={opinion!r} vs supporting roles {sorted(r for r in roles if r)}")
    res.report("opinion_type matches dissent-role support", bad, warn=True)

    bad = []
    for cap in c.capsules:
        docs = c.capsule_doc_ids(cap)
        ruling = " ".join(ch.get("text") or "" for ch in c.chunks
                          if C.doc_id_of(ch) in docs and ch.get("content_type") == "ruling")
        if not (cap.get("outcome") and ruling):
            continue
        up = tr_upper(ruling)
        hits = [k for k, pats in OUTCOME_PATTERNS.items() if any(re.search(p, up) for p in pats)]
        if not hits:
            continue
        # A ruling that states both a violation and a non-violation (different
        # rights in one decision) cannot adjudicate the label either way.
        if "violation" in hits and "no_violation" in hits:
            continue
        label = tr_lower(cap["outcome"])
        if not any(frag in label for k in hits for frag in OUTCOME_LABEL_FORMS.get(k, ())):
            bad.append(f"{cap.get('source_type')}/{cap.get('case_no')}: "
                       f"outcome={cap['outcome']!r} vs ruling suggests {'/'.join(hits)}")
    res.report("outcome not contradicted by ruling text", bad, warn=True)

    bad = [f"{ch.get('chunk_label')}: article_no={g.get('article_no')} but confidence=low"
           for ch in c.chunks for g in ch.get("cited_legislations") or []
           if g.get("article_no") and g.get("confidence") == "low"]
    res.report("citation confidence not contradicted by article_no", bad, warn=True)

    # ---- 5. language -------------------------------------------------------
    for field in ("reasoning_summary", "conclusion_sentence"):
        bad = [f"{cap.get('source_type')}/{cap.get('case_no')}: {field} not Turkish -- "
               f"{(cap.get(field) or '')[:60]!r}"
               for cap in c.capsules if not looks_turkish(cap.get(field) or "")]
        res.report(f"{field} is Turkish", bad, note=f"{len(c.capsules)} capsules")

    bad = [f"{cap.get('source_type')}/{cap.get('case_no')}: {cap.get('reasoning_summary_method')!r}"
           for cap in c.capsules if cap.get("reasoning_summary_method") != "llm_generated"]
    res.report("reasoning_summary_method == llm_generated", bad)

    # ---- 6. referential integrity -----------------------------------------
    bad, empty = [], []
    for cap in c.capsules:
        ids = cap.get("supporting_chunk_ids") or []
        if not ids:
            empty.append(f"{cap.get('source_type')}/{cap.get('case_no')}: no supporting chunks")
        for cid in ids:
            if cid not in c.chunk_by_id:
                bad.append(f"{cap.get('source_type')}/{cap.get('case_no')}: dangling chunk_id {cid}")
        docs = c.capsule_doc_ids(cap)
        if len(docs) > 1:
            bad.append(f"{cap.get('source_type')}/{cap.get('case_no')}: "
                       f"supporting chunks span {len(docs)} documents")
    res.report("supporting_chunk_ids resolve to real chunks", bad)
    res.report("every capsule has supporting chunks", empty)

    bad = [f"{ch.get('chunk_label')}: legislation_type={g.get('legislation_type')!r}"
           for ch in c.chunks for g in ch.get("cited_legislations") or []
           if g.get("legislation_type") not in LEGISLATION_TYPES]
    res.report("legislation_type within documented vocabulary", bad, warn=True)

    bad = [f"{ch.get('chunk_label')}: {g.get('legislation_type')} citation has no canonical_id"
           for ch in c.chunks for g in ch.get("cited_legislations") or []
           if g.get("canonical_id") is None and not g.get("law_short")]
    res.report("citations identifiable (canonical_id or law_short)", bad, warn=True)

    # ---- 6b. citation schema + law_short + canonical_id re-derivation --------
    bad = [f"{ch.get('chunk_label')}: citation keys missing={sorted(set(C.CITATION_FIELDS)-set(g))} "
           f"extra={sorted(set(g)-set(C.CITATION_FIELDS))}"
           for ch in c.chunks for g in ch.get("cited_legislations") or []
           if set(g) != set(C.CITATION_FIELDS)]
    res.report("citation schema conformance (10 keys)", bad)

    # law_short must be the normalised form: no dots/spaces, Turkish-upper.
    bad = []
    for ch in c.chunks:
        for g in ch.get("cited_legislations") or []:
            ls = g.get("law_short")
            if ls is None:
                continue
            if ls != normalise_law_short(ls) or len(ls) > 10:
                bad.append(f"{ch.get('chunk_label')}: law_short={ls!r} not normalised")
    res.report("law_short normalised (no dots, tr_upper)", bad)

    # canonical_id must equal what the key builder recomputes from the stored
    # fields, so the stored keys cannot drift from the builder (or from each
    # other: three spellings of 'Geçici 3' must give one key).
    bad = []
    for ch in c.chunks:
        for g in ch.get("cited_legislations") or []:
            want = make_canonical_id(g.get("law_no"), g.get("article_no"),
                                     g.get("legislation_type"), g.get("law_name"))
            if g.get("canonical_id") != want:
                bad.append(f"{ch.get('chunk_label')}: stored={g.get('canonical_id')!r} "
                           f"recomputed={want!r}")
    res.report("canonical_id == recomputed from stored fields", bad)

    # ---- 6c. decision_date ---------------------------------------------------
    bad, per_doc = [], {}
    for ch in c.chunks:
        d = ch.get("decision_date")
        if d is not None and not C.ISO_DATE_RE.match(str(d)):
            bad.append(f"{ch.get('chunk_label')}: decision_date={d!r} not ISO")
        per_doc.setdefault((ch["source_type"], C.doc_id_of(ch)), set()).add(d)
    for cap in c.capsules:
        d = cap.get("decision_date")
        if d is not None and not C.ISO_DATE_RE.match(str(d)):
            bad.append(f"capsule {cap.get('case_no')}: decision_date={d!r} not ISO")
        for doc in c.capsule_doc_ids(cap):
            per_doc.setdefault((cap["source_type"], doc), set()).add(d)
    for key, vals in per_doc.items():
        if len(vals) > 1:
            bad.append(f"{key[0]}/{key[1]}: decision_date differs within document: {sorted(map(str, vals))}")
    res.report("decision_date ISO and consistent within each document", bad)

    missing = [f"{s}/{d}" for (s, d), vals in per_doc.items() if vals == {None}]
    res.report("decision_date present", missing, warn=True,
               note="null is allowed only when neither metadata nor the ruling sentence carries a date")

    # ---- 7. case_no re-derived independently from metadata -----------------
    bad = []
    for (source, doc_id), doc in raw.items():
        want = expected_case_no(doc["record"], source)
        got = c.case_no_of_doc.get((source, doc_id))
        if want and got and want != got:
            bad.append(f"{source}/{doc_id}: stored={got!r} metadata={want!r}")
    res.report("stored case_no matches raw metadata", bad, warn=True)

    # The document's OWN header vs the database columns. Re-derived straight
    # from the court text, so this reproduces the metadata conflict without
    # trusting anything the generator reported about it.
    bad = []
    for (source, doc_id), doc in raw.items():
        header = " ".join(doc["paragraphs"][:12])
        stated = {m.group(1).replace(" ", "") for m in HEADER_CASE_RE.finditer(header)}
        got = c.case_no_of_doc.get((source, doc_id))
        if stated and got and got not in stated:
            bad.append(f"{source}/{doc_id}: stored={got!r} but document header "
                       f"states {sorted(stated)}")
    res.report("stored case_no matches the document's own header", bad, warn=True)

    # ---- summary -----------------------------------------------------------
    counts = Counter(x["status"] for x in res.checks)
    print(f"\n  {counts['PASS']} passed, {counts['WARN']} warnings, {counts['FAIL']} failures")
    if counts["FAIL"]:
        print("  FAIL means stored data is wrong and must be fixed.")
    if counts["WARN"]:
        print("  WARN means a cross-check disagrees -- for human review, not necessarily a bug.")

    if as_json:
        C.REPORT_DIR.mkdir(parents=True, exist_ok=True)
        out = C.REPORT_DIR / "storage_report.json"
        out.write_text(json.dumps({"corpus": s, "checks": res.checks}, ensure_ascii=False,
                                  indent=2), encoding="utf-8")
        print(f"  wrote {out}")
    return 1 if counts["FAIL"] else 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Stage 0 storage verification")
    ap.add_argument("--json", action="store_true", help="also write storage_report.json")
    sys.exit(run(ap.parse_args().json))
