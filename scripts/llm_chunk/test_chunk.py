"""
test_chunk.py -- every check for chunker.py, in one place. Free by default;
anything that calls the API needs --live.

    python test_chunk.py --schema                 # closed schema rejects illegal values; worked examples validate
    python test_chunk.py --extraction [--source X] # word-level: do our paragraphs carry every word of the record?
    python test_chunk.py --stress [--source X]     # every usable record x 6 broken model answers: 0 failures required
    python test_chunk.py --mock-retry              # 429, 429, 200 against the real SDK retry, offline
    python test_chunk.py --merge-test              # targeted rerun merge, fragment rule, laws windows, correction rules
    python test_chunk.py --verify output/chunk     # every paragraph and every raw word stored once? values legal?
    python test_chunk.py --replay output/chunk     # re-run repair/assemble/check on stored model answers, no API
    python test_chunk.py --rebuild output/eval --out-dir output/eval_rebuilt   # current code on saved answers
    python test_chunk.py --scorecard output/chunk --compare <older run dir>   # same quality numbers, side by side
    python test_chunk.py --live --smoke            # 2 documents (1963/18 norm review + one kvkk), then --verify
"""

import argparse
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import chunk_lib as lib
import chunker as C
import prompts

SMOKE = {"aym": ["b376aa8a-7f7a-43c3-6e76-2932be9f34a6"],
         "kvkk": ["c3ce8681-4774-5eb9-bcff-3fc0bd188a7d"]}
COURTS = ["aym", "bam", "danistay", "first_degree", "kvkk", "yargitay"]

# The stored shape (docs 4, 6, 13.1; same lists as retrieval_test/corpus.py).
CHUNK_FIELDS = {"chunk_id", "chunk_label", "source_type", "case_no", "decision_date", "citation_granularity",
                "source_paragraph_ids", "text", "char_length", "content_type", "firac_role", "reasoning_stage",
                "rights", "confidence", "cited_legislations", "role", "role_vocabulary", "schema_version"}
CAPSULE_FIELDS = {"case_no", "source_type", "decision_date", "subject_type", "subject_id", "opinion_type",
                  "outcome", "conclusion_sentence", "reasoning_summary", "reasoning_summary_method",
                  "supporting_chunk_ids"}
CITATION_FIELDS = {"canonical_id", "legislation_type", "law_no", "law_short", "law_name", "article_no",
                   "paragraph_no", "verbatim_mention", "law_date", "confidence"}
ALLOWED_NULL = {"decision_date", "case_no", "rights", "firac_role", "law_no", "law_short", "law_name",
                "article_no", "paragraph_no", "law_date", "verbatim_mention", "canonical_id"}


def raw_records(sources):
    out = {}
    for s in sources:
        p = C.DATA_DIR / f"{s}.json"
        if p.is_file():
            for r in json.loads(p.read_text(encoding="utf-8")):
                out[(s, str(r.get("doc_id")))] = r
    return out


def usable(source):
    recs = json.loads((C.DATA_DIR / f"{source}.json").read_text(encoding="utf-8"))
    return [r for r in recs if r.get("status") in C.USABLE_STATUSES
            and ((r.get("content_text") or "").strip() or (r.get("html_content") or "").strip())]


def resolve(d):
    base = Path(d)
    return base if base.is_absolute() else C.ROOT / base


def saved_answers(r, source, kind, cands):
    """(structure, capsules, [(window, laws answer)]) from a review entry, in the current
    format or the earlier one (capsules inside the structure answer, one laws answer)."""
    structure = C.structure_model_for(source, kind, cands).model_validate(r["raw_response"]) \
        if r.get("raw_response") else None
    if r.get("raw_capsules"):
        capsules = C.capsules_model_for(source, kind, cands).model_validate(r["raw_capsules"])
    else:
        capsules = C.legacy_capsules(r.get("raw_response"), source, kind, cands)
    saved, laws = r.get("raw_laws"), []
    if isinstance(saved, dict):
        laws.append((None, C.LawsResponse.model_validate(saved)))
    elif isinstance(saved, list):
        laws += [(tuple(x["window"]), C.LawsResponse.model_validate(x["answer"])) for x in saved if x.get("answer")]
    return structure, capsules, laws


def cites_from(laws, n, log):
    cites = {}
    for window, answer in laws:
        for i, found in C.citations_by_paragraph(answer, n, log, window).items():
            cites.setdefault(i, []).extend(found)
    return cites


def missing_words(rec, chunks):
    """End to end, from the raw record to stored text: raw words (longer column,
    as a multiset) that no chunk of the document carries. Scraper junk lines aside."""
    raw_text, raw_lines = C.raw_column(rec)
    junk = [q for q in raw_lines if C.JUNK_LINE_RE.match(q.strip())]
    return C.lost_words(raw_text, [c["text"] for c in chunks], junk)


def invariants(source, rec, paras, chunks, caps):
    """Every guarantee code gives, on one document's stored output. [] = all hold."""
    fails, n = [], len(paras)
    order, pieces = [], {}
    for ch in chunks:                                   # size-cap pieces share their refs
        key = tuple(ch["source_paragraph_ids"])
        if key not in pieces:
            order.append(key)
            pieces[key] = []
        pieces[key].append(ch)
    count = Counter()
    for key in order:
        idxs = C._ref_idxs(list(key), n)
        if [f"p{i}" for i in idxs] != list(key):
            fails.append("refs_not_clean")
        count.update(idxs)
        want = C.norm_ws(" ".join(paras[i - 1] for i in idxs))
        if C.norm_ws(" ".join(p["text"] for p in pieces[key])) != want:
            fails.append("text_differs_from_paragraphs")
    if any(count[i] == 0 for i in range(1, n + 1)):
        fails.append("paragraph_missing")
    if any(count[i] > 1 for i in range(1, n + 1)):
        fails.append("paragraph_duplicated")
    firsts = [int(k[0][1:]) for k in order if k]
    if firsts != sorted(firsts):
        fails.append("chunks_out_of_order")
    if len({c["chunk_id"] for c in chunks}) != len(chunks):
        fails.append("chunk_id_not_unique")
    roles = prompts.ROLE_VOCAB[source][1]
    role_field = C.SOURCES[source]["role_field"]
    fields = CHUNK_FIELDS | ({role_field} if role_field != "firac_role" else set())
    for ch in chunks:
        if set(ch) != fields:
            fails.append("chunk_fields")
        if ch["char_length"] != len(ch["text"]) or ch["char_length"] > lib.MAX_CHUNK_CHARS or not ch["text"]:
            fails.append("chunk_size")
        if ch["role"] not in roles or ch.get(role_field) != ch["role"]:
            fails.append("role_illegal")
        elif (ch["content_type"], ch["reasoning_stage"]) != C.derive(ch["role"]):
            fails.append("derived_fields_wrong")
        if any(v is None and k not in ALLOWED_NULL for k, v in ch.items()):
            fails.append("null_not_allowed")
        for cit in ch["cited_legislations"]:
            if set(cit) != CITATION_FIELDS or cit["legislation_type"] not in C.LEGISLATION_TYPES:
                fails.append("citation_shape")
            if cit["legislation_type"] == "constitution" and cit["law_no"]:
                fails.append("constitution_with_law_no")
    ids = {c["chunk_id"] for c in chunks}
    for cap in caps:
        if set(cap) != CAPSULE_FIELDS:
            fails.append("capsule_fields")
        if any(i not in ids for i in cap["supporting_chunk_ids"]):
            fails.append("capsule_link_broken")
        if cap["outcome"] not in C.OUTCOME_GLOSS:
            fails.append("outcome_illegal")
    if missing_words(rec, chunks):
        fails.append("raw_words_missing")
    return sorted(set(fails))


# ---------------------------------------------------------------------------

def test_schema():
    ok = True
    for source in COURTS:
        kinds = {"aym": ["aym_individual_application", "aym_norm_review"],
                 "yargitay": ["yargitay_hukuk", "yargitay_ceza"]}.get(source, [source])
        for kind in kinds:
            cands = ["mülkiyet_hakkı"] if kind == "aym_individual_application" else []
            sm, cm = C.structure_model_for(source, kind, cands), C.capsules_model_for(source, kind, cands)
            bad = {"capsules": [{"outcome": "karsi_oy", "opinion_type": "dissent",
                   "subject_id": cands[0] if cands else "x_y", "conclusion_sentence": "a" * 30,
                   "reasoning_summary": "b" * 90, "supporting_paragraph_refs": ["p1"]}]}
            try:
                cm.model_validate(bad)
                print(f"  FAIL {kind}: accepted outcome 'karsi_oy'")
                ok = False
            except Exception:
                pass
            vocab = {"outcomes": C.OUTCOME_BY_KIND[kind], "gloss": C.OUTCOME_GLOSS,
                     "opinion_kinds": C.OPINION_KINDS if source == "kvkk" else C.OPINION_KINDS[:3],
                     "legislation_types": C.RESPONSE_LEGISLATION_TYPES}
            few = prompts.build_fewshot(source, kind, "2020/1", cands, vocab["outcomes"])
            status = "no example"
            if few:
                try:
                    sm.model_validate({"segments": few["segments"]})
                    cm.model_validate({"capsules": few["capsules"]})
                    status = "example validates"
                except Exception as e:
                    status = f"EXAMPLE INVALID: {str(e)[:160]}"
                    ok = False
            # keywords Gemini's response_schema does not accept next to enums
            for label, model_ in (("structure", sm), ("capsules", cm), ("laws", C.LawsResponse)):
                js = json.dumps(model_.model_json_schema())
                bad_kw = [k for k in ("maxItems", "maxLength", "pattern", "minimum", "maximum") if f'"{k}"' in js]
                if bad_kw:
                    print(f"  FAIL {kind} {label} schema uses {bad_kw} -- Gemini rejects these")
                    ok = False
            si = prompts.build_structure_instruction(source, kind, "2020/1", cands, vocab)
            ci = prompts.build_capsules_instruction(source, kind, "2020/1", cands, [], vocab)
            for text in (prompts.build_structure_instruction(source, kind, "2020/1", cands, vocab, with_example=False),
                         prompts.build_capsules_instruction(source, kind, "2020/1", cands, [], vocab, with_example=False)):
                if "## Worked example" in text:
                    print(f"  FAIL {kind}: example document still gets a worked example")
                    ok = False
            print(f"  {kind:28} schemas ok | structure prompt {len(si):6} | capsules prompt {len(ci):6} chars | {status}")
    ex = []
    for source in COURTS:
        for r in usable(source):
            if prompts.is_example_document(source, C.compute_case_no(r, source), str(r["doc_id"])):
                ex.append(f"{source} {str(r['doc_id'])[:12]}")
    print(f"  worked-example documents in the corpus (chunked without their example): {len(ex)}: {', '.join(ex)}")
    print("SCHEMA:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def test_extraction(sources):
    rows = []
    for s in sources:
        for r in usable(s):
            paras, lost, _ = C.paragraphs(r)
            rows.append((s, str(r["doc_id"]), len(paras), sum(lost.values()), sum(len(p) for p in paras),
                         sorted(lost.items(), key=lambda x: -x[1])[:8]))
    print(f"{'source':13} {'docs':>5} {'median paras':>12} {'docs losing words':>18}")
    for s in sources:
        rs = [x for x in rows if x[0] == s]
        if not rs:
            continue
        ps = sorted(x[2] for x in rs)
        print(f"{s:13} {len(rs):>5} {ps[len(ps) // 2]:>12} {sum(1 for x in rs if x[3]):>18}")
    low = [x for x in rows if x[3] or not x[2]]
    for x in sorted(low, key=lambda x: -x[3])[:20]:
        print(f"   LOSS {x[0]:12} {x[1][:20]:20} words lost {x[3]} paras {x[2]} {x[5]}")
    out = C.ROOT / "output" / "extraction_report.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps([dict(zip(("source", "doc_id", "paragraphs", "words_lost", "chars", "lost"), x))
                               for x in rows], ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"records {len(rows)} | losing words {len(low)} | wrote {out}")
    return 1 if low else 0


def test_verify(out_dir):
    base = resolve(out_dir)
    raw = raw_records(C.SOURCES)
    rows, hard_total, values, words_total = [], 0, Counter(), 0
    for path in sorted(base.glob("*.json")):
        source = path.stem
        if source not in C.SOURCES:
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        review_path = base / f"{source}_review.json"
        reviews = {str(r.get("doc_id")): r for r in json.loads(review_path.read_text(encoding="utf-8"))} \
            if review_path.is_file() else {}
        by_doc = {}
        for ch in data["chunks"]:
            by_doc.setdefault(C.doc_of(ch), {"chunks": [], "capsules": []})["chunks"].append(ch)
        cdoc = {ch["chunk_id"]: C.doc_of(ch) for ch in data["chunks"]}
        for cap in data["reasoning_capsules"]:
            for d in {cdoc.get(i) for i in cap["supporting_chunk_ids"]} - {None}:
                by_doc[d]["capsules"].append(cap)
        for doc_id, d in by_doc.items():
            rec = raw.get((source, doc_id))
            if rec is None:
                rows.append((source, doc_id, "?", 0, 0, 0, 0, 0, ["no_raw_record"], []))
                continue
            paras, lost, _ = C.paragraphs(rec)
            hard, soft = C.verify_document(d["chunks"], d["capsules"], paras)
            hard = C.extraction_issues(lost) + hard
            if (reviews.get(doc_id) or {}).get("fallback"):
                hard.insert(0, f"fallback_no_model:{reviews[doc_id]['fallback']} (rerun this doc_id)")
            miss = missing_words(rec, d["chunks"])
            words_total += sum(miss.values())
            if miss:
                hard.append(f"missing_words:{sum(miss.values())}: "
                            + ", ".join(f"{w}x{c}" for w, c in sorted(miss.items(), key=lambda x: -x[1])[:6]))
            firsts = [int(ch["source_paragraph_ids"][0][1:]) for ch in d["chunks"] if ch["source_paragraph_ids"]]
            if firsts != sorted(firsts):
                hard.append("chunks_out_of_order")
            # value audit: closed fields, nulls only where allowed
            for ch in d["chunks"]:
                values[("role", ch["role"])] += 1
                for k, v in ch.items():
                    if v is None and k not in ALLOWED_NULL:
                        hard.append(f"null_field:{k}")
                    if k == "rights" and v is not None and source != "aym":
                        hard.append("rights_set_outside_aym")
                for cit in ch["cited_legislations"]:
                    values[("legislation_type", cit["legislation_type"])] += 1
                    if cit["legislation_type"] not in C.LEGISLATION_TYPES:
                        hard.append(f"legislation_type_illegal:{cit['legislation_type']}")
            for cap in d["capsules"]:
                values[("outcome", cap["outcome"])] += 1
                values[("opinion", cap["opinion_type"].split(":")[0])] += 1
                if cap["outcome"] not in C.OUTCOME_GLOSS:
                    hard.append(f"outcome_illegal:{cap['outcome']}")
            hard_total += bool(hard)
            n = len(paras)
            stored = sum(1 for i in range(1, n + 1) if any(f"p{i}" in ch["source_paragraph_ids"] for ch in d["chunks"]))
            rows.append((source, doc_id, C.kind_of(rec, source), n, stored, len(d["chunks"]),
                         len(d["capsules"]), sum(miss.values()), hard, soft))
        # a document in the review file with no stored chunks at all
        for doc_id, r in reviews.items():
            if doc_id not in by_doc:
                hard_total += 1
                rows.append((source, doc_id, r.get("kind", "?"), r.get("paragraphs", 0), 0, 0, 0, 0,
                             [f"no_chunks_stored:{r.get('reason')}"], []))
    print(f"{'source':12} {'doc':14} {'kind':27} {'paras':>5} {'stored':>6} {'chunks':>6} {'caps':>4} "
          f"{'miss_w':>6}  verdict")
    for r in rows:
        v = "ISSUE" if r[8] else ("PASS" + (f" ({len(r[9])} note{'s' if len(r[9]) > 1 else ''})" if r[9] else ""))
        print(f"{r[0]:12} {r[1][:14]:14} {str(r[2]):27} {r[3]:>5} {r[4]:>6} {r[5]:>6} {r[6]:>4} {r[7]:>6}  {v}"
              + (f"  {'; '.join(x[:70] for x in r[8][:3])}" if r[8] else ""))
    print(f"\ndocuments {len(rows)} | with issues {hard_total} | paragraphs in {sum(r[3] for r in rows)} "
          f"stored {sum(r[4] for r in rows)} missing {sum(r[3] - r[4] for r in rows)} | "
          f"missing_words {words_total}")
    kinds = Counter(x.split(':')[0] for r in rows for x in r[9])
    if kinds:
        print("soft notes:", dict(kinds.most_common()))
    issue_kinds = Counter(x.split(':')[0] for r in rows for x in r[8])
    if issue_kinds:
        print("issues:", dict(issue_kinds.most_common()))
    print("values:", {f"{k[0]}={k[1]}": v for k, v in sorted(values.items())})
    report = [dict(zip(("source", "doc_id", "kind", "paragraphs", "stored", "chunks", "capsules",
                        "missing_words", "hard", "soft"), r)) for r in rows]
    (base / "verify_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    return 1 if hard_total else 0


def test_replay(out_dir):
    """Stored raw model answers -> repair -> assemble -> check, no API call, no audit."""
    base = resolve(out_dir)
    raw = raw_records(C.SOURCES)
    n = fails = 0
    for path in sorted(base.glob("*_review.json")):
        source = path.stem.replace("_review", "")
        for r in json.loads(path.read_text(encoding="utf-8")):
            if not r.get("raw_response"):
                continue
            rec = raw.get((source, str(r["doc_id"])))
            if rec is None:
                continue
            paras, _, _ = C.paragraphs(rec)
            kind = C.kind_of(rec, source)
            cands = C.candidate_rights(rec) if kind == "aym_individual_application" else []
            try:
                structure, capsules, laws = saved_answers(r, source, kind, cands)
            except Exception as e:
                print(f"  {source} {r['doc_id'][:12]} raw answer no longer validates: {str(e)[:100]}")
                fails += 1
                n += 1
                continue
            segs, alias, log = C.normalise_segments(structure.segments, paras)
            cites = cites_from(laws, len(paras), log)
            chunks, caps, hard, soft, log = C.assemble(paras, source, str(r["doc_id"]), r.get("case_no"), None, "x",
                                                       segs, alias, capsules.capsules if capsules else [], cites, log)
            broken = invariants(source, rec, paras, chunks, caps)
            n += 1
            fails += bool(hard or broken)
            if hard or broken:
                print(f"  {source} {r['doc_id'][:12]} ISSUES: {'; '.join(h[:80] for h in (broken + hard)[:3])}")
    print(f"replayed {n} stored answers | with issues: {fails}")
    return 1 if fails else 0


# --- stress: the whole corpus under broken model answers ----------------------

STRESS_SCENARIOS = ("perfect", "empty", "half_missing", "duplicate_and_invalid_refs", "shuffled_fragments",
                    "api_failed")


def fake_answer(scenario, source, kind, paras, cands, rng):
    """A model answer broken in one specific way, as the parsed pydantic object."""
    n = len(paras)
    roles = prompts.ROLE_VOCAB[source][1]
    ruling = "outcome" if source in ("kvkk", "rekabet") else "conclusion"
    groups = [list(range(i, min(i + 5, n + 1))) for i in range(1, n + 1, 5)]
    segs = []
    for k, g in enumerate(groups, 1):
        role = ruling if k == len(groups) else roles[(k - 1) % len(roles)]
        segs.append({"local_id": f"seg_{k}", "paragraph_refs": [f"p{i}" for i in g], "role": role,
                     "confidence": "high"})
    if scenario == "empty":
        segs = []
    elif scenario == "half_missing":
        segs = segs[::2]
    elif scenario == "duplicate_and_invalid_refs":
        for s in segs:
            s["paragraph_refs"] = s["paragraph_refs"] + s["paragraph_refs"][:1] + ["p0", f"p{n + 7}", "zz"]
        segs.append({"local_id": "seg_dup", "paragraph_refs": [f"p{i}" for i in range(1, min(n, 12) + 1)],
                     "role": roles[0], "confidence": "low"})
        segs.insert(0, {"local_id": "seg_all", "paragraph_refs": [f"p{i}" for i in range(n, 0, -3)],
                        "role": roles[-1], "confidence": "low"})
    elif scenario == "shuffled_fragments":
        segs = [{"local_id": f"seg_{i}", "paragraph_refs": [f"p{i}"], "role": rng.choice(roles),
                 "confidence": "low"} for i in range(1, n + 1)]
        big = [f"p{i}" for i in range(1, n + 1, 3)]
        rng.shuffle(big)
        segs.append({"local_id": "seg_shuffled_refs", "paragraph_refs": big, "role": roles[0],
                     "confidence": "low"})
        rng.shuffle(segs)
    subject = cands[0] if cands else "deneme_konusu"
    last = [f"p{i}" for i in range(max(1, n - 2), n + 1)]
    caps = [{"outcome": C.OUTCOME_BY_KIND[kind][0], "opinion_type": "majority", "dissent_authors": [],
             "subject_id": subject, "conclusion_sentence": "Mahkeme başvurunun reddine karar vermiştir.",
             "reasoning_summary": "Mahkeme, dosyadaki bilgi ve belgeler ile ilgili mevzuat hükümleri "
                                  "birlikte değerlendirildiğinde talebin yerinde olmadığı sonucuna varmıştır.",
             "supporting_paragraph_refs": last + ["p0", f"p{n + 9}", "zz"]},
            {"outcome": C.OUTCOME_BY_KIND[kind][1], "opinion_type": "dissent", "dissent_authors": ["Yılmaz"],
             "subject_id": subject, "conclusion_sentence": "Karşı oy yazısında başvurunun kabulü gerektiği belirtilmiştir.",
             "reasoning_summary": "Karşı oy yazısına göre ilgili mevzuat hükümleri başvurucunun lehine "
                                  "yorumlanmalı ve talebin kabulüne karar verilmeliydi.",
             "supporting_paragraph_refs": ["p1"]}]
    structure = C.structure_model_for(source, kind, cands).model_validate({"segments": segs})
    return structure, C.capsules_model_for(source, kind, cands).model_validate({"capsules": caps}).capsules


def fake_laws(scenario, paras, rng):
    """A LAWS answer broken in one specific way; None = that call failed."""
    n = len(paras)
    if scenario == "empty":
        return None

    def entry(i):
        quote = paras[i - 1][:25]
        return {"ref": f"p{i}", "cited_legislations": [
            {"law_no": "6698", "article_no": "12", "legislation_type": "statute", "verbatim_mention": quote},
            {"law_no": "153", "legislation_type": "constitution", "verbatim_mention": None},
            {"law_no": "6698", "article_no": "12", "legislation_type": "statute", "verbatim_mention": quote}]}
    items = [entry(i) for i in range(1, n + 1)]
    if scenario == "half_missing":
        items = items[::2]
    elif scenario == "duplicate_and_invalid_refs":
        items += [{"ref": r, "cited_legislations": []} for r in ("p0", f"p{n + 7}", "zz")] + items[:3]
    elif scenario == "shuffled_fragments":
        rng.shuffle(items)
    return C.LawsResponse.model_validate({"paragraphs": items})


def checker_selftest():
    """A checker that never fails proves nothing: corrupt a correct output in
    each way and require the matching invariant to fire."""
    import copy
    rec = usable("danistay")[3]
    paras, lost, _ = C.paragraphs(rec)
    chunks, _, _, _, _ = C.mechanical_document(paras, "danistay", str(rec["doc_id"]), None, None, "t", lost)
    cap = {k: None for k in CAPSULE_FIELDS} | {"supporting_chunk_ids": [chunks[0]["chunk_id"]], "outcome": "denied"}
    cases = {
        "paragraph_missing": lambda ch, cp: ch.pop(),
        "chunks_out_of_order": lambda ch, cp: ch.reverse(),
        "raw_words_missing": lambda ch, cp: ch[0].update(text=ch[0]["text"].split(" ", 1)[1]),
        "paragraph_duplicated": lambda ch, cp: ch.append(dict(ch[0], chunk_id="x", source_paragraph_ids=ch[0]["source_paragraph_ids"] + ["p1"])),
        "chunk_size": lambda ch, cp: ch[0].update(text=ch[0]["text"] * 40),
        "role_illegal": lambda ch, cp: ch[0].update(role="facts_x"),
        "constitution_with_law_no": lambda ch, cp: ch[0]["cited_legislations"].append(
            {k: None for k in CITATION_FIELDS} | {"legislation_type": "constitution", "law_no": "153"}),
        "capsule_link_broken": lambda ch, cp: cp[0].update(supporting_chunk_ids=["nope"]),
        "chunk_fields": lambda ch, cp: ch[0].update(extra=1),
    }
    ok = invariants("danistay", rec, paras, chunks, [cap]) == []
    for want, corrupt in cases.items():
        ch, cp = copy.deepcopy(chunks), [copy.deepcopy(cap)]
        corrupt(ch, cp)
        got = invariants("danistay", rec, paras, ch, cp)
        if want not in got:
            print(f"  CHECKER BLIND: '{want}' not detected (got {got})")
            ok = False
    print(f"  checker self-test: {'every corruption detected' if ok else 'FAILED'} ({len(cases)} kinds)")
    return ok


def test_stress(sources):
    import random
    if not checker_selftest():
        print("STRESS: FAIL (the checker itself is blind)")
        return 1
    rng = random.Random(7)
    failures, runs, examples = Counter(), Counter(), {}
    for source in sources:
        for rec in usable(source):
            doc_id, kind = str(rec["doc_id"]), C.kind_of(rec, source)
            paras, lost, _ = C.paragraphs(rec)
            if not paras:
                failures["no_paragraphs"] += 1
                continue
            cands = C.candidate_rights(rec) if kind == "aym_individual_application" else []
            case_no = C.compute_case_no(rec, source)
            date = C.compute_decision_date(rec, paras)
            for scenario in STRESS_SCENARIOS:
                runs[scenario] += 1
                try:
                    if scenario == "api_failed":
                        cites = C.citations_by_paragraph(fake_laws("perfect", paras, rng), len(paras), [])
                        chunks, caps, _, _, _ = C.mechanical_document(paras, source, doc_id, case_no, date,
                                                                     "api_or_parse_failed", lost, cites)
                    else:
                        parsed, fake_caps = fake_answer(scenario, source, kind, paras, cands, rng)
                        segs, alias, log = C.normalise_segments(parsed.segments, paras)
                        laws = fake_laws(scenario, paras, rng)
                        cites = C.citations_by_paragraph(laws, len(paras), log)
                        chunks, caps, _, _, _ = C.assemble(paras, source, doc_id, case_no, date, "x",
                                                           segs, alias, fake_caps, cites, log)
                        if scenario == "perfect":
                            # the same answer split into windows must give the same citations
                            windowed = cites_from([(w, laws) for w in C.laws_windows(paras)], len(paras), [])
                            if sum(map(len, windowed.values())) != sum(map(len, cites.values())):
                                broken_extra = ["windowed_citations_differ"]
                            else:
                                broken_extra = []
                    broken = invariants(source, rec, paras, chunks, caps)
                    if scenario == "perfect":
                        broken += broken_extra
                    if scenario in ("perfect", "api_failed") and not any(ch["cited_legislations"] for ch in chunks):
                        broken.append("citations_not_attached")
                except Exception as e:                     # noqa: BLE001
                    broken = [f"exception:{type(e).__name__}:{str(e)[:80]}"]
                for b in broken:
                    failures[(scenario, b)] += 1
                    examples.setdefault((scenario, b), f"{source} {doc_id[:12]}")
        print(f"  {source:12} done")
    print(f"\nruns: {dict(runs)} = {sum(runs.values())} total")
    if failures:
        for k, v in failures.most_common():
            print(f"  FAIL {k}: {v}  e.g. {examples.get(k)}")
    print("STRESS:", "PASS (0 invariant failures)" if not failures else f"FAIL ({sum(failures.values())})")
    return 1 if failures else 0


# --- rebuild: current code on the saved model answers ---------------------------

def rebuild(src_dir, dst_dir):
    src, dst = resolve(src_dir), resolve(dst_dir)
    dst.mkdir(parents=True, exist_ok=True)
    raw = raw_records(C.SOURCES)
    for path in sorted(src.glob("*_review.json")):
        source = path.stem.replace("_review", "")
        stored = json.loads((src / f"{source}.json").read_text(encoding="utf-8"))
        by_doc = {}
        for ch in stored["chunks"]:
            by_doc.setdefault(C.doc_of(ch), []).append(ch)
        chunks_all, caps_all, reviews = [], [], []
        for r in json.loads(path.read_text(encoding="utf-8")):
            doc_id = str(r["doc_id"])
            rec = raw.get((source, doc_id))
            if rec is None:
                continue
            paras, lost, _ = C.paragraphs(rec)
            kind = C.kind_of(rec, source)
            case_no = C.compute_case_no(rec, source)
            date = C.compute_decision_date(rec, paras)
            cands = C.candidate_rights(rec) if kind == "aym_individual_application" else []
            structure, capsules, laws = saved_answers(r, source, kind, cands)
            if structure is None:
                reason = r.get("fallback") or r.get("reason") or "no_model_answer"
                cites = cites_from(laws, len(paras), [])
                chunks, caps, issues, soft, log = C.mechanical_document(paras, source, doc_id, case_no, date,
                                                                        reason, lost, cites)
                review = dict(r, fallback=reason)
            else:
                segs, alias, log = C.normalise_segments(structure.segments, paras)
                cites = cites_from(laws, len(paras), log)
                chunks, caps, issues, soft, log = C.assemble(paras, source, doc_id, case_no, date,
                                                             C.resolve_subject_type(rec, source, kind),
                                                             segs, alias, capsules.capsules if capsules else [],
                                                             cites, log)
                issues = C.extraction_issues(lost) + issues
                review = dict(r)
            review.update(issues=issues, soft=soft, log=log, words_lost=sum(lost.values()),
                          paragraphs=len(paras), kind=kind, rebuilt_from=str(src_dir))
            review.pop("coverage", None)
            chunks_all += chunks
            caps_all += caps
            reviews.append(review)
        C.write_output(dst, source, chunks_all, caps_all, reviews)
        print(f"  rebuilt {source:12} {len(reviews)} docs, {len(chunks_all)} chunks, {len(caps_all)} capsules")
    print()
    before, after = measure(src), measure(dst)
    print(f"{'measure':58} {'stored':>8} {'rebuilt':>8}")
    for k in before:
        print(f"{k:58} {before[k]:>8} {after[k]:>8}")
    return 0


def measure(base):
    """The numbers the plan promised, computed from a folder's files."""
    raw = raw_records(C.SOURCES)
    m = Counter({k: 0 for k in (
        "documents", "cross-role fragment merges", "docs losing a ruling line to a merge",
        "real citations dropped by code (citation_not_in_document)", "Constitution citations with a law_no",
        "law_no the decision never states", "paragraphs missing", "paragraphs duplicated",
        "chunks over 2,000 characters", "docs with chunks out of order", "raw words missing from chunks",
        "documents with issues", "docs with no ruling chunk")})
    for path in sorted(base.glob("*_review.json")):
        source = path.stem.replace("_review", "")
        data = json.loads((base / f"{source}.json").read_text(encoding="utf-8"))
        by_doc, cdoc = {}, {}
        for ch in data["chunks"]:
            by_doc.setdefault(C.doc_of(ch), []).append(ch)
            cdoc[ch["chunk_id"]] = C.doc_of(ch)
        caps = {}
        for cap in data["reasoning_capsules"]:
            for d in {cdoc.get(i) for i in cap["supporting_chunk_ids"]} - {None}:
                caps.setdefault(d, []).append(cap)
        catchall = C.CATCHALL_ROLE.get(source, "other")
        for r in json.loads(path.read_text(encoding="utf-8")):
            doc_id = str(r["doc_id"])
            rec = raw.get((source, doc_id))
            if rec is None or "paragraphs" not in r:
                continue
            m["documents"] += 1
            paras, _, _ = C.paragraphs(rec)
            n = len(paras)
            first_pass = (r.get("raw_response") or {}).get("segments", [])
            first_role = {s["local_id"]: s["role"] for s in first_pass}
            first_refs = {s["local_id"]: s["paragraph_refs"] for s in first_pass}
            ruling_lost = False
            for x in r.get("log", []):
                if x.startswith("fragment_merged:"):
                    a, b = x.split(":")[1].split("->")
                    if first_role.get(a) != first_role.get(b):
                        m["cross-role fragment merges"] += 1
                        ruling_lost |= first_role.get(a) in C.RULING_ROLES and first_role.get(b) not in C.RULING_ROLES
                elif x.startswith("citation_not_in_document:"):
                    m["real citations dropped by code (citation_not_in_document)"] += int(x.split(":")[1])
            m["docs losing a ruling line to a merge"] += ruling_lost
            chunks = by_doc.get(doc_id, [])
            doc_digits = " ".join(paras)
            for ch in chunks:
                m["chunks over 2,000 characters"] += ch["char_length"] > lib.MAX_CHUNK_CHARS
                for c in ch["cited_legislations"]:
                    if c["legislation_type"] == "constitution" and c["law_no"]:
                        m["Constitution citations with a law_no"] += 1
                    elif c["law_no"] and not C._number_in(re.sub(r"\D", "", str(c["law_no"])) or c["law_no"], doc_digits):
                        m["law_no the decision never states"] += 1
            count, seen = Counter(), set()
            for ch in chunks:
                key = tuple(ch["source_paragraph_ids"])
                if key not in seen:
                    seen.add(key)
                    count.update(C._ref_idxs(list(key), n))
            m["paragraphs missing"] += sum(1 for i in range(1, n + 1) if count[i] == 0)
            m["paragraphs duplicated"] += sum(1 for i in range(1, n + 1) if count[i] > 1)
            firsts = [int(ch["source_paragraph_ids"][0][1:]) for ch in chunks]
            m["docs with chunks out of order"] += firsts != sorted(firsts)
            m["raw words missing from chunks"] += sum(missing_words(rec, chunks).values())
            m["docs with no ruling chunk"] += bool(chunks) and not any(c["role"] in C.RULING_ROLES for c in chunks)
            hard, _ = C.verify_document(chunks, caps.get(doc_id, []), paras)
            m["documents with issues"] += bool(hard)
    return m


# --- offline unit tests -------------------------------------------------------

def test_mock_retry():
    """Two 429s then a 200, through the real google-genai retry code with our
    retry options. Pass: one call returns the answer, 3 HTTP requests, identical
    request bodies (the retry does not change temperature or anything else)."""
    import httpx
    from google import genai
    from google.genai import types
    bodies = []
    answer = {"candidates": [{"content": {"role": "model", "parts": [{"text": "{\"ok\": true}"}]},
                              "finishReason": "STOP"}],
              "usageMetadata": {"promptTokenCount": 3, "candidatesTokenCount": 2}}

    def handler(request):
        bodies.append(request.content)
        if len(bodies) <= 2:
            return httpx.Response(429, json={"error": {"code": 429, "message": "Resource exhausted",
                                                       "status": "RESOURCE_EXHAUSTED"}})
        return httpx.Response(200, json=answer)

    opts = C.http_options(httpx_client=httpx.Client(transport=httpx.MockTransport(handler)))
    client = genai.Client(api_key="offline-test-no-network", http_options=opts)
    try:
        resp = client.models.generate_content(
            model="gemini-2.5-flash-lite", contents="x",
            config=types.GenerateContentConfig(temperature=0, seed=42, response_mime_type="application/json"))
        text = resp.text
    except Exception as e:                                 # noqa: BLE001
        text = f"raised {type(e).__name__}: {str(e)[:200]}"
    ok = text == '{"ok": true}' and len(bodies) == 3 and len(set(bodies)) == 1
    print(f"  requests {len(bodies)} | identical bodies {len(set(bodies)) == 1} | answer {text!r}")
    print("MOCK RETRY:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def test_merge():
    """Layer 5: a targeted rerun replaces only its own documents."""
    import tempfile
    raw = [r for r in usable("bam")[:3]]
    docs = {}
    for r in raw:
        paras, lost, _ = C.paragraphs(r)
        doc_id = str(r["doc_id"])
        chunks, _, _, _, _ = C.mechanical_document(paras, "bam", doc_id, C.compute_case_no(r, "bam"), None, "t", lost)
        cap = {k: None for k in CAPSULE_FIELDS} | {"supporting_chunk_ids": [chunks[0]["chunk_id"]], "outcome": "denied"}
        docs[doc_id] = (chunks, [cap], {"doc_id": doc_id, "paragraphs": len(paras)})
    a, b, c = list(docs)
    tmp = Path(tempfile.mkdtemp())
    ok = True
    all_chunks = [x for d in docs.values() for x in d[0]]
    C.write_output(tmp, "bam", all_chunks, [x for d in docs.values() for x in d[1]], [d[2] for d in docs.values()])
    before = json.loads((tmp / "bam.json").read_text(encoding="utf-8"))
    # rerun of b only: new text marker so replacement is visible; flushed twice as run() does per document
    new_b = [dict(x, confidence="high") for x in docs[b][0]]
    new_cap = dict(docs[b][1][0], outcome="affirmed")
    for _ in range(2):
        C.write_output(tmp, "bam", new_b, [new_cap], [dict(docs[b][2], rerun=True)], ran={b})
    after = json.loads((tmp / "bam.json").read_text(encoding="utf-8"))
    reviews = json.loads((tmp / "bam_review.json").read_text(encoding="utf-8"))
    keep = lambda data, d: [x for x in data["chunks"] if C.doc_of(x) == d]
    ok &= keep(after, a) == keep(before, a) and keep(after, c) == keep(before, c)
    ok &= keep(after, b) == new_b
    ok &= len(after["chunks"]) == len(all_chunks)
    ok &= sorted(x["outcome"] for x in after["reasoning_capsules"]) == ["affirmed", "denied", "denied"]
    ok &= sorted(r["doc_id"] for r in reviews) == sorted(docs) and [r for r in reviews if r.get("rerun")][0]["doc_id"] == b
    # targeted run into an empty folder
    tmp2 = Path(tempfile.mkdtemp())
    C.write_output(tmp2, "bam", new_b, [new_cap], [docs[b][2]], ran={b})
    ok &= len(json.loads((tmp2 / "bam.json").read_text(encoding="utf-8"))["chunks"]) == len(new_b)
    print(f"  other documents unchanged, rerun document replaced once, capsules and reviews not duplicated")
    print("MERGE:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def test_windows_and_corrections():
    """F1: laws windows cover every paragraph exactly once within the limits.
    F3: each slip code can see names the request that caused it."""
    ok = True
    paras = ["x" * 100] * 250 + ["y" * 40000] + ["z" * 700] * 70
    w = C.laws_windows(paras)
    covered = [i for lo, hi in w for i in range(lo, hi + 1)]
    ok &= covered == list(range(1, len(paras) + 1))
    ok &= all(hi - lo + 1 <= C.LAWS_WINDOW_PARAS for lo, hi in w)
    ok &= all(sum(len(q) for q in paras[lo - 1:hi]) <= C.LAWS_WINDOW_CHARS or lo == hi for lo, hi in w)
    ok &= C.laws_windows([]) == [] and C.laws_windows(["a"]) == [(1, 1)]
    chunks = [{"role": "dissent", "source_paragraph_ids": ["p7", "p8"]},
              {"role": "dissent", "source_paragraph_ids": ["p9"]},
              {"role": "facts", "source_paragraph_ids": ["p1"]}]
    fx = C.corrections_for({"issues": ["reasoning_summary_not_in_decision_language:en!=tr:majority",
                                       "no_ruling_chunk", "dissent_chunk_without_opinion_capsule:2"]}, chunks)
    ok &= set(fx) == {"capsules", "structure"} and "Turkish" in fx["capsules"] and "p7-p9" in fx["capsules"]
    ok &= C.corrections_for({"issues": ["no_ruling_chunk"], "fallback": "api_or_parse_failed"}, chunks) == {}
    ok &= C.corrections_for({"issues": ["paragraph_overlap:p3"]}, chunks) == {}
    print(f"  laws windows for {len(paras)} paragraphs: {len(w)} | corrections asked of: {sorted(fx)}")
    print("WINDOWS + CORRECTIONS:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def test_merge_rule():
    """Fix 2 on a hand-made answer: short lines join only a same-role neighbour."""
    paras = ["DAVACI : Ahmet", "SUÇ : Mala zarar verme", "Sanık hakkında kamu davası açılmış olup yapılan "
             "yargılama sonunda mahkumiyet hükmü kurulmuştur.", "karar verilmiştir.",
             "Temyiz itirazlarının reddine, hükmün ONANMASINA oybirliğiyle karar verildi."]
    m = C.structure_model_for("yargitay", "yargitay_ceza")
    parsed = m.model_validate({"segments": [
        {"local_id": "seg_1", "paragraph_refs": ["p1"], "role": "other", "confidence": "high"},
        {"local_id": "seg_2", "paragraph_refs": ["p2"], "role": "facts", "confidence": "high"},
        {"local_id": "seg_3", "paragraph_refs": ["p3"], "role": "facts", "confidence": "high"},
        {"local_id": "seg_5", "paragraph_refs": ["p5"], "role": "conclusion", "confidence": "high"},
        {"local_id": "seg_4", "paragraph_refs": ["p4"], "role": "conclusion", "confidence": "high"}]})
    segs, alias, log = C.normalise_segments(parsed.segments, paras)
    got = [(s.local_id, s.paragraph_refs, s.role) for s in segs]
    want = [("seg_1", ["p1"], "other"), ("seg_3", ["p2", "p3"], "facts"), ("seg_5", ["p4", "p5"], "conclusion")]
    ok = got == want and alias == {"seg_2": "seg_3", "seg_4": "seg_5"}
    print(f"  {got}\n  alias {alias}")
    print("MERGE RULE:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


# ---------------------------------------------------------------------------

# --- scorecard: the same numbers on every run, side by side --------------------

# A measurement heuristic for the scorecard only (it decides nothing that is stored).
LAW_MENTION_RE = re.compile(
    r"\b\d+\s*['’]?\s*(?:inci|ıncı|uncu|üncü|nci|ncı|ncu|ncü|\.)?\s*madde|"
    r"sayılı\s+(?:[^\s.;,]+\s+){0,8}?(?:Kanun|Yasa|KHK|Kararname|Yönetmelik|Tebliğ)", re.IGNORECASE)
SCORE_ROWS = [  # (key, label, which direction is better)
    ("documents", "documents", None),
    ("chunks", "chunks", None),
    ("capsules", "capsules", None),
    ("citations", "citation rows stored (duplicates included)", None),
    ("distinct_articles", "distinct law articles cited (per document)", "up"),
    ("segments_law_uncited", "segments naming a law/article with NO citation", "down"),
    ("laws_named_not_cited", "'NNNN sayılı Kanun' in the text, never cited", "down"),
    ("no_ruling", "documents without a ruling chunk", "down"),
    ("dissent_without_capsule", "documents: dissent chunk, no dissent capsule", "down"),
    ("separate_without_dissent", "documents: dissent capsule, no dissent chunk", "down"),
    ("tiny_chunks", "tiny chunks (<40 chars, not a heading)", "down"),
    ("header_not_catchall", "documents whose first chunk is not catch-all", "down"),
    ("capsule_language", "capsule fields not in the decision's language", "down"),
    ("outcome_other", "capsules with outcome 'other'", "down"),
    ("paragraphs_missing", "paragraphs missing", "down"),
    ("paragraphs_duplicated", "paragraphs duplicated", "down"),
    ("raw_words_missing", "raw words missing from chunks", "down"),
    ("fallback", "documents stored without a structure answer", "down"),
    ("laws_failed", "documents whose laws call failed", "down"),
    ("with_issues", "documents with any issue", "down"),
]


def score(base):
    """{source: Counter of SCORE_ROWS} computed from a run folder's files only."""
    raw = raw_records(C.SOURCES)
    per = {}
    for path in sorted(base.glob("*_review.json")):
        source = path.stem.replace("_review", "")
        if source not in C.SOURCES or not (base / f"{source}.json").is_file():
            continue
        data = json.loads((base / f"{source}.json").read_text(encoding="utf-8"))
        reviews = {str(r.get("doc_id")): r for r in json.loads(path.read_text(encoding="utf-8"))}
        by_doc, cdoc, caps = {}, {}, {}
        for ch in data["chunks"]:
            by_doc.setdefault(C.doc_of(ch), []).append(ch)
            cdoc[ch["chunk_id"]] = C.doc_of(ch)
        for cap in data["reasoning_capsules"]:
            for d in {cdoc.get(i) for i in cap["supporting_chunk_ids"]} - {None}:
                caps.setdefault(d, []).append(cap)
        m = per.setdefault(source, Counter({k: 0 for k, _, _ in SCORE_ROWS}))
        catchall = C.CATCHALL_ROLE.get(source, "other")
        for doc_id, chunks in by_doc.items():
            rec = raw.get((source, doc_id))
            if rec is None:
                continue
            paras, lost, _ = C.paragraphs(rec)
            n, dcaps, r = len(paras), caps.get(doc_id, []), reviews.get(doc_id, {})
            m["documents"] += 1
            m["chunks"] += len(chunks)
            m["capsules"] += len(dcaps)
            m["citations"] += sum(len(c["cited_legislations"]) for c in chunks)
            m["distinct_articles"] += len({(c["legislation_type"] == "constitution",
                                            (re.match(r"\d+", c["article_no"]) or [c["article_no"]])[0])
                                           for ch in chunks for c in ch["cited_legislations"] if c["article_no"]})
            groups = {}
            for c in chunks:
                groups.setdefault(tuple(c["source_paragraph_ids"]), []).append(c)
            m["segments_law_uncited"] += sum(
                1 for g in groups.values()
                if LAW_MENTION_RE.search(" ".join(x["text"] for x in g)) and not any(x["cited_legislations"] for x in g))
            hard, soft = C.verify_document(chunks, dcaps, paras)
            for note in soft:
                if note.startswith("laws_named_not_cited:"):
                    m["laws_named_not_cited"] += len(note.split(":", 1)[1].split(","))
                elif note.startswith("fragment_chunk:"):
                    m["tiny_chunks"] += 1
            m["with_issues"] += bool(hard or C.extraction_issues(lost))
            m["no_ruling"] += not any(c["role"] in C.RULING_ROLES for c in chunks)
            dissent_chunk = any(c["role"] in C.DISSENT_ROLES for c in chunks)
            separate_cap = any(C.is_separate(c["opinion_type"]) for c in dcaps)
            m["dissent_without_capsule"] += dissent_chunk and not separate_cap
            m["separate_without_dissent"] += separate_cap and not dissent_chunk
            first = min(chunks, key=lambda c: int(c["source_paragraph_ids"][0][1:]))
            m["header_not_catchall"] += first["role"] != catchall
            doc_lang = C.language_of(" ".join(paras))
            for cap in dcaps:
                for f in ("conclusion_sentence", "reasoning_summary"):
                    m["capsule_language"] += bool(doc_lang and C.language_of(cap[f]) != doc_lang)
                m["outcome_other"] += cap["outcome"] == "other"
            count = Counter()
            for key in groups:
                count.update(C._ref_idxs(list(key), n))
            m["paragraphs_missing"] += sum(1 for i in range(1, n + 1) if count[i] == 0)
            m["paragraphs_duplicated"] += sum(1 for i in range(1, n + 1) if count[i] > 1)
            m["raw_words_missing"] += sum(missing_words(rec, chunks).values())
            m["fallback"] += bool(r.get("fallback"))
            m["laws_failed"] += any(str(x).startswith("laws_call_failed") for x in r.get("issues") or [])
    return per


def scorecard(dir_a, dir_b=None):
    a = score(resolve(dir_a))
    b = score(resolve(dir_b)) if dir_b else None

    def total(per):
        t = Counter({k: 0 for k, _, _ in SCORE_ROWS})
        for m in per.values():
            t.update(m)
        return t
    ta, tb = total(a), (total(b) if b is not None else None)
    print(f"scorecard  THIS RUN: {resolve(dir_a)}" + (f"\n           COMPARED: {resolve(dir_b)}" if b is not None else ""))
    print(f"\n{'measure':50} {'THIS RUN':>9}" + (f" {'COMPARED':>9}  verdict" if b is not None else ""))
    for key, label, better in SCORE_ROWS:
        line = f"{label:50} {ta[key]:>9}"
        if b is not None:
            va, vb = ta[key], tb[key]
            verdict = "" if better is None or va == vb else ("better" if (va > vb) == (better == "up") else "WORSE")
            line += f" {vb:>9}  {verdict}"
        print(line)
    keys = [("documents", "docs"), ("distinct_articles", "articles"), ("segments_law_uncited", "uncited"),
            ("no_ruling", "no_ruling"), ("dissent_without_capsule", "dis_nocap"), ("tiny_chunks", "tiny"),
            ("header_not_catchall", "header"), ("capsule_language", "lang"), ("with_issues", "issues")]
    print("\nper court" + (" (this run/compared)" if b is not None else "") + ":")
    print(f"{'court':13}" + "".join(f"{h:>12}" for _, h in keys))
    for src in sorted(set(a) | set(b or {})):
        ma, mb = a.get(src, Counter()), (b or {}).get(src, Counter())
        print(f"{src:13}" + "".join(f"{(str(ma[k]) + ('/' + str(mb[k]) if b is not None else '')):>12}" for k, _ in keys))
    return 0


KIND_SOURCE = {"aym_individual_application": "aym", "aym_norm_review": "aym", "bam": "bam",
               "danistay": "danistay", "first_degree": "first_degree",
               "yargitay_hukuk": "yargitay", "yargitay_ceza": "yargitay", "kvkk": "kvkk"}


def pick(n, out_name, seed=11, kinds=None, exclude=None):
    """A seeded, stratified run set: n documents per kind, spread across years,
    the largest document of each kind included once. Written to output/<out_name>.
    Worked-example documents are left out so an evaluation set never grades one."""
    import random
    exclude = set(exclude or [])
    rng = random.Random(seed)
    chosen = {}
    for kind in kinds or KIND_SOURCE:
        source = KIND_SOURCE[kind]
        recs = [r for r in usable(source) if C.kind_of(r, source) == kind
                and not prompts.is_example_document(source, C.compute_case_no(r, source), str(r["doc_id"]))
                and str(r["doc_id"]) not in exclude]
        if not recs:
            continue
        by_year = {}
        for r in recs:
            by_year.setdefault(r.get("karar_year") or r.get("esas_year") or 0, []).append(r)
        years = sorted(by_year)
        rng.shuffle(years)
        take = []
        biggest = max(recs, key=lambda r: len(r.get("content_text") or "") + len(r.get("html_content") or ""))
        take.append(biggest)
        while len(take) < min(n, len(recs)):
            for y in years:
                pool = [r for r in by_year[y] if r not in take]
                if pool:
                    take.append(rng.choice(pool))
                if len(take) >= min(n, len(recs)):
                    break
        chosen.setdefault(source, []).extend(str(r["doc_id"]) for r in take)
        print(f"  {kind:28} {len(take)} docs")
    out = C.ROOT / "output" / out_name
    out.write_text(json.dumps(chosen, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"wrote {out} ({sum(len(v) for v in chosen.values())} documents)")
    return 0


def report(out_dir):
    """One compact line per document: role sequence, capsules, notes. What to
    read after a run instead of dumping chunks by hand."""
    base = resolve(out_dir)
    for path in sorted(base.glob("*_review.json")):
        source = path.stem.replace("_review", "")
        data = json.loads((base / f"{source}.json").read_text(encoding="utf-8"))
        by_doc = {}
        for ch in data["chunks"]:
            by_doc.setdefault(C.doc_of(ch), []).append(ch)
        cdoc = {ch["chunk_id"]: C.doc_of(ch) for ch in data["chunks"]}
        caps = {}
        for cap in data["reasoning_capsules"]:
            for d in {cdoc.get(i) for i in cap["supporting_chunk_ids"]} - {None}:
                caps.setdefault(d, []).append(cap)
        roles_total = Counter()
        print(f"\n=== {source}")
        for r in json.loads(path.read_text(encoding="utf-8")):
            doc = r["doc_id"]
            if "paragraphs" not in r:
                print(f"  {doc[:12]:12} FAILED: {r.get('reason')}: {str(r.get('error', ''))[:90]}")
                continue
            chs = sorted(by_doc.get(doc, []), key=lambda c: int(c["source_paragraph_ids"][0][1:]))
            seq, last = [], None
            for c in chs:
                roles_total[c["role"]] += 1
                if c["role"] == last:
                    seq[-1] = (last, seq[-1][1] + 1)
                else:
                    seq.append((c["role"], 1))
                    last = c["role"]
            seq_s = ",".join(f"{a}×{k}" if k > 1 else a for a, k in seq)
            outs = "; ".join(f"{c['opinion_type'].split(':')[0][:5]}={c['outcome']}" for c in caps.get(doc, []))
            notes = [x.split(":")[0] for x in r.get("log", []) if x.startswith((
                "retry", "laws_", "uncovered_filled", "dangling", "fallback_no_model", "noncontiguous_segment"))]
            notes = ",".join(f"{k}×{v}" if v > 1 else k for k, v in Counter(notes).items())
            print(f"  {doc[:12]:12} {r.get('kind', ''):27} p{r['paragraphs']:<4} {seq_s}")
            print(f"  {'':12} {'':27} caps: {outs}" + (f"  | {notes}" if notes else "") + (f"  | ISSUES: {'; '.join(r['issues'])[:120]}" if r.get("issues") else ""))
        print(f"  roles: {dict(roles_total.most_common())}")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--schema", action="store_true")
    ap.add_argument("--extraction", action="store_true")
    ap.add_argument("--stress", action="store_true", help="every usable record x 6 broken answers, offline")
    ap.add_argument("--mock-retry", action="store_true", help="offline: 429, 429, 200 through the SDK retry")
    ap.add_argument("--merge-test", action="store_true", help="offline: targeted rerun merge + fragment merge rule")
    ap.add_argument("--source")
    ap.add_argument("--verify", metavar="DIR")
    ap.add_argument("--replay", metavar="DIR")
    ap.add_argument("--rebuild", metavar="DIR", help="current code on the saved answers of DIR -> --out-dir")
    ap.add_argument("--out-dir", help="target folder for --rebuild")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--live", action="store_true", help="allow API calls (--smoke, --schema-online)")
    ap.add_argument("--schema-online", action="store_true",
                    help="PAID (under a cent): ask the API to accept every kind's structure and laws schema")
    ap.add_argument("--pick", type=int, metavar="N", help="write a seeded stratified run set: N docs per kind")
    ap.add_argument("--out", default="eval_set.json", help="file name under output/ for --pick")
    ap.add_argument("--exclude", metavar="FILE", help="run-set JSON whose doc_ids --pick must avoid")
    ap.add_argument("--report", metavar="DIR", help="compact per-document report of stored output")
    ap.add_argument("--scorecard", metavar="DIR", help="the same quality numbers for every run")
    ap.add_argument("--compare", metavar="DIR", help="with --scorecard: an earlier run folder to compare against")
    a = ap.parse_args()
    srcs = [a.source] if a.source else COURTS
    if a.pick:
        excl = []
        if a.exclude:
            excl = [d for v in json.loads(Path(a.exclude).read_text(encoding="utf-8")).values() for d in v]
        return pick(a.pick, a.out, exclude=excl)
    if a.report:
        return report(a.report)
    if a.scorecard:
        return scorecard(a.scorecard, a.compare)
    if a.schema:
        return test_schema()
    if a.stress:
        return test_stress(srcs)
    if a.mock_retry:
        return test_mock_retry()
    if a.merge_test:
        return test_merge_rule() | test_merge() | test_windows_and_corrections()
    if a.rebuild:
        if not a.out_dir:
            print("--rebuild needs --out-dir")
            return 2
        return rebuild(a.rebuild, a.out_dir)
    if a.schema_online:
        if not a.live:
            print("this needs --live (it calls the API)")
            return 2
        from dotenv import load_dotenv
        load_dotenv(C.ROOT / ".env")
        pairs = [(KIND_SOURCE[k], k) for k in KIND_SOURCE]
        problems = C.schema_preflight(C.build_client(), os.getenv("MODEL", "gemini-2.5-flash-lite"), pairs)
        for pr in problems:
            print("  REJECTED:", pr)
        print("schema online:", "PASS" if not problems else f"FAIL ({len(problems)})")
        return 1 if problems else 0
    if a.extraction:
        return test_extraction(srcs)
    if a.verify:
        return test_verify(a.verify)
    if a.replay:
        return test_replay(a.replay)
    if a.smoke:
        if not a.live:
            print("this needs --live (it calls the API)")
            return 2
        C.run(["aym", "kvkk"], 0, SMOKE, "output/smoke")
        return test_verify("output/smoke")
    ap.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
