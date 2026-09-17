"""Will these documents chunk correctly? Answer BEFORE paying the LLM.

Every failure this pipeline has had was silent. A different word for "ready"
skipped 45 documents. Yargitay's text was in the other column. Yargitay's text
produced ONE paragraph instead of many. KVKK has no esas_year, which makes it
invisible to production's search. None of those raised an error; they just
produced less than expected, quietly.

So this reads every document the way the chunker will, and reports what would go
wrong -- with no Gemini call, no cost, and no writes.

    python preflight.py                      # every source in data/
    python preflight.py --source aym
    python preflight.py --verbose            # per-document detail
    python preflight.py --pick-risky 2       # the doc_ids worth a paid smoke test

VERDICTS
  RED    do not send. It will fail, or produce nothing.
  AMBER  send, but expect flags -- usually an unfamiliar layout.
  GREEN  looks like documents we have already chunked successfully.
"""

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import chunk_generate as gen        # noqa: E402

# Roughly 4.11 characters per token, measured with bge-m3's own tokenizer on this
# corpus. Used only for cost estimation, so a rough figure is fine.
CHARS_PER_TOKEN = 4.11

# The section markers our prompt tells the model to look for, per source. If a
# document contains none of them, the prompt is describing a layout that document
# does not have -- which is exactly the risk with decisions from the 1970s.
# Stored without Turkish diacritics and matched case-insensitively against a
# diacritic-stripped copy, because the same heading appears as both "GEREKÇE" and
# "GEREKCE" depending on the scrape.
EXPECTED_MARKERS = {
    "aym": ["OLAY VE OLGULAR", "ILGILI HUKUK", "GENEL ILKELER", "DEGERLENDIRME",
            "HUKUM", "KARSIOY", "KARSI OY", "INCELEME VE GEREKCE"],
    "bam": ["DAVA", "CEVAP", "ILK DERECE", "ISTINAF", "GEREKCE", "HUKUM"],
    "danistay": ["ISTEMIN KONUSU", "YARGILAMA SURECI", "TEMYIZ EDEN",
                 "HUKUKI DEGERLENDIRME", "KARAR SONUCU", "KARSI OY"],
    "first_degree": ["DAVA", "GEREGI DUSUNULDU", "HUKUM", "GEREKCE"],
    # kvkk decisions are short narrative summaries with no headings at all, and
    # yargitay has ~46 chambers with no fixed structure -- their prompts are
    # written around function rather than markers, so marker absence is expected
    # and must not be reported as drift.
    "kvkk": [],
    "yargitay": [],
}

TR_MAP = str.maketrans("çğıöşüÇĞİÖŞÜâîû", "cgiosuCGIOSUaiu")


def deaccent(s):
    return (s or "").translate(TR_MAP)


def _finish(out, raw=""):
    """Single place the verdict is decided, so an early return cannot skip it."""
    out["verdict"] = ("RED" if out["problems"]
                      else "AMBER" if out["warnings"] else "GREEN")
    out.setdefault("input_tokens", int(len(raw) / CHARS_PER_TOKEN) if raw else 0)
    return out


def check_document(record, source):
    """Everything that can be known without calling the model."""
    out = {"doc_id": str(record.get("doc_id")), "source": source,
           "problems": [], "warnings": [], "info": {}}

    # 1. STATUS -- the word mismatch that silently skipped 45 documents
    status = record.get("status")
    out["info"]["status"] = status
    if status not in gen.USABLE_STATUSES:
        out["problems"].append(
            f"status {status!r} is not in {sorted(gen.USABLE_STATUSES)} -- "
            f"pick_documents will skip this document without saying so")

    # 2. TEXT IN THE COLUMN WE READ -- yargitay's was in the other one
    field = gen.SOURCES[source]["text_field"]
    out["info"]["text_field"] = field
    if field is None:
        out["problems"].append("no text_field configured for this source")
        return _finish(out)
    raw = record.get(field) or ""
    other = "content_text" if field == "html_content" else "html_content"
    out["info"]["chars"] = len(raw)
    if not raw.strip():
        if (record.get(other) or "").strip():
            out["problems"].append(
                f"{field} is empty but {other} has "
                f"{len(record.get(other))} chars -- reading the wrong column")
        else:
            out["problems"].append("no text in either column")
        return _finish(out)

    # 3. PARAGRAPH SPLITTING -- yargitay produced one giant blob
    try:
        paras = gen.extract_paragraphs(record, source)
    except Exception as exc:                                   # noqa: BLE001
        out["problems"].append(f"paragraph extraction raised "
                               f"{type(exc).__name__}: {str(exc)[:80]}")
        return _finish(out, raw)
    out["info"]["paragraphs"] = len(paras)
    if len(paras) <= 1:
        out["problems"].append(
            f"splitting produced {len(paras)} paragraph(s) -- the whole decision "
            f"would become one chunk, and paragraph references would be useless")
    elif len(paras) < 4:
        out["warnings"].append(f"only {len(paras)} paragraphs")

    # 3b. IS THERE ANY SUBSTANCE? -- the hollow-document check
    #
    # A KVKK decision in the smoke test read, in full:
    #   "Kurum'a intikal eden şikayette özetle;"   <- introduces a list
    #   "hususları ifade edilerek ... talep edilmiştir."   <- refers to it
    #   "... alınan cevabi yazıda özetle;"          <- introduces another
    #   "ifade edilmiştir. ... Kararı ile;"
    #   "değerlendirmelerinden hareketle;"
    #   "karar verilmiştir."
    # 519 characters of pure connective tissue: the scraper kept the sentences
    # that introduce each list and dropped every list. The model correctly
    # returned an empty summary, because there was nothing to summarise -- but the
    # chunk is useless, and we paid for it. Colon- and semicolon-terminated
    # paragraphs with nothing after them are the signature.
    body_chars = sum(len(p) for p in paras)
    dangling = sum(1 for p in paras if p.rstrip().endswith((";", ":")))
    out["info"]["body_chars"] = body_chars
    out["info"]["dangling_intros"] = dangling
    if paras and body_chars < 700 and dangling >= 2:
        out["problems"].append(
            f"hollow document: {body_chars} chars across {len(paras)} paragraphs, "
            f"{dangling} of them ending in ';' or ':' with no list after -- the "
            f"substance was lost upstream, so a summary cannot be written")
    elif paras and dangling >= max(3, len(paras) // 2):
        out["warnings"].append(
            f"{dangling}/{len(paras)} paragraphs end in ';' or ':' -- content "
            f"may be missing between them")

    # 4. DOES THE PROMPT DESCRIBE THIS DOCUMENT? -- the old-format question
    expected = EXPECTED_MARKERS.get(source, [])
    if expected:
        body = deaccent(" ".join(paras)).upper()
        found = [m for m in expected if deaccent(m).upper() in body]
        out["info"]["markers_found"] = len(found)
        out["info"]["markers_expected"] = len(expected)
        if not found:
            out["warnings"].append(
                "NONE of the section markers our prompt describes appear -- this "
                "is a layout we have not seen, so role tagging may be guesswork")
        elif len(found) == 1:
            out["warnings"].append(f"only 1 of {len(expected)} markers found "
                                   f"({found[0]})")

    # 5. YEARS -- kvkk has no esas_year, and that makes it invisible to search
    esas, karar = record.get("esas_year"), record.get("karar_year")
    out["info"]["esas_year"], out["info"]["karar_year"] = esas, karar
    if esas is None and karar is None:
        out["problems"].append(
            "both esas_year and karar_year are null -- production splits every "
            "search on esas_year, so this document matches NEITHER bucket and "
            "would never be returned")

    # 6. CHAMBER -- chamber_id is not the chamber for 5 of 6 sources
    title = record.get("title") or ""
    m = re.search(r"(\d{1,2})\s*\.\s*(?:[^\s]+\s+){0,3}?(?:Daire|Dairesi|Mahkemesi)",
                  title)
    out["info"]["chamber_from_title"] = int(m.group(1)) if m else None
    if not m and source not in ("aym", "kvkk"):
        out["warnings"].append("no chamber number in the title -- court filters "
                               "will not match this document")

    # 7. LENGTH vs THE MODEL'S OUTPUT BUDGET -- one document was already refused
    predicted = len(raw) * gen.OUTPUT_TOKENS_PER_CHAR
    out["info"]["predicted_output_tokens"] = int(predicted)
    if predicted > gen.MAX_OUTPUT_TOKENS:
        out["problems"].append(
            f"predicted output {int(predicted):,} tokens exceeds the "
            f"{gen.MAX_OUTPUT_TOKENS:,} limit -- the call would truncate")
    elif predicted > gen.MAX_OUTPUT_TOKENS * 0.75:
        out["warnings"].append(f"predicted output {int(predicted):,} tokens is "
                               f"close to the limit")

    # 8. THE METADATA THE CHUNKER DERIVES
    try:
        out["info"]["case_no"] = gen.compute_case_no(record, source)
        date, reason = gen.compute_decision_date(record, source, paras)
        out["info"]["decision_date"] = date
        if not date:
            out["warnings"].append(f"no decision_date ({reason})")
        out["info"]["subject_type"] = gen.resolve_subject_type(record, source)
    except Exception as exc:                                   # noqa: BLE001
        out["problems"].append(f"metadata derivation raised "
                               f"{type(exc).__name__}: {str(exc)[:80]}")

    if not out["info"].get("case_no"):
        out["warnings"].append("no case_no could be derived")

    return _finish(out, raw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source")
    ap.add_argument("--data-dir", help="check this directory instead of data/")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--pick-risky", type=int, default=0,
                    help="print the N riskiest doc_ids, for a paid smoke test")
    args = ap.parse_args()

    sources = [args.source] if args.source else sorted(gen.SOURCES)
    results = []
    for source in sources:
        base = Path(args.data_dir) if args.data_dir else gen.DATA_DIR
        path = base / f"{source}.json"
        if not path.is_file():
            continue
        records = json.loads(path.read_text(encoding="utf-8"))
        for rec in records:
            results.append(check_document(rec, source))

    if not results:
        print("no documents found in data/")
        return 1

    print("=" * 78)
    print(" PRE-FLIGHT — will these documents chunk correctly?")
    print("=" * 78)
    print(f" documents checked : {len(results)}   (no LLM call, nothing written)\n")

    print(f"{'source':14} {'docs':>5} {'GREEN':>6} {'AMBER':>6} {'RED':>5} "
          f"{'median ¶':>9} {'in tokens':>10}")
    by = {}
    for r in results:
        by.setdefault(r["source"], []).append(r)
    total_in = total_out = 0
    for src, rs in sorted(by.items()):
        paras = sorted(r["info"].get("paragraphs", 0) for r in rs)
        med = paras[len(paras) // 2] if paras else 0
        tin = sum(r.get("input_tokens", 0) for r in rs)
        tout = sum(r["info"].get("predicted_output_tokens", 0) for r in rs)
        total_in += tin
        total_out += tout
        print(f"{src:14} {len(rs):>5} "
              f"{sum(1 for r in rs if r['verdict'] == 'GREEN'):>6} "
              f"{sum(1 for r in rs if r['verdict'] == 'AMBER'):>6} "
              f"{sum(1 for r in rs if r['verdict'] == 'RED'):>5} "
              f"{med:>9} {tin:>10,}")

    print(f"\n ESTIMATED COST OF CHUNKING ALL OF THESE")
    print(f"   input  ~{total_in:,} tokens")
    print(f"   output ~{total_out:,} tokens  (predicted, not measured)")

    # what is actually wrong, grouped
    problems = Counter()
    warnings = Counter()
    for r in results:
        for p in r["problems"]:
            problems[p.split(" -- ")[0][:64]] += 1
        for w in r["warnings"]:
            warnings[w.split(" -- ")[0][:64]] += 1

    if problems:
        print(f"\n BLOCKING PROBLEMS (RED — do not send these):")
        for p, c in problems.most_common():
            print(f"   {c:>4}  {p}")
    else:
        print(f"\n BLOCKING PROBLEMS: none")

    if warnings:
        print(f"\n WARNINGS (AMBER — send, but expect flags):")
        for w, c in warnings.most_common(12):
            print(f"   {c:>4}  {w}")

    if args.verbose:
        print(f"\n{'verdict':8} {'source':13} {'doc_id':22} {'¶':>5} {'chars':>8}  note")
        for r in sorted(results, key=lambda x: (x["verdict"] != "RED",
                                                x["verdict"] != "AMBER")):
            note = (r["problems"] + r["warnings"] or [""])[0][:44]
            print(f"{r['verdict']:8} {r['source']:13} {r['doc_id'][:22]:22} "
                  f"{r['info'].get('paragraphs', 0):>5} "
                  f"{r['info'].get('chars', 0):>8,}  {note}")

    if args.pick_risky:
        risky = sorted(results,
                       key=lambda r: (-len(r["problems"]), -len(r["warnings"]),
                                      -r["info"].get("chars", 0)))
        print(f"\n THE {args.pick_risky} RISKIEST DOCUMENTS — chunk these first, cheaply:")
        picks = risky[:args.pick_risky]
        for r in picks:
            print(f"   {r['source']:13} {r['doc_id']}  {r['verdict']}  "
                  f"{(r['problems'] + r['warnings'] or ['looks clean'])[0][:52]}")
        ids = ",".join(r["doc_id"] for r in picks)
        srcs = sorted({r["source"] for r in picks})
        print(f"\n   python chunk_generate.py --source {srcs[0]} --doc-id {ids} "
              f"--out-dir output/smoke")

    out = gen.ROOT / "output" / "preflight.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    print(f"\n wrote {out}")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
