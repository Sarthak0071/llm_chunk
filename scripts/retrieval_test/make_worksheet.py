"""
make_worksheet.py -- Stage 1 support: one worksheet per document so queries can
be written from the RAW court text and nothing else.

Why this exists: if a query is written by reading the generated
reasoning_summary and rewording it, BM25 is matching text against a paraphrase
of itself and the test passes by construction. These worksheets contain only
the raw document, so the query author never needs to open output/chunk/.

ON PARAGRAPH NUMBERS -- read this before using them as an answer key.
  aym : [pN] IS the court's own numbered paragraph. Real citation unit.
  others: [pN] is OUR synthetic split -- splitlines() for content_text, <p>
        tags for kvkk html. Not the court's numbering. It is still sound as an
        answer key because it is deterministic CODE output, not generated
        content, so no circularity is introduced -- but it is a location
        marker, not an official paragraph reference.

Usage:
    python make_worksheet.py                 # all 10 documents
    python make_worksheet.py --source bam    # one source
"""

import argparse

import corpus as C

HEADER = """\
{bar}
 {source}   doc_id {doc_id}
 case_no {case_no}
{bar}

 PARAGRAPH NUMBERING: {numbering}

 HOW TO WRITE A QUERY
   Read the paragraphs below. Write questions a lawyer would actually ask.
   Do NOT open output/chunk/ -- these paragraphs are the only source.

   STAGE 1 asks "which case is this?"  -> a natural, factual question about
   what happened and how it was decided. Expected answer is this document.

   STAGE 2 asks "which paragraph answers this?" -> a narrower question, plus
   the paragraph number that answers it. Code resolves which chunk contains
   that paragraph, so you never pick a chunk by hand.

 Paraphrase. Do not copy distinctive phrases out of the text -- a query that
 reuses a rare phrase wins on keyword overlap and proves nothing. The audit in
 test_retrieval.py --audit-only reports exactly how much you reused.

{bar}
 WRITE QUERIES HERE  (copy into queries.json when done)
{bar}

 STAGE1_QUERY:

 STAGE2_QUERY:
 STAGE2_PARAGRAPH:      (e.g. p18)

 STAGE2_QUERY:
 STAGE2_PARAGRAPH:

{bar}
 RAW DOCUMENT -- {n} paragraphs, {chars} characters
{bar}

"""

NUMBERING_NOTE = {
    True: "[pN] is the COURT'S OWN paragraph number (aym). A real citation unit.",
    False: ("[pN] is OUR synthetic split, not the court's numbering. A location "
            "marker only."),
}


def main():
    ap = argparse.ArgumentParser(description="Build query-authoring worksheets")
    ap.add_argument("--source", choices=C.SOURCES, help="only this source")
    args = ap.parse_args()

    c = C.Corpus()
    if not c.documents:
        raise SystemExit("No generated output in output/chunk/. Run chunk_generate.py first.")

    wanted = [d for d in c.documents if args.source is None or d[0] == args.source]
    raw = C.load_raw_documents(wanted_doc_ids=set(wanted))
    C.WORKSHEET_DIR.mkdir(parents=True, exist_ok=True)
    bar = "=" * 78

    written = []
    for source, doc_id in wanted:
        doc = raw.get((source, doc_id))
        if not doc:
            print(f"  SKIP {source}/{doc_id}: raw document not found in data/")
            continue
        paras = doc["paragraphs"]
        body = "\n\n".join(f"[p{i}] {p}" for i, p in enumerate(paras, 1))
        text = HEADER.format(
            bar=bar, source=source, doc_id=doc_id,
            case_no=c.case_no_of_doc.get((source, doc_id)),
            numbering=NUMBERING_NOTE[source == "aym"],
            n=len(paras), chars=sum(len(p) for p in paras),
        ) + body + "\n"

        short = doc_id[:8] if len(doc_id) > 12 else doc_id
        path = C.WORKSHEET_DIR / f"{source}__{short}.txt"
        path.write_text(text, encoding="utf-8")
        written.append((path, len(paras)))
        print(f"  {path.name:34} {len(paras):4} paragraphs")

    print(f"\n{len(written)} worksheet(s) in {C.WORKSHEET_DIR}")
    print("Write queries into queries.json -- see queries.example.json for the format.")


if __name__ == "__main__":
    main()
