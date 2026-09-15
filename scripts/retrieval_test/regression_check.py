"""Prove that registering rekabet and uyusmazlik changed nothing for the six
sources that already work. Replays pure functions over the documents already in
output/chunk/ and demands exact equality with what is stored.

No Gemini call, no cost, no write. Run it after ANY change to the shared
functions -- it is the cheap half of the regression story, and verify_storage.py
plus test_retrieval.py are the other half.

Four things are checked, and they are exactly the four the source registration
touched:

  1. extract_paragraphs, generator vs verifier, BYTE-IDENTICAL. These are two
     independent implementations on purpose, and they must agree: chunk `text` is
     assembled from paragraph refs, so if the verifier splits differently then
     every grounding check is testing the checker rather than the data.
  2. compute_case_no           -- gained a rekabet branch
  3. compute_decision_date     -- deliberately NOT changed; asserted, not assumed
  4. resolve_subject_type      -- restructured from an early-return into dispatch

Exit code 1 on any difference.
"""

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]                                   # llm_chunk/
sys.path.insert(0, str(ROOT / "scripts" / "llm_chunk"))
sys.path.insert(0, str(HERE))

import chunk_generate as gen        # noqa: E402
import corpus as cor                # noqa: E402

OUT_DIR = ROOT / "output" / "chunk"
DATA_DIR = ROOT / "data"


def load_generated():
    """{(source, doc_id): {"chunks": [...], "capsules": [...]}} from output/chunk/."""
    out = {}
    for path in sorted(OUT_DIR.glob("*.json")):
        if path.stem.endswith("_review"):
            continue
        source = path.stem
        payload = json.loads(path.read_text(encoding="utf-8"))
        for ch in payload.get("chunks", []):
            out.setdefault((source, cor.doc_id_of(ch)),
                           {"chunks": [], "capsules": []})["chunks"].append(ch)
        # Capsules carry no chunk_label, so they are matched through the chunk
        # ids they support rather than by parsing an id out of a string.
        chunk_home = {c["chunk_id"]: cor.doc_id_of(c) for c in payload.get("chunks", [])}
        for cap in payload.get("reasoning_capsules", []):
            homes = {chunk_home.get(cid) for cid in cap.get("supporting_chunk_ids", [])}
            homes.discard(None)
            if len(homes) == 1:
                out[(source, homes.pop())]["capsules"].append(cap)
    return out


def load_records():
    """{(source, doc_id): raw record} for every source with a data file."""
    out = {}
    for path in sorted(DATA_DIR.glob("*.json")):
        if ".bak" in path.name or path.stem.startswith("documents_document"):
            continue
        source = path.stem
        if source not in gen.SOURCES:
            continue
        for rec in json.loads(path.read_text(encoding="utf-8")):
            out[(source, str(rec.get("doc_id")))] = rec
    return out


def main():
    generated = load_generated()
    records = load_records()
    if not generated:
        print("no generated output in output/chunk/ -- nothing to regress against")
        return 1

    failures, checked = [], {"paragraphs": 0, "case_no": 0, "date": 0, "subject": 0}

    for (source, doc_id), got in sorted(generated.items()):
        rec = records.get((source, str(doc_id)))
        if rec is None:
            failures.append(f"{source}/{doc_id}: generated but no longer in data/")
            continue

        # 1. the two independent paragraph extractors must agree exactly
        p_gen = gen.extract_paragraphs(rec, source)
        p_cor = cor.extract_paragraphs(rec, source)
        checked["paragraphs"] += 1
        if p_gen != p_cor:
            first = next((i for i, (a, b) in enumerate(zip(p_gen, p_cor)) if a != b),
                         min(len(p_gen), len(p_cor)))
            failures.append(
                f"{source}/{doc_id}: paragraph extractors DISAGREE "
                f"(generator {len(p_gen)} paras, verifier {len(p_cor)}); "
                f"first difference at index {first}")
            continue

        # 2. case_no, as stored on every chunk of this document
        want = gen.compute_case_no(rec, source)
        for ch in got["chunks"]:
            checked["case_no"] += 1
            if ch.get("case_no") != want:
                failures.append(f"{source}/{doc_id}: case_no now {want!r}, "
                                f"stored {ch.get('case_no')!r}")
                break

        # 3. decision_date -- this function was deliberately left alone
        want_date, _ = gen.compute_decision_date(rec, source, p_gen)
        for ch in got["chunks"]:
            checked["date"] += 1
            if ch.get("decision_date") != want_date:
                failures.append(f"{source}/{doc_id}: decision_date now {want_date!r}, "
                                f"stored {ch.get('decision_date')!r}")
                break

        # 4. subject_type, as stored on the capsules
        want_subj = gen.resolve_subject_type(rec, source)
        for cap in got["capsules"]:
            checked["subject"] += 1
            if cap.get("subject_type") != want_subj:
                failures.append(f"{source}/{doc_id}: subject_type now {want_subj!r}, "
                                f"stored {cap.get('subject_type')!r}")
                break

    docs = len(generated)
    srcs = sorted({s for s, _ in generated})
    print(f"documents replayed : {docs}  across {len(srcs)} sources ({', '.join(srcs)})")
    print(f"paragraph extractor agreement : {checked['paragraphs']} documents")
    print(f"field comparisons  : case_no {checked['case_no']}, "
          f"decision_date {checked['date']}, subject_type {checked['subject']}")

    if failures:
        print(f"\nFAIL -- {len(failures)} regression(s):")
        for f in failures:
            print(f"  {f}")
        return 1
    print("\nPASS -- every replayed field is identical to what is stored.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
