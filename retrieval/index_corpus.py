"""Build both arms: embed the corpus and write it into Qdrant.

  Arm A  docs_baseline   every decision as one point          (production-faithful)
  Arm B  chunks_ours     the same corpus, but the decisions we have chunked are
                         replaced by their chunks

Arm B holds both units on purpose. It is what production would look like the day
chunks ship for part of the corpus, and it keeps the two arms differing in exactly
one respect: how the 12 chunked decisions are represented. Every other document is
byte-identical between the arms, which is what makes a score difference
attributable to chunking rather than to the record builder.

Embedding is the slow part and it is cached to disk by text hash, so the cost is
paid once. Re-running is then nearly free, and point ids are the record uuids, so
re-running overwrites rather than duplicating.

    python index_corpus.py --limit 40      # quick smoke pass
    python index_corpus.py                 # full corpus
    python index_corpus.py --recreate      # clean rebuild
"""

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "scripts" / "llm_chunk"))
sys.path.insert(0, str(ROOT / "scripts" / "retrieval_test"))
sys.path.insert(0, str(HERE))

import chunk_generate as gen        # noqa: E402
import contract                     # noqa: E402
import qdrant_index as qi           # noqa: E402
from embed import Embedder          # noqa: E402
from to_index_records import build_all as build_chunk_records  # noqa: E402


def load_documents(limit=None):
    """(source, doc_id, record, text) for every decision that has text.

    Text is the joined paragraph list — the same view the chunker saw — so the
    baseline reads exactly the same characters our chunks were cut from. Using
    the raw field instead would make the arms differ in their input, not just
    their unit.
    """
    out = []
    for path in sorted((ROOT / "data").glob("*.json")):
        if ".bak" in path.name or path.stem.startswith("documents_document"):
            continue
        source = path.stem
        if source not in gen.SOURCES:
            continue
        for rec in json.loads(path.read_text(encoding="utf-8")):
            try:
                paragraphs = gen.extract_paragraphs(rec, source)
            except NotImplementedError:
                continue            # rekabet / uyusmazlik: no text upstream
            if paragraphs:
                out.append((source, str(rec.get("doc_id")), rec,
                            "\n".join(paragraphs)))
            if limit and len(out) >= limit:
                return out
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, help="only the first N documents")
    ap.add_argument("--recreate", action="store_true", help="drop and rebuild")
    ap.add_argument("--batch", type=int, default=8, help="(ignored; batching is "
                                                         "planned from memory)")
    ap.add_argument("--distractor-max-tokens", type=int, default=2048,
                    help="token ceiling for haystack documents not under test. "
                         "Identical in both arms, so it cannot bias the comparison.")
    args = ap.parse_args()

    t0 = time.time()
    docs = load_documents(args.limit)
    chunk_recs, skipped = build_chunk_records()
    chunked_parents = {r["parent_uuid"] for r in chunk_recs}

    print("=" * 74)
    print(" INDEXING BOTH ARMS")
    print("=" * 74)
    print(f" documents with text : {len(docs)}")
    print(f" chunks              : {len(chunk_recs)} "
          f"covering {len(chunked_parents)} decisions")
    if skipped:
        print(f" skipped chunks      : {len(skipped)}")

    # --- build records ------------------------------------------------------
    doc_recs = [contract.build_document_record(rec, source, text)
                for source, doc_id, rec, text in docs]

    bad = [(r, p) for r in doc_recs + chunk_recs if (p := contract.validate(r))]
    if bad:
        print(f"\n CONTRACT FAILURES: {len(bad)} — nothing indexed")
        for r, p in bad[:10]:
            print(f"   {r.get('high_court')}/{r.get('uuid','?')[:12]} {p}")
        return 1
    print(f" contract            : {len(doc_recs) + len(chunk_recs)} records, 0 failures")

    # Arm B = every document we did NOT chunk, plus the chunks themselves.
    arm_b = [r for r in doc_recs if r["parent_uuid"] not in chunked_parents] + chunk_recs
    replaced = len(doc_recs) - (len(arm_b) - len(chunk_recs))
    print(f"\n Arm A (baseline)    : {len(doc_recs)} points, all whole documents")
    print(f" Arm B (ours)        : {len(arm_b)} points "
          f"= {len(arm_b) - len(chunk_recs)} documents + {len(chunk_recs)} chunks")
    print(f"   decisions replaced by their chunks: {replaced}")

    # --- embed --------------------------------------------------------------
    # TWO TIERS, for cost, and it is sound rather than a shortcut.
    #
    # The arms differ in exactly one respect: whether the chunked decisions are
    # stored whole or as chunks. Every other document is the SAME record with the
    # SAME vector in both arms, so a lower token ceiling on those distractors
    # applies identically to A and B and cannot tilt the comparison. The documents
    # actually under test keep production's full 8,192.
    #
    # Measured on this machine: full fidelity everywhere costs ~4.5 h, almost all
    # of it in the distractor tail. This is ~50 min.
    emb = Embedder()
    target_recs = [r for r in doc_recs if r["parent_uuid"] in chunked_parents]
    distractors = [r for r in doc_recs if r["parent_uuid"] not in chunked_parents]

    print(f"\n embedding, two tiers (cached by text+ceiling; first run is slow)")
    print(f"   under test  : {len(target_recs)} documents + {len(chunk_recs)} chunks "
          f"@ {emb.max_tokens} tokens (production fidelity)")
    print(f"   haystack    : {len(distractors)} distractors @ "
          f"{args.distractor_max_tokens} tokens (identical in both arms)")

    by_uuid = {}
    full = target_recs + chunk_recs
    if full:
        v = emb.embed([r["text"] for r in full], progress=True)
        by_uuid.update({r["uuid"]: vec for r, vec in zip(full, v)})
    if distractors:
        v = emb.embed([r["text"] for r in distractors], progress=True,
                      max_tokens=args.distractor_max_tokens)
        by_uuid.update({r["uuid"]: vec for r, vec in zip(distractors, v)})
    print(f" embedded            : {len(by_uuid)} vectors")

    # --- index --------------------------------------------------------------
    qc = qi.client()
    results = {}
    for name, recs, chunky in ((qi.BASELINE, doc_recs, False),
                               (qi.CHUNKS, arm_b, True)):
        wanted = qi.ensure_collection(qc, name, chunky, recreate=args.recreate)
        n = qi.upsert(qc, name, recs, [by_uuid[r["uuid"]] for r in recs])
        results[name] = qi.verify(qc, name, len(recs), wanted)
        print(f"\n {name}")
        print(f"   upserted   : {n}")
        print(f"   verified   : {results[name]['count']} points, "
              f"dim {results[name]['size']}, {results[name]['distance']}")
        print(f"   indexes ok : {len(results[name]['indexes'])} "
              f"({', '.join(results[name]['indexes'])})")
        if results[name]["problems"]:
            print("   PROBLEMS:")
            for p in results[name]["problems"]:
                print(f"     - {p}")

    ok = not any(r["problems"] for r in results.values())
    print(f"\n {'ALL CHECKS PASS' if ok else 'PROBLEMS ABOVE'}  "
          f"({time.time() - t0:.0f}s)")

    (HERE / "report" / "index_summary.json").write_text(
        json.dumps({"arm_a": len(doc_recs), "arm_b": len(arm_b),
                    "chunks": len(chunk_recs), "replaced": replaced,
                    "verify": results}, ensure_ascii=False, indent=2),
        encoding="utf-8")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
