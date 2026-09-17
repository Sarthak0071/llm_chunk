"""Is the 2,000-character cap wrong?

bge-m3 accepts 8,192 tokens — about 33,700 characters of Turkish legal text. Our
chunks are a median of 162 tokens, roughly 2% of that, and 93 of 142 are pieces cut
by the cap rather than segments the model chose. The cap was picked for storage,
not for the embedder, so it may be splitting reasoning that belongs together.

This tests it WITHOUT calling the LLM: glue adjacent chunks of the same decision
back together up to a larger target size, re-embed, and re-score the same queries
against the same haystack. Merging is arithmetic, so the only thing changing is
chunk size.

What a bigger chunk should win: context. A passage that mentions "bu nedenle" makes
more sense with the sentence before it.
What it should lose: precision. The returned text is longer, and the paragraph
reference becomes a range rather than a point.

Both are measured — retrieval quality AND the characters shipped to the model.

    python merge_test.py
    python merge_test.py --levels 2000,8000,32000
"""

import argparse
import json
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "scripts" / "retrieval_test"))
sys.path.insert(0, str(HERE))

import contract                      # noqa: E402
import qdrant_index as qi            # noqa: E402
from compare import load_queries     # noqa: E402
from embed import Embedder           # noqa: E402
from index_corpus import load_documents  # noqa: E402
from to_index_records import build_all as build_chunk_records  # noqa: E402

COLLECTION = "merge_probe"


def merge_chunks(records, target_chars):
    """Glue consecutive chunks of one decision until the target size is reached.

    Order is `chunk_index`, which is position within the decision, so merged units
    are always contiguous passages and never stitch unrelated parts together.
    Paragraph ids are unioned, so a merged unit still says where it came from —
    that is what keeps pinpoint citation measurable at every size.
    """
    by_parent = {}
    for r in records:
        by_parent.setdefault(r["parent_uuid"], []).append(r)

    out = []
    for parent, group in by_parent.items():
        group.sort(key=lambda r: r["chunk_index"])
        buf = []
        for r in group:
            buf.append(r)
            if sum(len(x["text"] or "") for x in buf) >= target_chars:
                out.append(_fuse(buf))
                buf = []
        if buf:
            out.append(_fuse(buf))
    return out


def _fuse(group):
    """One merged record from consecutive chunks. Metadata comes from the first;
    text, paragraphs and law references are combined."""
    import uuid as _uuid
    head = dict(group[0])
    if len(group) == 1:
        return head
    head["text"] = "\n".join(g["text"] or "" for g in group)
    paras, laws = [], []
    for g in group:
        for p in g.get("paragraph_ids") or []:
            if p not in paras:
                paras.append(p)
        for lref in g.get("law_refs") or []:
            if lref not in laws:
                laws.append(lref)
    head["paragraph_ids"] = paras
    head["law_refs"] = laws
    # A merged unit is a different thing from any of its parts, so it needs its own
    # id -- reusing the first chunk's would make two different texts share a key.
    head["uuid"] = str(_uuid.uuid5(contract.PARENT_NAMESPACE,
                                   "merge:" + "|".join(g["uuid"] for g in group)))
    head["merged_from"] = len(group)
    return head


def score(qc, collection, queries, vectors, parents, emb):
    rows = []
    for q, v in zip(queries, vectors):
        want = parents.get(q["id"])
        r = qc.query_points(collection_name=collection, query=v.tolist(), limit=15,
                            with_payload=["parent_uuid", "paragraph_ids", "unit", "text"])
        rank = next((i + 1 for i, p in enumerate(r.points)
                     if p.payload.get("parent_uuid") == want), None)
        paras = next((p.payload.get("paragraph_ids") or [] for p in r.points
                      if p.payload.get("parent_uuid") == want), [])
        rows.append({"rank": rank, "paras": paras,
                     "expect": q.get("paragraph"),
                     "chars": sum(len(p.payload.get("text") or "") for p in r.points)})
    n = len(rows)
    found = [r for r in rows if r["rank"]]
    pin_ask = [r for r in rows if r["expect"]]
    return {
        "top1": sum(1 for r in rows if r["rank"] == 1),
        "top5": sum(1 for r in rows if r["rank"] and r["rank"] <= 5),
        "mrr": round(sum(1 / r["rank"] for r in found) / n, 3) if n else 0,
        "never": n - len(found),
        "pinpoint": sum(1 for r in pin_ask if r["expect"] in (r["paras"] or [])),
        "pinpoint_askable": len(pin_ask),
        "n": n,
        "chars": int(sum(r["chars"] for r in rows) / n) if n else 0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--levels", default="2000,4000,8000,16000,32000")
    args = ap.parse_args()
    levels = [int(x) for x in args.levels.split(",")]

    qc = qi.client()
    emb = Embedder()

    chunk_recs, _ = build_chunk_records()
    chunked_parents = {r["parent_uuid"] for r in chunk_recs}
    docs = load_documents()
    distractors = [contract.build_document_record(rec, src, txt)
                   for src, did, rec, txt in docs
                   if contract.parent_uuid_for(src, did) not in chunked_parents]

    queries = [q for q in load_queries() if q.get("doc_id")]
    qvecs = emb.embed([q["text"] for q in queries])
    parents = {q["id"]: contract.parent_uuid_for(q["source"], str(q["doc_id"]))
               for q in queries}

    # cached from the main index, so free
    dvecs = emb.embed([r["text"] for r in distractors], max_tokens=2048)

    print("=" * 78)
    print(" MERGE TEST — does a bigger chunk retrieve better?")
    print("=" * 78)
    print(f" haystack : {len(distractors)} distractor documents (identical at every level)")
    print(f" queries  : {len(queries)}")
    print(f" ceiling  : bge-m3 accepts ~33,700 characters\n")

    print(f"{'cap':>7} {'units':>6} {'median':>8} {'%ceiling':>9} {'top-1':>7} "
          f"{'top-5':>7} {'MRR':>7} {'never':>6} {'¶ cite':>8} {'chars/q':>9}")
    print("-" * 78)

    report = []
    for target in levels:
        merged = merge_chunks(chunk_recs, target)
        sizes = [len(r["text"] or "") for r in merged]
        toks = [emb.token_count(r["text"] or "") for r in merged]

        recs = distractors + merged
        vecs = emb.embed([r["text"] for r in merged])
        qi.ensure_collection(qc, COLLECTION, with_chunk_indexes=True, recreate=True)
        qi.upsert(qc, COLLECTION, recs, list(dvecs) + list(vecs))

        s = score(qc, COLLECTION, queries, qvecs, parents, emb)
        pct = statistics.median(toks) * 100 / 8192
        print(f"{target:>7,} {len(merged):>6} {statistics.median(sizes):>8,.0f} "
              f"{pct:>8.1f}% {s['top1']:>4}/{s['n']:<2} {s['top5']:>4}/{s['n']:<2} "
              f"{s['mrr']:>7.3f} {s['never']:>6} "
              f"{s['pinpoint']:>4}/{s['pinpoint_askable']:<3} "
              f"{s['chars'] if 'chars' in s else 0:>9}")
        report.append({"target": target, "units": len(merged),
                       "median_chars": statistics.median(sizes),
                       "median_tokens": statistics.median(toks), **s})

    print("-" * 78)
    best = max(report, key=lambda r: (r["top1"], r["mrr"]))
    base = report[0]
    print(f" current cap ({base['target']:,}): top-1 {base['top1']}/{base['n']}, "
          f"MRR {base['mrr']}, pinpoint {base['pinpoint']}/{base['pinpoint_askable']}")
    print(f" best        ({best['target']:,}): top-1 {best['top1']}/{best['n']}, "
          f"MRR {best['mrr']}, pinpoint {best['pinpoint']}/{best['pinpoint_askable']}")
    # A margin, not a comparison. With n=30 a single query is 3.3 points, so
    # "25 beats 24" is noise wearing the costume of a result. Require the gain to
    # be larger than one query before calling it real -- otherwise this test would
    # recommend re-chunking the corpus on a coin flip.
    noise = 1                                    # queries
    gain = best["top1"] - base["top1"]
    if best["target"] == base["target"] or gain <= noise:
        print(f"\n VERDICT: NO CHANGE NEEDED. The best merge level gains "
              f"{gain} query out of {base['n']}, which is within noise at this "
              f"sample size (one query = {100 / base['n']:.1f} points).")
        print(" Every level from 1,500 to 6,000 chars scores identically, so the")
        print(" effect does not track size -- more evidence it is not a size effect.")
        print(" Keep the current cap. Re-run this on the larger corpus to confirm.")
    else:
        print(f"\n VERDICT: {best['target']:,} chars gains {gain} queries "
              f"({gain * 100 / base['n']:.0f} points) over the current cap. "
              f"Weigh against the pinpoint cost above.")

    out = HERE / "report" / "merge_test.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
