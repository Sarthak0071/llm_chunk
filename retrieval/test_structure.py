"""Does OUR chunk structure earn its keep?

A fixed-size splitter would score the same as us on "chunks beat documents". That
comparison measures chunking, not our chunking. These four tests measure the parts
we actually built: roles, capsules, legislation citations, and summaries.

  1. LEGISLATION SEARCH  "find decisions applying HMK 353"
     Production cannot do this at all -- its payload carries no legislation -- so
     this is a capability, not an improvement. Scored as coverage and precision.

  2. ROLE FILTER  search only the reasoning parts
     Headers, case numbers and file-routing directives are noise. If tagging roles
     is worth its cost, filtering to reasoning should not lose targets and should
     cut the text we send.

  3. CAPSULE SEARCH  find the case by its summary, then the passage inside it
     Our two-level design, never tested. Stage 1 over 14 capsules, then drill into
     that decision's chunks.

  4. SUMMARY vs RAW TEXT  which is the better thing to search against?
     Production embeds decision prose; a summary is a different register. Decides
     what the chunker should optimise for.

    python test_structure.py
"""

import json
import sys
import uuid as _uuid
from collections import Counter
from pathlib import Path

import numpy as np
from qdrant_client import models

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "scripts" / "retrieval_test"))
sys.path.insert(0, str(HERE))

import contract                      # noqa: E402
import corpus as cor                 # noqa: E402
import qdrant_index as qi            # noqa: E402
from compare import load_queries     # noqa: E402
from embed import Embedder           # noqa: E402

CAPSULES = "capsules_ours"
REASONING_ROLES = {"rule", "application", "rule_application", "analysis", "issue"}


def hr(title):
    print("\n" + "=" * 74)
    print(f" {title}")
    print("=" * 74)


# ---------------------------------------------------------------- test 1
def test_legislation(qc):
    hr("1. LEGISLATION SEARCH — a capability production does not have")

    refs = Counter()
    n_with = 0
    pts, off = [], None
    while True:
        batch, off = qc.scroll(qi.CHUNKS, limit=512, offset=off,
                               with_payload=["law_refs", "unit", "parent_uuid"],
                               with_vectors=False)
        pts.extend(batch)
        if off is None:
            break
    for p in pts:
        lr = (p.payload or {}).get("law_refs") or []
        if lr:
            n_with += 1
        refs.update(lr)

    chunks = [p for p in pts if (p.payload or {}).get("unit") == "chunk"]
    print(f" chunks carrying legislation : {n_with}/{len(chunks)}")
    print(f" distinct references          : {len(refs)}")
    print(f"\n {'reference':16} {'chunks':>7}  {'decisions':>10}")
    results = []
    for ref, _ in refs.most_common(10):
        r = qc.query_points(
            collection_name=qi.CHUNKS,
            query=np.zeros(qi.DIM).tolist(),
            query_filter=models.Filter(must=[models.FieldCondition(
                key="law_refs", match=models.MatchAny(any=[ref]))]),
            limit=200, with_payload=["parent_uuid", "law_refs"])
        docs = {p.payload.get("parent_uuid") for p in r.points}
        # precision: every returned chunk must really carry the reference
        bad = [p for p in r.points if ref not in (p.payload.get("law_refs") or [])]
        print(f" {ref:16} {len(r.points):>7}  {len(docs):>10}"
              + ("   PRECISION BUG" if bad else ""))
        results.append({"ref": ref, "chunks": len(r.points), "docs": len(docs),
                        "false_positives": len(bad)})

    print(f"\n Arm A (whole documents) can answer NONE of these: production's payload")
    print(f" has no legislation field, so the query cannot be expressed at all.")
    return {"chunks_with_law": n_with, "chunk_total": len(chunks),
            "distinct_refs": len(refs), "probes": results}


# ---------------------------------------------------------------- test 2
def test_role_filter(qc, queries, vectors, parents):
    hr("2. ROLE FILTER — is tagging roles worth its cost?")

    role_f = models.Filter(should=[
        models.FieldCondition(key="role", match=models.MatchValue(value=r))
        for r in sorted(REASONING_ROLES)])

    rows = []
    for q, v in zip(queries, vectors):
        want = parents.get(q["id"])
        if not want:
            continue
        out = {}
        for label, flt in (("all", None), ("reasoning_only", role_f)):
            r = qc.query_points(collection_name=qi.CHUNKS, query=v.tolist(),
                                query_filter=flt, limit=15,
                                with_payload=["parent_uuid", "role", "unit"])
            rank = next((i + 1 for i, p in enumerate(r.points)
                         if p.payload.get("parent_uuid") == want), None)
            out[label] = rank
        rows.append(out)

    for label in ("all", "reasoning_only"):
        top1 = sum(1 for r in rows if r[label] == 1)
        top5 = sum(1 for r in rows if r[label] and r[label] <= 5)
        found = sum(1 for r in rows if r[label])
        print(f" {label:16} top-1 {top1:>2}/{len(rows)}   top-5 {top5:>2}/{len(rows)}"
              f"   found {found:>2}/{len(rows)}")

    lost = [r for r in rows if r["all"] and not r["reasoning_only"]]
    gained = [r for r in rows if r["reasoning_only"] and not r["all"]]
    print(f"\n targets lost by filtering   : {len(lost)}")
    print(f" targets gained by filtering : {len(gained)}")
    return {"rows": rows, "lost": len(lost), "gained": len(gained)}


# ---------------------------------------------------------------- test 3 & 4
def build_capsule_collection(qc, emb):
    """Index each capsule's summary text as its own point."""
    c = cor.Corpus()
    recs = []
    for cap in c.capsules:
        docs = c.capsule_doc_ids(cap)
        if len(docs) != 1:
            continue
        doc_id = next(iter(docs))
        src = cap.get("source_type")
        # Qdrant point ids must be a real UUID or an integer, so the capsule id is
        # derived in the same namespace rather than prefixed onto the parent's.
        parent = contract.parent_uuid_for(src, str(doc_id))
        recs.append({
            "uuid": str(_uuid.uuid5(contract.PARENT_NAMESPACE, f"capsule:{parent}")),
            "parent_uuid": parent,
            "text": c.capsule_text(cap),
            "source_type": src, "unit": "capsule",
            "subject_id": cap.get("subject_id"),
            "esas_year": 2024, "karar_year": 2024,
            "chamber": None, "court": None, "high_court": src,
        })
    vecs = emb.embed([r["text"] for r in recs])
    qi.ensure_collection(qc, CAPSULES, with_chunk_indexes=True, recreate=True)
    qi.upsert(qc, CAPSULES, recs, vecs)
    return recs


def test_capsule(qc, emb, queries, vectors, parents):
    hr("3. CAPSULE SEARCH — the two-level design, tested for the first time")
    recs = build_capsule_collection(qc, emb)
    print(f" capsules indexed : {len(recs)} "
          f"(covering {len({r['parent_uuid'] for r in recs})} decisions)")

    hits = miss = 0
    for q, v in zip(queries, vectors):
        want = parents.get(q["id"])
        if not want:
            continue
        r = qc.query_points(collection_name=CAPSULES, query=v.tolist(), limit=5,
                            with_payload=["parent_uuid", "subject_id"])
        rank = next((i + 1 for i, p in enumerate(r.points)
                     if p.payload.get("parent_uuid") == want), None)
        if rank == 1:
            hits += 1
        elif rank is None:
            miss += 1
    n = sum(1 for q in queries if parents.get(q["id"]))
    print(f" capsule top-1 : {hits}/{n}   never in top-5 : {miss}/{n}")
    print(f" NOTE: only {len(recs)} capsules exist, so the pool is tiny — chance"
          f" alone is 1/{len(recs)}. Directional only.")
    return {"capsules": len(recs), "top1": hits, "never": miss, "n": n}


def test_summary_vs_text(qc, queries, vectors, parents):
    hr("4. SUMMARY vs RAW TEXT — which is the better search target?")
    print(f" {'target':18} {'top-1':>8} {'top-5':>8}")
    out = {}
    for label, coll in (("capsule summary", CAPSULES), ("chunk text", qi.CHUNKS)):
        t1 = t5 = n = 0
        for q, v in zip(queries, vectors):
            want = parents.get(q["id"])
            if not want:
                continue
            n += 1
            r = qc.query_points(collection_name=coll, query=v.tolist(), limit=5,
                                with_payload=["parent_uuid"])
            rank = next((i + 1 for i, p in enumerate(r.points)
                         if p.payload.get("parent_uuid") == want), None)
            t1 += rank == 1
            t5 += bool(rank)
        print(f" {label:18} {t1:>4}/{n:<3} {t5:>4}/{n:<3}")
        out[label] = {"top1": t1, "top5": t5, "n": n}
    print("\n Not a fair fight: the capsule pool is 14 points, the chunk pool 1,167.")
    print(" Read it as 'can a summary find its own case', not as a head-to-head.")
    return out


def main():
    qc = qi.client()
    emb = Embedder()
    queries = [q for q in load_queries() if q.get("doc_id")]
    vectors = emb.embed([q["text"] for q in queries])
    parents = {q["id"]: contract.parent_uuid_for(q["source"], str(q["doc_id"]))
               for q in queries}
    print(f"queries with a document answer key: {len(queries)}")

    report = {
        "legislation": test_legislation(qc),
        "role_filter": test_role_filter(qc, queries, vectors, parents),
        "capsule": test_capsule(qc, emb, queries, vectors, parents),
        "summary_vs_text": test_summary_vs_text(qc, queries, vectors, parents),
    }
    out = HERE / "report" / "structure.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
