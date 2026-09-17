"""The experiment: are our chunks better than whole documents?

Same questions, same embedding model, same search path, same corpus. The two arms
differ in exactly one respect -- whether the twelve decisions under test are stored
whole or as our chunks. Every other document is the same record with the same
vector in both, so a difference in score is attributable to the chunking.

  Arm A  docs_baseline   the twelve as whole documents      (Hammurabi today)
  Arm B  chunks_ours     the same twelve as our 142 chunks  (our pipeline)

WHAT IS SCORED
  E2  did the right DECISION come back, and at what rank
  E3  did it name the right PARAGRAPH -- Arm A structurally cannot, and that
      asymmetry is the point rather than a flaw in the test
  E4  how many characters would be sent to the LLM to answer

HONESTY
  * queries and ground truth come from files written against RAW court text; the
    generated output was never read while writing them
  * a shuffled-label control re-scores against permuted answers -- a score that is
    not clearly above it is reported as not-evidence
  * chance baselines are computed from the real pool, not assumed
  * both arms always print, including where whole documents win

    python compare.py                    # faithful defaults, both arms
    python compare.py --upgrades         # Arm B with the chunk-aware flags on
    python compare.py --stage chunk      # paragraph-level queries only
"""

import argparse
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "scripts" / "retrieval_test"))
sys.path.insert(0, str(HERE))

import config as cfg                 # noqa: E402
import qdrant_index as qi            # noqa: E402
import search_agent as sa            # noqa: E402
from embed import Embedder           # noqa: E402

PERMUTATIONS = 2000


def load_queries(stage=None):
    """Questions with ground truth, from the existing harness files.

    queries.json is keyed to a document + paragraph, which is what makes E3
    possible. product_queries.json is topic-level and only knows the document.
    """
    out = []
    p = ROOT / "scripts" / "retrieval_test" / "queries.json"
    if p.is_file():
        for q in json.loads(p.read_text(encoding="utf-8")).get("queries", []):
            if stage and q.get("stage") != stage:
                continue
            out.append({"id": q.get("id") or q["query"][:40],
                        "text": q["query"], "source": q.get("source"),
                        "doc_id": q.get("expect_doc_id") or q.get("doc_id"),
                        "paragraph": q.get("expect_paragraph"),
                        "origin": "queries.json"})
    # product_queries.json is deliberately NOT loaded here.
    #
    # Its ground truth is a set of topic terms matched against raw court text, not
    # a document id. Scored by this harness those queries can never register a hit,
    # so an earlier run counted all fourteen of them as failures in BOTH arms and
    # reported 14/44 where the honest figure was 14/30. They are measured properly
    # by scripts/retrieval_test/test_product_queries.py, which knows how to resolve
    # them; mixing the two answer-key styles in one denominator only hides both.
    return out


def text_store_for(qc, collection):
    """id -> text for everything in a collection, plus parent_uuid -> text.

    Stands in for Meilisearch, which in the agent path is exactly this: a lookup
    table. Parent ids are included so chunk-aware hydration has something to
    resolve against, mirroring what H3 would do in production.
    """
    store, offset = {}, None
    while True:
        points, offset = qc.scroll(collection, limit=512, offset=offset,
                                   with_payload=True, with_vectors=False)
        for p in points:
            pay = p.payload or {}
            store[str(p.id)] = pay.get("text") or ""
            if pay.get("unit") == "document":
                store.setdefault(pay.get("parent_uuid"), pay.get("text") or "")
        if offset is None:
            break
    return store


def ground_truth_parent(query, records_by_doc):
    """The parent_uuid a correct answer must belong to."""
    if query.get("doc_id"):
        return records_by_doc.get((query["source"], str(query["doc_id"])))
    return None


def run_arm(qc, collection, queries, vectors, store, opts, parents):
    rows = []
    for q, vec in zip(queries, vectors):
        hits, trace = sa.search(qc, collection, [vec], store, opts)
        want = parents.get(q["id"])
        rank = None
        hit_paras = []
        for i, h in enumerate(hits):
            pay = h["payload"]
            same = (pay.get("parent_uuid") == want) if want else False
            if same and rank is None:
                rank = i + 1
                hit_paras = pay.get("paragraph_ids") or []
        rows.append({"query": q["id"], "rank": rank, "returned": len(hits),
                     "trace": trace,
                     "paragraph_ids": hit_paras,
                     "expect_paragraph": q.get("paragraph"),
                     "chars": sum(len(h.get("text") or "") for h in hits),
                     "unit": hits[0]["payload"].get("unit") if hits else None,
                     # kept so the control can re-judge this query against a
                     # different query's answer
                     "want": want,
                     "returned_parents": [h["payload"].get("parent_uuid")
                                          for h in hits]})
    return rows


def shuffled_control(rows, rounds=PERMUTATIONS):
    """Score each query against ANOTHER query's expected answer.

    The point of a control is to ask "how often would this look like a hit by
    luck?". That means permuting the query -> answer mapping and re-scoring the
    SAME result lists: query i keeps its own returned documents, but is judged
    against query j's target.

    An earlier version shuffled the list of ranks and counted how many equalled 1.
    Shuffling a list does not change how many 1s it contains, so it returned the
    real top-1 rate every time and silently agreed with whatever it was meant to
    test. It is recorded here because a control that cannot disagree is worse than
    no control -- it manufactures confidence.
    """
    targets = [r.get("want") for r in rows]
    returned = [r.get("returned_parents") or [] for r in rows]
    if not any(targets):
        return 0.0
    idx = list(range(len(rows)))
    rng = random.Random(42)
    hits = 0
    for _ in range(rounds):
        rng.shuffle(idx)
        for i, j in enumerate(idx):
            if i == j:
                continue                      # not a permuted pairing
            want = targets[j]
            if want and returned[i] and returned[i][0] == want:
                hits += 1
    return hits / rounds / len(rows)


def summarise(name, rows):
    n = len(rows)
    top1 = sum(1 for r in rows if r["rank"] == 1)
    top5 = sum(1 for r in rows if r["rank"] and r["rank"] <= 5)
    found = [r for r in rows if r["rank"]]
    mrr = sum(1 / r["rank"] for r in found) / n if n else 0
    chars = sum(r["chars"] for r in rows) / n if n else 0
    pinpoint = sum(1 for r in rows
                   if r["expect_paragraph"] and r["paragraph_ids"]
                   and r["expect_paragraph"] in r["paragraph_ids"])
    askable = sum(1 for r in rows if r["expect_paragraph"])
    return {"arm": name, "queries": n, "top1": top1, "top5": top5,
            "mrr": round(mrr, 3), "avg_chars": int(chars),
            "pinpoint": pinpoint, "pinpoint_askable": askable,
            "never_found": n - len(found),
            "dropped_hydration": sum(r["trace"]["dropped_by_hydration"] for r in rows)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--upgrades", action="store_true",
                    help="run Arm B with the chunk-aware flags on")
    ap.add_argument("--stage", help="only queries with this stage (chunk / case)")
    ap.add_argument("--limit", type=int, help="first N queries only")
    ap.add_argument("--no-freshness", action="store_true",
                    help="disable production's year boost/decay in BOTH arms")
    ap.add_argument("--queries-per-question", type=int, default=1,
                    help="production's tool demands >= 3 formulations; the budget is "
                         "10 x this number, so it also widens the candidate pool")
    args = ap.parse_args()

    qc = qi.client()
    for name in (qi.BASELINE, qi.CHUNKS):
        if not qc.collection_exists(name):
            print(f"collection {name} does not exist — run index_corpus.py first")
            return 1

    queries = load_queries(args.stage)
    if args.limit:
        queries = queries[:args.limit]
    if not queries:
        print("no queries loaded")
        return 1

    # parent_uuid for each query's expected document, derived the same way the
    # records were built -- never read back from generated output.
    import contract
    parents = {}
    for q in queries:
        if q.get("doc_id") and q.get("source"):
            parents[q["id"]] = contract.parent_uuid_for(q["source"], str(q["doc_id"]))

    print("=" * 76)
    print(" CHUNKS vs WHOLE DOCUMENTS")
    print("=" * 76)
    print(f" Arm A : {qi.BASELINE:14} {qc.count(qi.BASELINE, exact=True).count:>6} points")
    print(f" Arm B : {qi.CHUNKS:14} {qc.count(qi.CHUNKS, exact=True).count:>6} points")
    print(f" queries: {len(queries)}  ({sum(1 for q in queries if q.get('paragraph'))}"
          f" with a paragraph-level answer key)")
    print(f" scoring: bge-m3, raw text, production constants "
          f"(min_score {cfg.MIN_SCORE}, max_docs {cfg.MAX_DOCS}, pivot "
          f"{cfg.FRESHNESS_PIVOT_YEAR})")

    emb = Embedder()
    vectors = emb.embed([q["text"] for q in queries])

    fresh = not args.no_freshness
    budget = cfg.LIMIT_PER_QUERY * args.queries_per_question
    base_opts = sa.SearchOptions(freshness=fresh, limit_per_query=budget)
    b_opts = sa.SearchOptions(
        freshness=fresh,
        limit_per_query=budget,
        chunk_aware_hydration=args.upgrades,
        group_by_parent=args.upgrades,
        emit_paragraph_ids=True,
        per_parent=3 if args.upgrades else 1,
    )
    print(f" freshness: {'ON (production)' if fresh else 'OFF'}   "
          f"candidate budget: {budget} recent + "
          f"{max(cfg.OLD_BUCKET_MIN, budget // cfg.OLD_BUCKET_DIVISOR)} old")

    results = {}
    for label, coll, opts in (("A (documents)", qi.BASELINE, base_opts),
                              ("B (our chunks)", qi.CHUNKS, b_opts)):
        store = text_store_for(qc, coll)
        rows = run_arm(qc, coll, queries, vectors, store, opts, parents)
        results[label] = {"rows": rows, "summary": summarise(label, rows),
                          "control": shuffled_control(rows)}

    print(f"\n{'':18} {'top-1':>7} {'top-5':>7} {'MRR':>7} {'never':>7} "
          f"{'¶ cite':>8} {'chars/q':>9} {'dropped':>8}")
    for label, r in results.items():
        s = r["summary"]
        pin = (f"{s['pinpoint']}/{s['pinpoint_askable']}"
               if s["pinpoint_askable"] else "n/a")
        print(f"{label:18} {s['top1']:>3}/{s['queries']:<3} {s['top5']:>3}/{s['queries']:<3}"
              f" {s['mrr']:>7.3f} {s['never_found']:>7} {pin:>8} "
              f"{s['avg_chars']:>9,} {s['dropped_hydration']:>8}")

    for label, r in results.items():
        print(f"\n {label}: shuffled control top-1 {r['control']:.3f}")

    a, b = results["A (documents)"]["summary"], results["B (our chunks)"]["summary"]
    print("\n" + "-" * 76)
    print(f" top-1     {a['top1']} -> {b['top1']}   ({b['top1'] - a['top1']:+d})")
    print(f" MRR       {a['mrr']:.3f} -> {b['mrr']:.3f}   ({b['mrr'] - a['mrr']:+.3f})")
    print(f" chars/q   {a['avg_chars']:,} -> {b['avg_chars']:,}   "
          f"({b['avg_chars'] - a['avg_chars']:+,})")
    print(f" pinpoint  {a['pinpoint']}/{a['pinpoint_askable']} -> "
          f"{b['pinpoint']}/{b['pinpoint_askable']}  "
          f"(Arm A cannot do this by construction)")
    if b["dropped_hydration"]:
        print(f"\n NOTE: {b['dropped_hydration']} chunk results were DISCARDED by "
              f"hydration.\n That is the production blocker (H3) reproducing "
              f"faithfully. Re-run with --upgrades.")

    out = HERE / "report" / "compare.json"
    out.write_text(json.dumps(
        {"upgrades": args.upgrades,
         "arms": {k: {"summary": v["summary"], "control": v["control"],
                      "rows": v["rows"]} for k, v in results.items()}},
        ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
