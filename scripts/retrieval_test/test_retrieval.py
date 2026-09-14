"""
test_retrieval.py -- Stages 2-5: audit the query set, run retrieval, run
controls, print a report that states what it cannot prove.

WHAT THIS IS AND IS NOT
  With 10 documents this is a SMOKE TEST, not an accuracy measurement. Guessing
  at random gets Stage 1 top-1 right 1 time in 10. One query flipping moves a
  per-source score by 50 points. So no aggregate percentage is printed anywhere:
  every number appears as k/n with its own chance baseline beside it.

  What it CAN do: catch catastrophic failure (a question written from document X
  cannot find X among 10), and regress against itself over time. What it CANNOT
  do at this size: tell you the system's accuracy.

  BM25 also has no Turkish stemmer, so every number here is PESSIMISTIC relative
  to real Meilisearch. See bm25.py LIMITATIONS.

Usage:
    python test_retrieval.py --audit-only     # leakage audit, no scoring
    python test_retrieval.py                  # full report
    python test_retrieval.py --self-test      # prove the harness can fail
"""

import argparse
import hashlib
import json
import math
import random
import sys
from collections import Counter, defaultdict

import bm25
import corpus as C

QUERIES_PATH = C.__file__.replace("corpus.py", "queries.json")
SHUFFLE_N = 2000
SHUFFLE_SEED = 20260911
LEAK_NGRAM_FLAG = 5          # a shared 5-gram is a copied phrase, not a paraphrase


# --------------------------------------------------------------------------- #
# query set
# --------------------------------------------------------------------------- #

def load_queries(path=QUERIES_PATH):
    try:
        raw = open(path, "rb").read()
    except FileNotFoundError:
        return None, None
    sha = hashlib.sha256(raw).hexdigest()
    data = json.loads(raw.decode("utf-8"))
    queries = data.get("queries") if isinstance(data, dict) else data
    return queries or [], sha


def validate_queries(queries, c):
    """Reject a query set that cannot be scored, rather than silently skipping
    entries and reporting a flattering denominator."""
    errs = []
    docs = set(c.documents)
    for i, q in enumerate(queries):
        where = f"queries[{i}]"
        if not (q.get("query") or "").strip():
            errs.append(f"{where}: empty query")
        if q.get("stage") not in ("case", "chunk"):
            errs.append(f"{where}: stage must be 'case' or 'chunk', got {q.get('stage')!r}")
        key = (q.get("source"), q.get("expect_doc_id"))
        if key not in docs:
            errs.append(f"{where}: {key} is not a document in the generated output")
            continue
        if q["stage"] == "chunk":
            para = q.get("expect_paragraph")
            if not para:
                errs.append(f"{where}: stage 'chunk' needs expect_paragraph")
            elif not C.PARAGRAPH_RE.match(str(para)):
                errs.append(f"{where}: expect_paragraph must look like 'p18', got {para!r}")
            elif key + (para,) not in c.paragraph_index:
                errs.append(f"{where}: paragraph {para} is not referenced by any chunk of "
                            f"{key[1]} -- the model may not have covered it")
    return errs


# --------------------------------------------------------------------------- #
# leakage audit
# --------------------------------------------------------------------------- #

def ngrams(tokens, n):
    return {tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)} if len(tokens) >= n else set()


def max_shared_ngram(a_tokens, b_tokens, cap=12):
    best = 0
    for n in range(1, cap + 1):
        if ngrams(a_tokens, n) & ngrams(b_tokens, n):
            best = n
        else:
            break
    return best


def target_text(c, q):
    """The text a pass would be scored against -- what leakage would leak from."""
    key = (q["source"], q["expect_doc_id"])
    if q["stage"] == "case":
        return " ".join(c.capsule_text(cap) for cap in c.capsules
                        if q["expect_doc_id"] in c.capsule_doc_ids(cap))
    ids = c.paragraph_index.get(key + (q["expect_paragraph"],), [])
    return " ".join(c.chunk_by_id[i].get("text") or "" for i in ids)


def audit(c, queries):
    rows = []
    for q in queries:
        qt = bm25.query_terms(q["query"])
        tt = bm25.tokenize(target_text(c, q))
        shared = set(qt) & set(tt)
        rows.append({
            "query": q["query"][:58],
            "source": q["source"],
            "stage": q["stage"],
            "query_terms": len(qt),
            "unigram_overlap": round(len(shared) / len(qt), 2) if qt else 0.0,
            "max_shared_ngram": max_shared_ngram(qt, tt),
        })
    return rows


def print_audit(rows):
    print("\nQUERY SET AUDIT -- how much of each query is lifted from the target text")
    print("  A high max_shared_ngram means the query reuses a phrase verbatim, so a")
    print("  pass would be keyword overlap rather than retrieval. Not auto-failed;")
    print("  shown so it cannot hide.\n")
    print(f"  {'src':13} {'stage':6} {'terms':>5} {'uni':>5} {'ngram':>5}  query")
    for r in rows:
        flag = "  <-- copied phrase" if r["max_shared_ngram"] >= LEAK_NGRAM_FLAG else ""
        print(f"  {r['source']:13} {r['stage']:6} {r['query_terms']:5} "
              f"{r['unigram_overlap']:5} {r['max_shared_ngram']:5}  {r['query']}{flag}")
    flagged = sum(1 for r in rows if r["max_shared_ngram"] >= LEAK_NGRAM_FLAG)
    print(f"\n  {flagged} of {len(rows)} queries share a {LEAK_NGRAM_FLAG}-gram with their target.")


# --------------------------------------------------------------------------- #
# baselines
# --------------------------------------------------------------------------- #

def random_hit_prob(pool, accept, k):
    """P(at least one acceptable item in a random top-k) from a pool."""
    if pool <= 0 or accept <= 0:
        return 0.0
    k = min(k, pool)
    if pool - accept < k:
        return 1.0
    return 1.0 - math.comb(pool - accept, k) / math.comb(pool, k)


# --------------------------------------------------------------------------- #
# retrieval
# --------------------------------------------------------------------------- #

def rank_documents(c, query, capsule_index, capsule_docs):
    """Stage 1: rank capsules, collapse to documents keeping each document's best
    position. A Stage 1 query resolves to a DOCUMENT -- 12 capsules cover 10
    documents, and scoring capsules directly would double-count."""
    ranked = capsule_index.rank(bm25.query_terms(query))
    out = []
    for i, score in ranked:
        for doc in capsule_docs[i]:
            if doc not in out:
                out.append(doc)
    return out


def stage1(c, queries):
    caps = c.capsules
    index = bm25.BM25([bm25.tokenize(c.capsule_text(x)) for x in caps])
    capsule_docs = [sorted(c.capsule_doc_ids(x)) for x in caps]
    pool = len({d for ds in capsule_docs for d in ds})

    results, failures = [], []
    for q in [x for x in queries if x["stage"] == "case"]:
        docs = rank_documents(c, q["query"], index, capsule_docs)
        want = q["expect_doc_id"]
        rank = (docs.index(want) + 1) if want in docs else None
        results.append({"query": q["query"], "source": q["source"], "rank": rank,
                        "pool": pool, "top1": rank == 1, "top3": bool(rank and rank <= 3),
                        "baseline_top1": round(random_hit_prob(pool, 1, 1), 3),
                        "baseline_top3": round(random_hit_prob(pool, 1, 3), 3)})
        if not (rank and rank <= 3):
            failures.append({"stage": "stage1 case", "source": q["source"],
                             "query": q["query"], "expected": want,
                             "got": docs[:3]})
    return results, failures, pool


def stage2(c, queries, filtered):
    results, failures = [], []
    all_chunks = c.chunks
    global_index = bm25.BM25([bm25.tokenize(x.get("text") or "") for x in all_chunks])

    for q in [x for x in queries if x["stage"] == "chunk"]:
        key = (q["source"], q["expect_doc_id"])
        accept_ids = set(c.paragraph_index.get(key + (q["expect_paragraph"],), []))
        if not accept_ids:
            continue

        if filtered:
            pool_chunks = c.chunks_of_doc(*key)
            index = bm25.BM25([bm25.tokenize(x.get("text") or "") for x in pool_chunks])
        else:
            pool_chunks, index = all_chunks, global_index

        accept_idx = {i for i, ch in enumerate(pool_chunks) if ch["chunk_id"] in accept_ids}
        ranked = index.rank(bm25.query_terms(q["query"]))
        rank = bm25.rank_of(ranked, accept_idx)
        pool, acc = len(pool_chunks), len(accept_idx)

        results.append({
            "query": q["query"], "source": q["source"], "paragraph": q["expect_paragraph"],
            "rank": rank, "pool": pool, "acceptable": acc,
            "top1": rank == 1, "top3": bool(rank and rank <= 3),
            "baseline_top1": round(random_hit_prob(pool, acc, 1), 3),
            "baseline_top3": round(random_hit_prob(pool, acc, 3), 3),
            # With a pool this small, "in the top 3" is arithmetic, not retrieval.
            "top3_meaningless": pool <= 3 or random_hit_prob(pool, acc, 3) >= 0.9,
        })
        if not (rank and rank <= 3):
            got = [pool_chunks[i].get("chunk_label") for i, _ in ranked[:3]]
            failures.append({
                "stage": "stage2 " + ("filtered" if filtered else "unfiltered"),
                "source": q["source"], "query": q["query"],
                "expected": f"{q['expect_doc_id']} {q['expect_paragraph']}", "got": got})
    return results, failures


def mrr(results):
    if not results:
        return 0.0
    return round(sum((1.0 / r["rank"]) if r["rank"] else 0.0 for r in results) / len(results), 3)


# --------------------------------------------------------------------------- #
# shuffled-label control
# --------------------------------------------------------------------------- #

def shuffled_control_stage1(c, queries, n=SHUFFLE_N):
    """Re-score with query->document labels permuted. If the real score is not
    meaningfully above this, the real score is not evidence of anything."""
    qs = [x for x in queries if x["stage"] == "case"]
    if len(qs) < 2:
        return None
    caps = c.capsules
    index = bm25.BM25([bm25.tokenize(c.capsule_text(x)) for x in caps])
    capsule_docs = [sorted(c.capsule_doc_ids(x)) for x in caps]
    ranked_docs = [rank_documents(c, q["query"], index, capsule_docs) for q in qs]
    truth = [q["expect_doc_id"] for q in qs]

    rng = random.Random(SHUFFLE_SEED)
    t1 = t3 = 0
    for _ in range(n):
        perm = truth[:]
        rng.shuffle(perm)
        for docs, want in zip(ranked_docs, perm):
            r = (docs.index(want) + 1) if want in docs else None
            t1 += r == 1
            t3 += bool(r and r <= 3)
    total = n * len(qs)
    return {"permutations": n, "queries": len(qs),
            "mean_top1": round(t1 / total, 3), "mean_top3": round(t3 / total, 3)}


# --------------------------------------------------------------------------- #
# reporting
# --------------------------------------------------------------------------- #

def per_source(results):
    by = defaultdict(list)
    for r in results:
        by[r["source"]].append(r)
    return by


def print_stage(title, results, note=""):
    print(f"\n{title}")
    if not results:
        print("  (no queries)")
        return
    t1 = sum(r["top1"] for r in results)
    t3 = sum(r["top3"] for r in results)
    n = len(results)
    b1 = sum(r["baseline_top1"] for r in results) / n
    b3 = sum(r["baseline_top3"] for r in results) / n
    print(f"  overall: top-1 {t1}/{n} (chance {b1:.2f})   "
          f"top-3 {t3}/{n} (chance {b3:.2f})   MRR {mrr(results)}")
    if note:
        print(f"  {note}")
    for src, rows in sorted(per_source(results).items()):
        k1 = sum(r["top1"] for r in rows)
        k3 = sum(r["top3"] for r in rows)
        print(f"    {src:13} top-1 {k1}/{len(rows)}   top-3 {k3}/{len(rows)}"
              f"   (n={len(rows)}: one flip moves this by {100/len(rows):.0f} points)")
    warn = [r for r in results if r.get("top3_meaningless")]
    if warn:
        print(f"  NOTE: {len(warn)} query(ies) have a pool small enough that top-3 is")
        print("        arithmetically near-certain; read top-1 only for those:")
        for r in warn:
            print(f"          {r['source']}/{r['paragraph']} pool={r['pool']} "
                  f"acceptable={r['acceptable']} chance_top3={r['baseline_top3']}")


def print_failures(failures):
    print("\nNAMED FAILURES (never hidden in an aggregate)")
    if not failures:
        print("  None.")
        return
    for f in failures:
        print(f"  [{f['source']} / {f['stage']}]")
        print(f"    query:    {f['query'][:100]}")
        print(f"    expected: {f['expected']}")
        print(f"    got:      {f['got']}")


def main():
    ap = argparse.ArgumentParser(description="Retrieval smoke test")
    ap.add_argument("--audit-only", action="store_true", help="leakage audit, no scoring")
    ap.add_argument("--self-test", action="store_true",
                    help="inject a deliberately wrong expectation and prove it is reported")
    ap.add_argument("--queries", default=QUERIES_PATH,
                    help="path to a query file (default queries.json)")
    args = ap.parse_args()

    c = C.Corpus()
    if not c.chunks:
        raise SystemExit("No generated output in output/chunk/. Run chunk_generate.py first.")

    queries, sha = load_queries(args.queries)
    if queries is None:
        print("No queries.json found.\n")
        print("  1. python make_worksheet.py      -> worksheet/ files with raw paragraphs")
        print("  2. write queries.json            -> see queries.example.json")
        print("  3. python test_retrieval.py --audit-only")
        print("\nQueries must be written from the worksheets (raw court text) only.")
        return 2
    if not queries:
        raise SystemExit("queries.json contains no queries.")

    errs = validate_queries(queries, c)
    if errs:
        print("queries.json has problems -- refusing to report a score on a broken set:")
        for e in errs:
            print("  " + e)
        return 2

    if args.self_test:
        victim = next((dict(q) for q in queries if q["stage"] == "case"), None)
        if not victim:
            raise SystemExit("--self-test needs at least one stage 'case' query.")
        # Point it at the WORST-ranking document for this query, not merely a
        # different one: an arbitrary other document can legitimately sit in the
        # top 3, which would make the self-test pass or fail by luck.
        caps = c.capsules
        idx = bm25.BM25([bm25.tokenize(c.capsule_text(x)) for x in caps])
        cdocs = [sorted(c.capsule_doc_ids(x)) for x in caps]
        order = rank_documents(c, victim["query"], idx, cdocs)
        unranked = [d for _s, d in c.documents if d not in order]
        wrong = unranked[0] if unranked else order[-1]
        victim["source"] = next(s for s, d in c.documents if d == wrong)
        victim["expect_doc_id"] = wrong
        victim["expect_case_no"] = c.case_no_of_doc[(victim["source"], wrong)]
        victim["query"] = "[SELF-TEST] " + victim["query"]
        queries = queries + [victim]
        print("SELF-TEST: one query has been pointed at the worst-ranking document on")
        print("           purpose. It MUST appear below as a named failure.\n")

    s = c.summary()
    print("=" * 78)
    print(" RETRIEVAL SMOKE TEST -- not an accuracy measurement")
    print("=" * 78)
    print(f" corpus     : {s['documents']} documents, {s['chunks']} chunks, {s['capsules']} capsules")
    print(f" queries    : {len(queries)}  (sha256 {sha[:16]})")
    print(f" scorer     : BM25 k1={bm25.K1} b={bm25.B}, no Turkish stemmer")
    print(f" reminder   : a 10-document pool means chance alone scores 1/10 at top-1.")

    rows = audit(c, queries)
    print_audit(rows)
    if args.audit_only:
        return 0

    s1, f1, pool1 = stage1(c, queries)
    s2f, f2f = stage2(c, queries, filtered=True)
    s2u, f2u = stage2(c, queries, filtered=False)
    ctrl = shuffled_control_stage1(c, queries)

    print_stage(f"STAGE 1 -- find the right DOCUMENT (pool = {pool1} documents)", s1)
    print_stage("STAGE 2 -- find the right paragraph, FILTERED to the correct case", s2f,
                note="this is the documented architecture: case_no is a hard filter")
    print_stage(f"STAGE 2 -- same queries, NO filter (pool = {len(c.chunks)} chunks)", s2u,
                note="measures whether the hard filter is load-bearing, with real counters")

    print("\nSHUFFLED-LABEL CONTROL")
    if ctrl:
        real1 = sum(r["top1"] for r in s1)
        print(f"  real     top-1 {real1}/{len(s1)}")
        print(f"  shuffled top-1 {ctrl['mean_top1']:.3f} mean over {ctrl['permutations']} "
              f"permutations  (top-3 {ctrl['mean_top3']:.3f})")
        print("  If real is not clearly above shuffled, the real number is not evidence.")
    else:
        print("  Needs at least 2 stage 'case' queries to permute.")

    print_failures(f1 + f2f + f2u)

    report = {
        "corpus": s, "queries_sha256": sha, "query_count": len(queries),
        "scorer": {"bm25_k1": bm25.K1, "bm25_b": bm25.B, "stemming": False},
        "audit": rows,
        "stage1": s1, "stage2_filtered": s2f, "stage2_unfiltered": s2u,
        "shuffled_control_stage1": ctrl,
        "failures": f1 + f2f + f2u,
        "caveat": ("Smoke test on a 10-document pool. No aggregate accuracy is "
                   "reported because none is supportable at this size. BM25 has no "
                   "Turkish stemmer, so results are pessimistic vs Meilisearch."),
    }
    C.REPORT_DIR.mkdir(parents=True, exist_ok=True)
    out = C.REPORT_DIR / "report.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")

    if args.self_test:
        ok = any("[SELF-TEST]" in f["query"] for f in f1)
        print(f"\nSELF-TEST: injected failure was reported -- {'PASS' if ok else 'FAIL'}")
        return 0 if ok else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
