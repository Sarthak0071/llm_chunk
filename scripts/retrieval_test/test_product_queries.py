"""Topic retrieval, scored honestly -- including the questions we should refuse.

DIFFERENT QUESTION FROM test_retrieval.py. That harness asks "which chunk holds
paragraph 18 of this document?" and keys the answer to a paragraph number. This
one asks "find decisions about X", the way the product team actually asks, where
the honest answer is sometimes NOTHING.

Two kinds of query, scored by opposite rules:

  findable        the corpus holds a matching decision  -> pass by RETURNING it
  absent_control  the corpus holds nothing on the topic -> pass by RETURNING NOTHING

The second is not filler. In legal search a confident wrong answer is worse than
silence, because the user cannot tell "no such ruling exists" from "we did not
index it". A system that scores well on findable queries while inventing answers
for absent ones is not a good system.

WHAT KEEPS THIS HONEST
  1. Ground truth comes from RAW court text, never from generated chunks.
  2. Every query scored twice -- as written, and with retrieval boilerplate
     stripped. "kararlarini getir" is not a stopword and matches nearly every
     capsule; a pass that needs it is a pass bought with the word "karar".
  3. Chance baselines printed per query, computed from the actual pool size and
     the actual number of acceptable answers -- not assumed uniform.
  4. Shuffled-label control over 2000 permutations. If the real score is not
     clearly above the shuffled mean, the real score is not evidence.
  5. N-gram audit: the longest phrase each query shares with the capsule it is
     supposed to find. A shared 5-gram means the query and the summary use the
     same words, so a hit may be overlap rather than retrieval. Printed, never
     silently dropped.
  6. queries file SHA-256 in the report, so edits made after seeing results show.

    python test_product_queries.py
    python test_product_queries.py --audit-only
    python test_product_queries.py --verbose
"""

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "scripts" / "llm_chunk"))

import bm25                                  # noqa: E402
import corpus as cor                         # noqa: E402

QUERY_FILE = HERE / "product_queries.json"
DATA_DIR = ROOT / "data"
PERMUTATIONS = 2000
NGRAM_FLAG = 5

# Words that say "please search", not "search for this".
BOILERPLATE = {
    "karar", "kararı", "kararlar", "kararları", "kararlarını", "kararlarına",
    "kararların", "nelerdir", "getir", "bul", "bulun", "göster", "listele",
    "aym", "anayasa", "mahkemesi", "mahkeme", "yargıtay", "danıştay", "kvkk",
    "kurulu", "kurul", "kişisel", "verileri", "koruma", "ilişkin", "dair",
}


def load_raw_text():
    """{(source, doc_id): lowercased raw court text} -- what the answer key is
    built from. Never the generated output."""
    out = {}
    for path in sorted(DATA_DIR.glob("*.json")):
        if ".bak" in path.name or path.stem.startswith("documents_document"):
            continue
        for rec in json.loads(path.read_text(encoding="utf-8")):
            blob = " ".join([rec.get("content_text") or "",
                             rec.get("html_content") or "",
                             rec.get("title") or ""])
            out[(path.stem, str(rec.get("doc_id")))] = bm25.tr_lower(blob)
    return out


def distinctive(index, terms):
    """Query terms that are not merely common vocabulary.

    A term is distinctive when it appears in FEWER THAN HALF the documents --
    "more common than not is not distinctive". Stated as a rule rather than tuned
    against results, because tuning a threshold on 21 queries would convert this
    harness into a rubber stamp (bm25.py says the same about k1/b/stopwords).

    Motivation is measured: every absent-topic false positive so far matched only
    generic words while every distinctive word scored zero -- `amaciyla` ("for the
    purpose of") alone was enough to return an unrelated case.

    MEASURED RESULT AT 14 CAPSULES: THIS RULE CHANGES NOTHING. 12/14 findable and
    4/7 correct silence, both identical with and without it. The reason is the
    corpus size, not the idea: with N=14, `amaciyla` has df=2 and `yetki` df=1, so
    every word in the corpus is "rare" and no frequency threshold can separate
    generic vocabulary from distinctive vocabulary. The rule is therefore
    UNVALIDATED, not disproved -- at a realistic index size `amaciyla` would appear
    in most documents and `tenkis` in very few, which is exactly the separation it
    needs. Kept, reported, and explicitly NOT tuned: moving the threshold until the
    three failures disappear would be fitting a constant to three data points.
    Re-measure when the corpus is large enough for document frequency to mean
    something.
    """
    out = []
    for t in terms:
        df = sum(1 for tf in index.tf if t in tf)
        if df and df * 2 < index.N:
            out.append(t)
    return out


def longest_shared_ngram(a_tokens, b_tokens, cap=8):
    """Longest contiguous token run the query shares with the target text."""
    b = set()
    for n in range(1, cap + 1):
        b |= {tuple(b_tokens[i:i + n]) for i in range(len(b_tokens) - n + 1)}
    best = 0
    for n in range(1, cap + 1):
        for i in range(len(a_tokens) - n + 1):
            if tuple(a_tokens[i:i + n]) in b:
                best = max(best, n)
    return best


class Harness:
    def __init__(self):
        self.c = cor.Corpus()
        self.raw = load_raw_text()
        self.caps = list(self.c.capsules)
        self.cap_doc = [next(iter(self.c.capsule_doc_ids(x)), None) for x in self.caps]
        self.cap_src = [x.get("source_type") for x in self.caps]
        self.cap_tokens = [bm25.tokenize(self.c.capsule_text(x)) for x in self.caps]
        self.index = bm25.BM25(self.cap_tokens)

    def truth(self, q):
        """Documents that genuinely discuss the topic, split by whether we
        chunked them. The second number turns an absent result into an action."""
        terms = [bm25.tr_lower(t) for t in q["topic_terms"]]
        in_corpus = set(self.c.documents)
        hit = {k for k, txt in self.raw.items() if any(t in txt for t in terms)}
        return hit & in_corpus, len(hit - in_corpus)

    def score(self, q):
        t_in, t_out = self.truth(q)
        full = bm25.query_terms(q["query_tr"])
        topic = [t for t in full if t not in BOILERPLATE]
        want_docs = {d for _, d in t_in}

        r = {"id": q["id"], "kind": q["kind"], "expect_source": q["expect_source"],
             "query_tr": q["query_tr"], "truth_in_corpus": sorted(t_in),
             "truth_raw_only": t_out, "topic_terms_scored": topic}

        for label, terms in (("as_written", full), ("topic_only", topic)):
            ranked = self.index.rank(terms, top_k=len(self.caps))
            hits = [{"score": round(s, 3), "source": self.cap_src[i],
                     "doc_id": self.cap_doc[i], "subject": self.caps[i].get("subject_id")}
                    for i, s in ranked if s > 0]
            rank = next((n + 1 for n, h in enumerate(hits) if h["doc_id"] in want_docs), None)
            r[label] = {"top_score": round(ranked[0][1], 3) if ranked else 0.0,
                        "n_scored": len(hits), "rank_of_truth": rank,
                        "hits": hits[:3]}

        # GUARDED: the rule proposed for the search layer. A document qualifies
        # only if it matches at least one distinctive query term; matching purely
        # on common vocabulary is treated as no answer. Reported alongside the
        # unguarded score so its cost on findable queries is visible too, not just
        # its benefit on absent ones.
        dist = distinctive(self.index, topic)
        qualified = [(i, s) for i, s in self.index.rank(topic, top_k=len(self.caps))
                     if s > 0 and any(self.index.score([t], i) > 0 for t in dist)]
        g_hits = [{"score": round(s, 3), "source": self.cap_src[i],
                   "doc_id": self.cap_doc[i]} for i, s in qualified]
        r["guarded"] = {
            "distinctive_terms": dist,
            "top_score": g_hits[0]["score"] if g_hits else 0.0,
            "rank_of_truth": next((n + 1 for n, h in enumerate(g_hits)
                                   if h["doc_id"] in want_docs), None),
            "hits": g_hits[:3],
        }

        # why the top hit scored -- every false positive so far came from generic
        # vocabulary while every distinctive topic word scored zero
        if topic:
            best = self.index.rank(topic, top_k=1)[0][0]
            matched = [t for t in topic if self.index.score([t], best) > 0]
            r["term_breakdown"] = {"matched": matched,
                                   "zero": [t for t in topic if t not in matched]}

        # chance: with this pool and this many acceptable answers
        pool = len({d for d in self.cap_doc if d})
        r["chance_top1"] = round(len(want_docs) / pool, 3) if pool and want_docs else 0.0
        r["pool_documents"] = pool

        # n-gram leakage against the capsule we are supposed to find
        qt = bm25.tokenize(q["query_tr"])
        shared = 0
        for i, d in enumerate(self.cap_doc):
            if d in want_docs:
                shared = max(shared, longest_shared_ngram(qt, self.cap_tokens[i]))
        r["max_shared_ngram"] = shared
        return r

    def verdict(self, r):
        topic = r["topic_only"]
        if r["kind"] == "absent_control":
            if topic["top_score"] == 0:
                return "PASS", "nothing in the corpus, nothing returned -- correct silence"
            h = topic["hits"][0]
            extra = " (and from the wrong court)" if h["source"] != r["expect_source"] else ""
            return "FAIL", (f"nothing in the corpus, but returned {h['source']}/"
                            f"{h['doc_id']} at {topic['top_score']}{extra}")
        if not r["truth_in_corpus"]:
            return "SKIP", "declared findable but the corpus holds no match -- fix the key"
        rank = topic["rank_of_truth"]
        if rank == 1:
            return "PASS", "correct document at rank 1"
        if rank and rank <= 3:
            return "WARN", f"correct document at rank {rank}"
        if rank:
            return "FAIL", f"correct document only at rank {rank}"
        return "FAIL", "correct document never returned"


def shuffled_control(h, findables, rounds=PERMUTATIONS):
    """Re-score with the query->document mapping permuted. If the real top-1 is
    not clearly above this, the real number is not evidence."""
    rng = random.Random(42)
    truths = [ {d for _, d in h.truth(q)[0]} for q in findables ]
    ranked_cache = []
    for q in findables:
        topic = [t for t in bm25.query_terms(q["query_tr"]) if t not in BOILERPLATE]
        ranked = h.index.rank(topic, top_k=len(h.caps))
        ranked_cache.append([h.cap_doc[i] for i, s in ranked if s > 0])
    total = 0
    for _ in range(rounds):
        perm = truths[:]
        rng.shuffle(perm)
        total += sum(1 for hits, want in zip(ranked_cache, perm)
                     if hits and hits[0] in want)
    return total / rounds / len(findables) if findables else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audit-only", action="store_true",
                    help="n-gram leakage audit only, no scoring")
    ap.add_argument("--verbose", action="store_true", help="show top 3 hits")
    args = ap.parse_args()

    spec = json.loads(QUERY_FILE.read_text(encoding="utf-8"))
    qhash = hashlib.sha256(QUERY_FILE.read_bytes()).hexdigest()[:16]
    h = Harness()
    s = h.c.summary()

    print("=" * 79)
    print(" TOPIC RETRIEVAL -- real questions, including ones we should refuse")
    print("=" * 79)
    print(f" corpus  : {s['documents']} documents, {s['chunks']} chunks, "
          f"{s['capsules']} capsules")
    print(f" queries : {len(spec['queries'])}  (sha256 {qhash})")
    print(f" scorer  : BM25 k1=1.5 b=0.75, NO Turkish stemmer -- pessimistic vs "
          f"production\n")

    results = [h.score(q) for q in spec["queries"]]

    if args.audit_only:
        print(" N-GRAM AUDIT -- longest phrase each query shares with its target capsule")
        print(f" {NGRAM_FLAG}+ means the query may be matching a paraphrase of itself.\n")
        print(f"  {'query':30} {'ngram':>5}  {'terms':>5}")
        for r in results:
            if r["kind"] != "findable":
                continue
            flag = "  <-- CHECK" if r["max_shared_ngram"] >= NGRAM_FLAG else ""
            print(f"  {r['id']:30} {r['max_shared_ngram']:>5} "
                  f"{len(r['topic_terms_scored']):>5}{flag}")
        return 0

    tally = {"PASS": 0, "WARN": 0, "FAIL": 0, "SKIP": 0}
    for r in results:
        v, why = h.verdict(r)
        r["verdict"], r["reason"] = v, why
        tally[v] += 1

    for kind, title in (("findable", "FINDABLE -- the corpus holds a match; passing means returning it"),
                        ("absent_control", "ABSENT CONTROLS -- passing means returning NOTHING")):
        print("-" * 79)
        print(f" {title}")
        print("-" * 79)
        for r in [x for x in results if x["kind"] == kind]:
            print(f" [{r['verdict']:4}] {r['id']}")
            print(f"        {r['query_tr'][:74]}")
            if kind == "findable":
                print(f"        rank {str(r['topic_only']['rank_of_truth'] or '-'):>3}"
                      f"   chance top-1 {r['chance_top1']:.3f}"
                      f"   shared n-gram {r['max_shared_ngram']}"
                      f"   score {r['topic_only']['top_score']}")
            else:
                print(f"        in corpus 0 docs | elsewhere in data/ {r['truth_raw_only']}"
                      f" | score as written {r['as_written']['top_score']}"
                      f" | topic only {r['topic_only']['top_score']}")
                if r["verdict"] == "FAIL":
                    tb = r["term_breakdown"]
                    print(f"        matched {tb['matched']} | scored zero {tb['zero']}")
            print(f"        -> {r['reason']}")
            if args.verbose:
                for x in r["topic_only"]["hits"]:
                    print(f"           {x['score']:>7}  {x['source']}/{x['doc_id']}")
        print()

    findables = [q for q in spec["queries"] if q["kind"] == "findable"]
    fr = [r for r in results if r["kind"] == "findable"]
    ar = [r for r in results if r["kind"] == "absent_control"]
    top1 = sum(1 for r in fr if r["topic_only"]["rank_of_truth"] == 1)
    top3 = sum(1 for r in fr if (r["topic_only"]["rank_of_truth"] or 99) <= 3)
    mrr = sum(1 / r["topic_only"]["rank_of_truth"] for r in fr
              if r["topic_only"]["rank_of_truth"]) / len(fr) if fr else 0
    silent = sum(1 for r in ar if r["topic_only"]["top_score"] == 0)
    mean_chance = sum(r["chance_top1"] for r in fr) / len(fr) if fr else 0
    shuffled = shuffled_control(h, findables)

    print("=" * 79)
    print(" NUMBERS")
    print("=" * 79)
    print(f" findable queries        : {len(fr)}")
    print(f"   top-1                 : {top1}/{len(fr)}   "
          f"(chance {mean_chance:.3f}, shuffled control {shuffled:.3f})")
    print(f"   top-3                 : {top3}/{len(fr)}")
    print(f"   MRR                   : {mrr:.3f}")
    print(f" absent controls         : {len(ar)}")
    print(f"   correct silence       : {silent}/{len(ar)}")

    # What the proposed search-layer rule would do, measured on both sides.
    g_top1 = sum(1 for r in fr if r["guarded"]["rank_of_truth"] == 1)
    g_top3 = sum(1 for r in fr if (r["guarded"]["rank_of_truth"] or 99) <= 3)
    g_silent = sum(1 for r in ar if r["guarded"]["top_score"] == 0)
    print()
    print(" WITH the distinctive-term rule (a hit must match >=1 term appearing in")
    print(" fewer than half the documents -- proposed for the search layer, not the")
    print(" pipeline; the pipeline does no retrieval):")
    print(f"   findable top-1        : {g_top1}/{len(fr)}   (was {top1}/{len(fr)})")
    print(f"   findable top-3        : {g_top3}/{len(fr)}   (was {top3}/{len(fr)})")
    print(f"   correct silence       : {g_silent}/{len(ar)}   (was {silent}/{len(ar)})")
    if (g_top1, g_silent) == (top1, silent):
        print("   -> NO EFFECT at this corpus size, and that is a fact about the")
        print("      corpus rather than the rule: with N=%d every term is rare "
              "(`amaciyla` df=2)," % len(h.caps))
        print("      so no frequency threshold separates generic from distinctive")
        print("      vocabulary. UNVALIDATED, not disproved. Do not tune it here.")
    print(f" overall                 : {tally['PASS']} pass | {tally['WARN']} warn "
          f"| {tally['FAIL']} fail" + (f" | {tally['SKIP']} skip" if tally['SKIP'] else ""))
    print()
    print(f" Read with care: {len(fr)} findable queries over a {results[0]['pool_documents']}"
          f"-document pool. One flip moves top-1 by "
          f"{100/len(fr):.0f} points. This is a smoke test, not an accuracy measurement.")
    if top1 and shuffled and top1 / len(fr) < shuffled * 2:
        print(" WARNING: the real score is not clearly above the shuffled control.")

    out = ROOT / "output" / "retrieval" / "product_queries_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(
        {"corpus": s, "queries_sha256": qhash,
         "summary": {"findable": len(fr), "top1": top1, "top3": top3, "mrr": round(mrr, 3),
                     "chance_top1": round(mean_chance, 3), "shuffled_top1": round(shuffled, 3),
                     "absent_controls": len(ar), "correct_silence": silent},
         "results": results}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n wrote {out}")
    return 1 if tally["FAIL"] else 0


if __name__ == "__main__":
    sys.exit(main())
