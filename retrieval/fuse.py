"""Reciprocal Rank Fusion, matching production exactly — including its tie-break.

Used by the hybrid path (the website's search screen), not by the assistant. The
assistant ranks with Qdrant alone.

    RRF(d) = sum over systems of  weight / (k + rank + 1)

with k = 60, ranks 0-based, and weights 1.0 / 1.0. The weights are parameters in
production but nothing ever passes anything else — not the FastAPI request model,
not the Django view — so 1:1 is the real behaviour, not a default nobody reached.

TWO DETAILS THAT LOOK LIKE TRIVIA AND ARE NOT

1. THE LEXICAL SIDE WINS TIES. Production builds its score dict by iterating the
   lexical results first, then the vector results, and finally sorts with Python's
   `sorted`, which is stable. So when two documents have identical RRF scores the
   one the lexical engine saw first stays ahead. With 1:1 weights ties are common —
   any document found only by lexical at rank r ties with any found only by vector
   at rank r — so this is systematic, not an edge case.

2. THE JOIN IS BY ID, AND THE TWO SIDES MUST AGREE. The lexical side keys on the
   stored `uuid` field; the vector side keys on the point id. A record whose id
   differs between the two stores can never fuse: it appears twice, once from each
   system, each with HALF the score a document found by both would get. That is
   exactly what would happen to chunks if their ids were minted independently, and
   it is why the contract insists on one id in both stores.
"""

import config as cfg


def reciprocal_rank_fusion(lexical, vector, k=cfg.RRF_K,
                           w_lexical=cfg.RRF_WEIGHT_LEXICAL,
                           w_vector=cfg.RRF_WEIGHT_VECTOR):
    """Fuse two ranked lists of dicts carrying an "id".

    Returns them merged and sorted, each with `rrf_score` and a `source` of
    "lexical", "vector" or "both".
    """
    scores, meta, order = {}, {}, []

    # Lexical FIRST — insertion order is what breaks ties, see the docstring.
    for rank, hit in enumerate(lexical):
        did = str(hit.get("id", ""))
        if not did:
            continue
        if did not in scores:
            order.append(did)
            meta[did] = dict(hit)
            meta[did]["source"] = "lexical"
        scores[did] = scores.get(did, 0.0) + w_lexical / (k + rank + 1)

    for rank, hit in enumerate(vector):
        did = str(hit.get("id", ""))
        if not did:
            continue
        if did not in scores:
            order.append(did)
            meta[did] = dict(hit)
            meta[did]["source"] = "vector"
        else:
            # Found by both. Production keeps the metadata it already had (the
            # lexical copy) and only upgrades the label.
            meta[did]["source"] = "both"
        scores[did] = scores.get(did, 0.0) + w_vector / (k + rank + 1)

    fused = []
    for did in order:                      # insertion order, then a stable sort
        row = meta[did]
        row["id"] = did
        row["rrf_score"] = scores[did]
        fused.append(row)
    fused.sort(key=lambda r: r["rrf_score"], reverse=True)
    return fused


def _self_test():
    """Pin the behaviour that matters, so drift from production is caught."""
    ok = True

    # 1. the formula
    fused = reciprocal_rank_fusion([{"id": "a"}], [])
    expect = 1.0 / (cfg.RRF_K + 0 + 1)
    if abs(fused[0]["rrf_score"] - expect) > 1e-12:
        print(f"FAIL formula: {fused[0]['rrf_score']} != {expect}"); ok = False

    # 2. found by both scores double, and is labelled "both"
    fused = reciprocal_rank_fusion([{"id": "a"}], [{"id": "a"}])
    if abs(fused[0]["rrf_score"] - 2 * expect) > 1e-12 or fused[0]["source"] != "both":
        print(f"FAIL both-sides: {fused[0]}"); ok = False

    # 3. THE TIE-BREAK: same rank in each system -> lexical wins
    fused = reciprocal_rank_fusion([{"id": "lex"}], [{"id": "vec"}])
    if fused[0]["id"] != "lex":
        print(f"FAIL tie-break: expected lexical first, got {fused[0]['id']}"); ok = False

    # 4. ids that do not match never fuse -- each keeps HALF the combined score
    fused = reciprocal_rank_fusion([{"id": "doc-1"}], [{"id": "chunk-1"}])
    if len(fused) != 2 or any(f["source"] == "both" for f in fused):
        print(f"FAIL mismatched ids should not fuse: {fused}"); ok = False
    if abs(fused[0]["rrf_score"] - expect) > 1e-12:
        print("FAIL mismatched id should score half of a both-sides hit"); ok = False

    # 5. rank order is respected
    fused = reciprocal_rank_fusion([{"id": "x"}, {"id": "y"}], [])
    if [f["id"] for f in fused] != ["x", "y"]:
        print(f"FAIL ordering: {[f['id'] for f in fused]}"); ok = False

    print(f"fuse self-test: {'PASS' if ok else 'FAIL'}  "
          f"(k={cfg.RRF_K}, weights {cfg.RRF_WEIGHT_LEXICAL}:{cfg.RRF_WEIGHT_VECTOR})")
    return 0 if ok else 1


if __name__ == "__main__":
    import sys
    sys.exit(_self_test())
