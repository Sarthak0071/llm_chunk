"""Hammurabi's agent search path, replicated stage for stage.

This is the path the assistant actually runs. It is Qdrant only -- Meilisearch does
no ranking here, it is a text lookup by id -- so it is also where the question
"are our chunks better?" gets answered.

THE STAGES, in production's order and with production's constants:

  1. several query formulations (the tool schema demands >= 3)
  2. embed them raw -- no prefix, no metadata
  3. build filters (court, chamber as int, year ranges)
  4. TWO Qdrant queries, not one: recent (esas_year >= 2023) at 10 per query, and
     old (< 2023) capped at max(5, limit // 10)
  5. rescale: boost recent, decay old
  6. dedup by filename, keep the best
  7. hydrate text by id -- and DROP anything the text store does not know
  8. filter: min score 0.10, dedup again, cap at 15 documents

Stage 7 is the one that silently discards chunks, so it is reproduced faithfully
and can then be switched to a chunk-aware form -- that switch is the Hammurabi
change H3, and the flag here is what measures whether it is needed.

Every upgrade is a flag, default off, so the same code produces production's
behaviour and ours, and the difference between them is attributable.
"""

from dataclasses import dataclass, field

import numpy as np
from qdrant_client import models

import config as cfg


@dataclass
class SearchOptions:
    """Defaults reproduce production exactly. Each flag is one Hammurabi change."""
    # --- faithful knobs ---
    limit_per_query: int = cfg.LIMIT_PER_QUERY
    min_score: float = cfg.MIN_SCORE
    max_docs: int = cfg.MAX_DOCS
    freshness: bool = True

    # --- upgrades, default off ---
    group_by_parent: bool = False      # replaces dedup-by-filename (H2)
    chunk_aware_hydration: bool = False  # resolve via parent_uuid (H3) -- the blocker
    per_parent: int = 1                # passages kept per decision when grouping
    emit_paragraph_ids: bool = False   # carry paragraph_ids to the caller (H4)

    filters: dict = field(default_factory=dict)


def build_filter(filters, extra=None):
    """Production's filter builder: all conditions ANDed, years as strict ranges,
    chamber matched as an INTEGER because that is how it is indexed."""
    conds = list(extra or [])
    for key, fld, op in (("esas_year_gt", "esas_year", "gt"),
                         ("esas_year_lt", "esas_year", "lt"),
                         ("karar_year_gt", "karar_year", "gt"),
                         ("karar_year_lt", "karar_year", "lt")):
        if filters.get(key):
            conds.append(models.FieldCondition(
                key=fld, range=models.Range(**{op: filters[key]})))

    for fld in ("chamber", "court", "high_court"):
        val = filters.get(fld)
        if val is None:
            continue
        if fld == "chamber":
            try:
                val = int(val) if not isinstance(val, list) else [int(x) for x in val]
            except (TypeError, ValueError):
                continue          # production skips an unparseable chamber, not 400s
        if isinstance(val, list):
            conds.append(models.FieldCondition(key=fld, match=models.MatchAny(any=val)))
        else:
            conds.append(models.FieldCondition(key=fld,
                                               match=models.MatchValue(value=val)))
    return models.Filter(must=conds) if conds else None


def _rescale(hits, opts):
    """Production's freshness rescale. The returned score is NOT cosine similarity
    any more -- it is similarity times a year multiplier, which is why old law
    struggles to clear the min-score floor."""
    if not opts.freshness:
        return hits
    out = []
    for h in hits:
        year = h["payload"].get("esas_year")
        year = year if isinstance(year, int) else cfg.FRESHNESS_PIVOT_YEAR
        if year >= cfg.FRESHNESS_PIVOT_YEAR:
            mult = 1.0 + cfg.FRESHNESS_BOOST_PER_YEAR * (year - cfg.FRESHNESS_PIVOT_YEAR)
        else:
            mult = cfg.FRESHNESS_DECAY_BASE ** (
                (cfg.FRESHNESS_PIVOT_YEAR - year) / cfg.FRESHNESS_DECAY_YEARS)
        h = dict(h)
        h["raw_score"] = h["score"]
        h["score"] = h["score"] * mult
        h["freshness_multiplier"] = mult
        out.append(h)
    return out


def _collapse(hits, opts):
    """Production keeps ONE hit per filename, highest score.

    For chunks that is fatal: forty passages of one decision collapse to one, and
    the collapse happens before the result budget is spent, so asking for 100 can
    yield 8. `group_by_parent` keeps the best `per_parent` passages of each
    decision instead -- same intent, but aware that a decision now has parts.
    """
    key_of = (lambda h: h["payload"].get("parent_uuid") or h["payload"].get("filename")
              ) if opts.group_by_parent else (lambda h: h["payload"].get("filename", ""))

    groups = {}
    for h in sorted(hits, key=lambda x: -x["score"]):
        groups.setdefault(key_of(h), []).append(h)

    keep = opts.per_parent if opts.group_by_parent else 1
    out = [h for g in groups.values() for h in g[:keep]]
    return sorted(out, key=lambda x: -x["score"])


def _hydrate(hits, text_store, opts):
    """Stage 7 -- and the reason chunks vanish today.

    Production keeps a result ONLY if its point id is a key in the text store:

        if doc_uuid and retrieve_doc_content.get(doc_uuid): keep

    A chunk id is not a decision id, so every chunk result is dropped with no
    error. With `chunk_aware_hydration` the text is resolved through
    `parent_uuid` when the id itself is unknown, which is Hammurabi change H3.
    """
    out, dropped = [], 0
    for h in hits:
        uid = h["id"]
        text = text_store.get(uid)
        if text is None and opts.chunk_aware_hydration:
            text = text_store.get(h["payload"].get("parent_uuid"))
        if text is None:
            dropped += 1
            continue
        h = dict(h)
        h["text"] = text
        out.append(h)
    return out, dropped


def search(qc, collection, query_vectors, text_store, opts=None):
    """Run the agent path. `query_vectors` is (n_queries, dim).

    Returns (results, trace). The trace records what each stage did, because the
    interesting failures here are all silent ones -- things dropped, collapsed or
    decayed out of existence.
    """
    opts = opts or SearchOptions()
    n_q = len(query_vectors)
    limit = opts.limit_per_query * n_q
    old_limit = max(cfg.OLD_BUCKET_MIN, limit // cfg.OLD_BUCKET_DIVISOR)

    recent_f = build_filter(opts.filters, [models.FieldCondition(
        key="esas_year", range=models.Range(gte=cfg.FRESHNESS_PIVOT_YEAR))])
    old_f = build_filter(opts.filters, [models.FieldCondition(
        key="esas_year", range=models.Range(lt=cfg.FRESHNESS_PIVOT_YEAR))])

    payload_fields = list(cfg.PAYLOAD_FIELDS) + ["parent_uuid", "unit"]
    if opts.emit_paragraph_ids:
        payload_fields += ["paragraph_ids", "chunk_index", "role", "case_no"]

    # query_points, not the older search(): production's agent path builds
    # QueryRequest objects and calls query_batch_points, so this is the same API
    # family. Ranking is identical either way.
    raw = []
    for vec in query_vectors:
        v = vec.tolist() if isinstance(vec, np.ndarray) else list(vec)
        for flt, lim in ((recent_f, limit), (old_f, old_limit)):
            resp = qc.query_points(collection_name=collection, query=v,
                                   query_filter=flt, limit=lim,
                                   with_payload=payload_fields)
            for p in resp.points:
                raw.append({"id": str(p.id), "score": float(p.score),
                            "payload": dict(p.payload or {})})

    trace = {"queries": n_q, "recent_limit": limit, "old_limit": old_limit,
             "raw_hits": len(raw)}

    hits = _rescale(raw, opts)
    hits = _collapse(hits, opts)
    trace["after_collapse"] = len(hits)

    hits = hits[:limit]
    hits, dropped = _hydrate(hits, text_store, opts)
    trace["dropped_by_hydration"] = dropped
    trace["after_hydration"] = len(hits)

    kept = [h for h in hits if h["score"] >= opts.min_score]
    trace["dropped_by_min_score"] = len(hits) - len(kept)
    final = kept[:opts.max_docs]
    trace["returned"] = len(final)
    return final, trace
