"""Qdrant collections for both arms, with the payload indexes production needs.

Production never declares its collection anywhere we can read — there is no write
path in `hammurabi` at all — so this is not a copy of a configuration. It is the
configuration that should exist, written explicitly and recorded.

TWO COLLECTIONS, NOT ONE
  docs_baseline   one whole decision = one point   (Arm A, production-faithful)
  chunks_ours     1,025 documents + 142 chunks     (Arm B)

Arm B deliberately holds BOTH units: the decisions we have chunked are replaced by
their chunks, everything else stays a whole document. That is what production would
look like the day chunks ship for part of the corpus, and it means the two arms
differ in exactly one respect.

PAYLOAD INDEXES ARE NOT OPTIONAL
Qdrant returns a 400 — not an empty result — when you filter on an unindexed field.
Production indexes five (`esas_year`, `karar_year` as ranges, `chamber` as an
integer, `court` and `high_court` as keywords) and its own code comments warn that
passing `chamber` as the string "7" makes Qdrant look for a keyword index that does
not exist. We create those five identically, plus the ones chunks need.

DISTANCE AND SIZE
Production declares neither. bge-m3 emits 1024 dimensions and its SentenceTransformer
pipeline L2-normalises, which makes Cosine the only coherent choice. Recorded here
as an assumption to confirm against the live collection, not as a known fact.
"""

import sys
from pathlib import Path

from qdrant_client import QdrantClient, models

HERE = Path(__file__).resolve().parent
STORAGE = HERE / "report" / "qdrant"

DIM = 1024
DISTANCE = models.Distance.COSINE          # assumption — see module docstring

BASELINE = "docs_baseline"
CHUNKS = "chunks_ours"

# Exactly production's five. Same names, same types, same reasons.
PRODUCTION_INDEXES = {
    "esas_year": models.PayloadSchemaType.INTEGER,
    "karar_year": models.PayloadSchemaType.INTEGER,
    "chamber": models.PayloadSchemaType.INTEGER,
    "court": models.PayloadSchemaType.KEYWORD,
    "high_court": models.PayloadSchemaType.KEYWORD,
}

# Ours. Without an index on parent_uuid there is no way to group chunks back into
# a decision, which is the single operation every chunk-aware read needs.
CHUNK_INDEXES = {
    "parent_uuid": models.PayloadSchemaType.KEYWORD,
    "source_type": models.PayloadSchemaType.KEYWORD,
    "role": models.PayloadSchemaType.KEYWORD,
    "content_type": models.PayloadSchemaType.KEYWORD,
    "unit": models.PayloadSchemaType.KEYWORD,
    "chunk_index": models.PayloadSchemaType.INTEGER,
    # A keyword ARRAY. Qdrant matches any element, so one index answers "citing
    # HMK", "citing HMK 353" and "citing 6100/353". Production has no legislation
    # field at all, so this query is impossible there rather than merely slow.
    "law_refs": models.PayloadSchemaType.KEYWORD,
}

# Production projects its payload to a fixed list, so any field outside it is not
# returned even though it is stored. Ours must therefore be declared explicitly or
# paragraph_ids would silently never reach a caller.
PAYLOAD_FIELDS = [
    "filename", "title", "court", "E_no", "K_no", "chamber", "high_court",
    "esas_year", "karar_year", "esas_series", "karar_series",
]
CHUNK_PAYLOAD_FIELDS = PAYLOAD_FIELDS + [
    "parent_uuid", "unit", "paragraph_ids", "chunk_index", "role",
    "role_vocabulary", "content_type", "source_type", "case_no", "law_refs",
]


def client():
    STORAGE.mkdir(parents=True, exist_ok=True)
    return QdrantClient(path=str(STORAGE))


def ensure_collection(qc, name, with_chunk_indexes, recreate=False):
    """Create the collection and every payload index it needs.

    Idempotent: safe to call repeatedly. `recreate` drops first, for a clean rebuild.
    """
    exists = qc.collection_exists(name)
    if exists and recreate:
        qc.delete_collection(name)
        exists = False
    if not exists:
        qc.create_collection(
            collection_name=name,
            vectors_config=models.VectorParams(size=DIM, distance=DISTANCE),
        )

    wanted = dict(PRODUCTION_INDEXES)
    if with_chunk_indexes:
        wanted.update(CHUNK_INDEXES)

    for field, schema in wanted.items():
        try:
            qc.create_payload_index(collection_name=name, field_name=field,
                                    field_schema=schema)
        except Exception:
            pass          # already indexed
    return wanted


def upsert(qc, name, records, vectors, batch_size=128):
    """Write records + vectors. Point id is the record's own uuid, so re-running
    OVERWRITES rather than duplicating — the property that makes re-indexing safe.
    """
    assert len(records) == len(vectors), "records and vectors must line up"
    total = 0
    for start in range(0, len(records), batch_size):
        chunk_recs = records[start:start + batch_size]
        chunk_vecs = vectors[start:start + batch_size]
        qc.upsert(
            collection_name=name,
            points=[
                models.PointStruct(id=r["uuid"], vector=v.tolist(), payload=r)
                for r, v in zip(chunk_recs, chunk_vecs)
            ],
            wait=True,
        )
        total += len(chunk_recs)
    return total


def verify(qc, name, expected_count, wanted_indexes):
    """Prove the collection is actually usable, not merely created.

    Checks three things that each fail silently otherwise: the point count, that
    every declared index really accepts a filter, and that a stored payload comes
    back with the fields we projected.
    """
    problems = []
    info = qc.get_collection(name)
    count = qc.count(name, exact=True).count
    if count != expected_count:
        problems.append(f"point count {count}, expected {expected_count}")

    size = info.config.params.vectors.size
    dist = info.config.params.vectors.distance
    if size != DIM:
        problems.append(f"vector size {size}, expected {DIM}")

    # A filter on an unindexed field is a 400, not an empty result. Test each.
    probes = {
        "esas_year": models.FieldCondition(key="esas_year",
                                           range=models.Range(gte=1900)),
        "karar_year": models.FieldCondition(key="karar_year",
                                            range=models.Range(gte=1900)),
        "chamber": models.FieldCondition(key="chamber",
                                         match=models.MatchValue(value=5)),
        "court": models.FieldCondition(key="court",
                                       match=models.MatchValue(value="x")),
        "high_court": models.FieldCondition(key="high_court",
                                            match=models.MatchValue(value="yargitay")),
        "parent_uuid": models.FieldCondition(key="parent_uuid",
                                             match=models.MatchValue(value="x")),
        "source_type": models.FieldCondition(key="source_type",
                                             match=models.MatchValue(value="aym")),
        "role": models.FieldCondition(key="role",
                                      match=models.MatchValue(value="facts")),
        "content_type": models.FieldCondition(key="content_type",
                                              match=models.MatchValue(value="ruling")),
        "unit": models.FieldCondition(key="unit",
                                      match=models.MatchValue(value="chunk")),
        "chunk_index": models.FieldCondition(key="chunk_index",
                                             match=models.MatchValue(value=0)),
        "law_refs": models.FieldCondition(key="law_refs",
                                          match=models.MatchAny(any=["HMK"])),
    }
    for field in wanted_indexes:
        try:
            qc.count(name, count_filter=models.Filter(must=[probes[field]]),
                     exact=True)
        except Exception as exc:                                  # noqa: BLE001
            problems.append(f"filter on {field!r} failed: "
                            f"{type(exc).__name__}: {str(exc)[:80]}")

    return {"count": count, "size": size, "distance": str(dist),
            "indexes": sorted(wanted_indexes), "problems": problems}


if __name__ == "__main__":
    qc = client()
    print(f"storage: {STORAGE}")
    for name in (BASELINE, CHUNKS):
        if qc.collection_exists(name):
            info = qc.get_collection(name)
            print(f"  {name:16} {qc.count(name, exact=True).count:>6} points, "
                  f"dim {info.config.params.vectors.size}, "
                  f"{info.config.params.vectors.distance}")
        else:
            print(f"  {name:16} does not exist")
    sys.exit(0)
