"""Every constant production actually uses, with the file it came from.

Kept in one place so a drift from production is a one-line diff, and so the
fidelity tests have something to pin. Nothing here is tuned: if a number is wrong
it is wrong because production changed, not because we adjusted it to improve a
score.

Source: hammurabi @ service/vector_db/ and app_channels/ , read-only review.
"""

# --- the agent path -------------------------------------------------------

# service/vector_db/rag.py -- the year that splits "recent" from "old"
FRESHNESS_PIVOT_YEAR = 2023

# rag.py: boost = 1.0 + 0.25 * (year - PIVOT)   for year >= PIVOT
FRESHNESS_BOOST_PER_YEAR = 0.25

# rag.py: decay = 0.1 ** ((PIVOT - year) / 3)   for year < PIVOT
# A 2013 decision is multiplied by 0.1 ** 3.33 = 0.00046 and then culled by the
# min-score floor below. This is why the assistant rarely surfaces older law.
FRESHNESS_DECAY_BASE = 0.1
FRESHNESS_DECAY_YEARS = 3

# app_channels/utils/gemini_utils.py -- the agent asks for 10 results per query
LIMIT_PER_QUERY = 10

# rag.py -- the OLD bucket is capped independently, and hard: max(5, limit // 10).
# With three queries the recent bucket gets 30 and the old bucket gets 5, before
# the decay above is even applied. This matters more than the decay itself.
OLD_BUCKET_MIN = 5
OLD_BUCKET_DIVISOR = 10

# gemini_utils.py _lightweight_filter
MIN_SCORE = 0.10
MAX_DOCS = 15

# utils/tools/rag_search.py -- the tool schema requires at least three formulations
MIN_QUERY_FORMULATIONS = 3

# --- the hybrid path (website search screen) ------------------------------

# service/vector_db/main.py
TOP_K_RETRIEVAL = 1000
TOP_K_FUSION = 750
TOP_K_RERANK = 500
RERANK_DEFAULT = False              # main.py: rerank: bool = False

# utils/rrf_fusion.py -- weights are 1:1 and never overridden anywhere
RRF_K = 60
RRF_WEIGHT_LEXICAL = 1.0
RRF_WEIGHT_VECTOR = 1.0

# hybrid_search_engine.py _apply_procedural_penalty_inplace(penalty=0.4)
PROCEDURAL_PENALTY = 0.4

# utils/reranker.py -- the Turkish patterns that trigger that penalty
PROCEDURAL_PATTERNS = [
    r"dilekçenin reddine",
    r"duruşma ertelendi",
    r"süre verilmesine",
    r"dosyanın.*bekletilmesine",
    r"işlemden kaldırılmasına",
]

# --- the embedding contract ----------------------------------------------

# service/vector_db/models.py -- SentenceTransformer("BAAI/bge-m3")
EMBED_MODEL = "BAAI/bge-m3"
EMBED_DIM = 1024
EMBED_MAX_TOKENS = 8192

# rag.py embed_query passes the raw string. No prefix, no instruction template, no
# title or metadata prepended. Anything else moves our vectors relative to theirs.
EMBED_PREFIX = ""

# --- payload projection ---------------------------------------------------

# rag.py PAYLOAD_FIELDS. Qdrant honours this as a projection, so a field outside
# the list is stored but never returned.
PAYLOAD_FIELDS = [
    "filename", "title", "court", "E_no", "K_no", "chamber", "high_court",
    "esas_year", "karar_year", "esas_series", "karar_series",
]

# --- what production does NOT have ---------------------------------------

# Their search tool's enum. Half our corpus has nowhere to go: aym, kvkk, rekabet
# and uyusmazlik are absent, and the tool tells the model to default to yargitay.
PRODUCTION_HIGH_COURT_ENUM = ["bam", "yargitay", "danistay", "first_degree"]
