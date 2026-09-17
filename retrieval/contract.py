"""The index record contract — what a chunk must look like to survive in Hammurabi.

Our chunks and Hammurabi's index payload currently share ZERO field names. This
module is the bridge: it defines the record shape both sides agree on, derives it
from data we already hold, and refuses anything that would fail silently in
production.

WHY A CONTRACT RATHER THAN A MAPPING
Production discards, crashes on, or silently ignores records that are almost right:

  * a result whose id is unknown to Meilisearch is DROPPED with no error
  * `int(esas_year)` on a null raises and 500s the whole request
  * a record with no `esas_year` matches neither the recent (>= 2023) nor the old
    (< 2023) bucket, so it is invisible to the entire semantic path
  * `chamber` as the string "7" makes Qdrant look for a keyword index that does not
    exist -> 400
  * a field absent from the payload projection is simply never returned

Every rule below exists because one of those happens otherwise.

THREE THINGS FOUND IN THE REAL DATA, EACH LOAD-BEARING

1. `chamber_id` is NOT the chamber number, except for yargitay. bam stores 588,
   danistay 125, first_degree 25783 -- internal database ids -- while their titles
   say "11. Hukuk Dairesi", "13. Daire". Production wants an int in 1..50, so the
   chamber is parsed from the TITLE and chamber_id is deliberately ignored.

2. kvkk records have NO esas_year (esas_year/esas_no are both null). That is the
   exact field production splits its two queries on, so unfixed, every kvkk chunk
   would be invisible. We fall back to karar_year and record that we did.

3. aym records have NO karar_year. The mirror of (2); same treatment.

TWO VALUES ARE GUESSES UNTIL THE OTHER TEAM CONFIRMS
  * `filename`  -- production uses "__280908200.txt"; our doc_id is "1203538100",
                   so the pattern is plausible, not verified.
  * `parent_uuid` -- must equal production's uuid for the same decision. Their
                   generation scheme is unknown. We mint deterministically from
                   doc_id so re-running changes nothing, and keep it swappable.
Both are isolated in ONE function each, so confirming them is a one-line change.
"""

import re
import uuid

SCHEMA_VERSION = 1

# Generated once and fixed forever. Regenerating it would change every parent_uuid
# and turn re-indexing into duplication instead of overwrite.
PARENT_NAMESPACE = uuid.UUID("6f2b7c31-9d84-4a1e-bd0c-3e5f8a7d2c46")

# Exactly what production indexes on. A chunk missing any of these is not
# production-shaped, whatever else it carries.
PRODUCTION_FIELDS = [
    "uuid", "filename", "title", "court", "high_court", "chamber",
    "E_no", "K_no", "esas_year", "karar_year", "esas_series", "karar_series",
]

# What we add. These are the reason the exercise exists.
CHUNK_FIELDS = [
    "parent_uuid", "unit", "schema_version", "text", "paragraph_ids",
    "chunk_index", "role", "role_vocabulary", "content_type", "confidence",
    "case_no", "decision_date", "source_type", "cited_legislations",
    "law_refs", "derivation_notes",
]

INDEX_RECORD_FIELDS = PRODUCTION_FIELDS + CHUNK_FIELDS

# Production's court enum today. Half our corpus is not in it -- aym, kvkk,
# rekabet and uyusmazlik have nowhere to go, and the search tool even instructs the
# model to "default to 'yargitay'". Widening this enum is Hammurabi change H5, and
# it is backward compatible: no existing value changes meaning.
PRODUCTION_HIGH_COURTS = {"bam", "yargitay", "danistay", "first_degree"}
OUR_HIGH_COURTS = PRODUCTION_HIGH_COURTS | {"aym", "kvkk", "rekabet", "uyusmazlik"}

# Our role field is named differently per source (firac_role /
# court_reasoning_role / regulatory_role). A field whose NAME varies by row cannot
# be filtered as one thing, so the record carries a single `role` plus the
# vocabulary it came from. The original per-source field stays in our own output
# untouched -- nothing downstream breaks.
ROLE_FIELD_TO_VOCABULARY = {
    "firac_role": "firac",
    "court_reasoning_role": "court_reasoning",
    "regulatory_role": "regulatory",
}

# "11. Hukuk Dairesi", "13. Daire Başkanlığı", "7. Asliye Ticaret Mahkemesi".
# The number always precedes the court word, and there may be a city before it
# ("İzmir Bölge Adliye Mahkemesi 11. Hukuk Dairesi") -- so the LAST match wins,
# not the first: "İstanbul Anadolu 7. Asliye Ticaret" has no earlier number, but
# bam titles carry the regional court name ahead of the chamber.
RE_CHAMBER = re.compile(
    r"(\d{1,2})\s*\.\s*(?:[A-Za-zÇĞİÖŞÜçğıöşü]+\s+){0,3}?(?:Daire|Dairesi|Mahkemesi)")

# The trailing court phrase: everything from the chamber number to the court word.
RE_COURT = re.compile(
    r"(\d{1,2}\s*\.\s*(?:[A-Za-zÇĞİÖŞÜçğıöşü]+\s+){0,3}?(?:Daire(?:si)?|Mahkemesi)"
    r"(?:\s+Başkanlığı)?)")

# "... 2021/895 E. ,2022/1662 K." glued onto the end of a court name.
RE_TRAILING_CASE_NO = re.compile(
    r"\s*\d{4}/\d+\s*E\.?\s*,?\s*\d{4}/\d+\s*K\.?\s*$")


def parent_uuid_for(source, doc_id):
    """Deterministic parent id. GUESS until the other team confirms their scheme.

    Deterministic on purpose: re-running the chunker must produce the same id so
    re-indexing overwrites rather than duplicating. Isolated here so swapping in
    their real scheme is a one-line change.
    """
    return str(uuid.uuid5(PARENT_NAMESPACE, f"{source}:{doc_id}"))


def filename_for(doc_id):
    """GUESS. Production filenames look like "__280908200.txt" and our doc_id is
    "1203538100". Plausible, unverified -- isolated so confirming it is one line."""
    return f"__{doc_id}.txt"


def parse_chamber(title):
    """Chamber number from the title, or None.

    Deliberately NOT chamber_id: that column holds the real chamber only for
    yargitay (5), while bam stores 588, danistay 125 and first_degree 25783.
    Production constrains chamber to 1..50, so an out-of-range parse is discarded
    rather than passed through to a filter that would never match.
    """
    m = RE_CHAMBER.search(title or "")
    if not m:
        return None
    n = int(m.group(1))
    return n if 1 <= n <= 50 else None


# Institutions with no chamber structure. Their titles are the applicant's name
# (aym: "NACİYE ÇORUH Başvurusuna İlişkin Karar") or a summary of the matter (kvkk:
# "Plastik ev gereçleri ... veri sorumlusu ..."), so falling back to the title
# would put a person's name in a FILTERABLE court field -- useless to filter on and
# a privacy smell. The institution's own name is the right value.
INSTITUTION_COURT = {
    "aym": "Anayasa Mahkemesi",
    "kvkk": "Kişisel Verileri Koruma Kurulu",
    "rekabet": "Rekabet Kurulu",
    "uyusmazlik": "Uyuşmazlık Mahkemesi",
}


def parse_court(title, source=None):
    """The court phrase, e.g. "11. Hukuk Dairesi".

    Order matters: a chamber phrase in the title wins, because danistay's
    "Vergi Dava Daireleri Kurulu Başkanlığı" is a real court name and must survive.
    Only when no court phrase exists do we fall back to the institution name.
    """
    m = RE_COURT.search(title or "")
    if m:
        return m.group(1).strip()
    if source in INSTITUTION_COURT:
        return INSTITUTION_COURT[source]
    # Last resort: the title minus its trailing case numbers. danistay's
    # "Vergi Dava Daireleri Kurulu Başkanlığı 2021/895 E. ,2022/1662 K." is a real
    # court name with the case reference glued on; keeping the numbers would make
    # every such court a distinct filter value.
    return RE_TRAILING_CASE_NO.sub("", title or "").strip(" ,.") or None


def _int_or_none(v):
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def law_refs_for(citations):
    """Flatten cited_legislations into a keyword array Qdrant can index.

    `cited_legislations` is a list of objects, which no vector store filters on
    usefully. Flattening gives one searchable surface for the several ways a lawyer
    names the same provision -- by number ("6100/353"), by abbreviation
    ("HMK/353"), or by the law alone ("HMK") when they want everything under it.

    This is the field that makes "find decisions applying HMK 353" answerable.
    Production has no equivalent: its payload carries no legislation at all, so the
    query is not slow there, it is impossible.
    """
    refs = set()
    for c in citations or []:
        law_no = (c.get("law_no") or "").strip()
        short = (c.get("law_short") or "").strip()
        art = (c.get("article_no") or "").strip()
        if c.get("canonical_id"):
            refs.add(str(c["canonical_id"]))
        if law_no:
            refs.add(law_no)
            if art:
                refs.add(f"{law_no}/{art}")
        if short:
            refs.add(short)
            if art:
                refs.add(f"{short}/{art}")
    return sorted(refs)


def build_record(chunk, record, source, chunk_index, role_field):
    """One chunk + its raw court record -> one index record.

    `notes` accumulates every place we substituted or guessed, so a reviewer can
    see exactly where the data was not clean rather than trusting the output.
    """
    notes = []
    doc_id = str(record.get("doc_id"))
    title = record.get("title") or ""

    esas_year = _int_or_none(record.get("esas_year"))
    karar_year = _int_or_none(record.get("karar_year"))

    # THE INVISIBILITY FIX. esas_year drives the recent/old query split, so a null
    # means the record matches neither bucket and is never returned. kvkk has no
    # esas_year at all; aym has no karar_year. Substitute the other, and say so.
    if esas_year is None and karar_year is not None:
        esas_year = karar_year
        notes.append("esas_year_from_karar_year")
    if karar_year is None and esas_year is not None:
        karar_year = esas_year
        notes.append("karar_year_from_esas_year")

    esas_no = _int_or_none(record.get("esas_no"))
    karar_no = _int_or_none(record.get("karar_no"))

    chamber = parse_chamber(title)
    if chamber is None:
        notes.append("no_chamber_in_title")

    # chunk_generate now emits a unified `role` + `role_vocabulary` (schema v2).
    # Chunks produced before that carry only the per-source field, so read the
    # unified one first and fall back -- otherwise every chunk already on disk
    # would lose its role the moment the producer changed.
    role = chunk.get("role") or chunk.get(role_field)
    vocabulary = (chunk.get("role_vocabulary")
                  or ROLE_FIELD_TO_VOCABULARY.get(role_field))

    return {
        # --- production's fields, so the record is indexable at all ---
        "uuid": chunk["chunk_id"],
        "filename": filename_for(doc_id),
        "title": title or None,
        "court": parse_court(title, source),
        "high_court": source,
        "chamber": chamber,
        "E_no": f"{esas_year}/{esas_no}" if esas_year and esas_no else None,
        "K_no": f"{karar_year}/{karar_no}" if karar_year and karar_no else None,
        "esas_year": esas_year,
        "karar_year": karar_year,
        "esas_series": esas_no,
        "karar_series": karar_no,

        # --- ours: what makes a chunk a chunk ---
        "parent_uuid": parent_uuid_for(source, doc_id),
        "unit": "chunk",
        "schema_version": SCHEMA_VERSION,
        "text": chunk.get("text"),
        "paragraph_ids": chunk.get("source_paragraph_ids") or [],
        "chunk_index": chunk_index,
        "role": role,
        "role_vocabulary": vocabulary,
        "content_type": chunk.get("content_type"),
        "confidence": chunk.get("confidence"),
        "case_no": chunk.get("case_no"),
        "decision_date": chunk.get("decision_date"),
        "source_type": source,
        "cited_legislations": chunk.get("cited_legislations") or [],
        "law_refs": law_refs_for(chunk.get("cited_legislations")),
        "derivation_notes": notes,
    }


def build_document_record(record, source, text):
    """A WHOLE decision as one index record — the baseline unit.

    Arm A is built entirely from these. Arm B is built from these too, except for
    the decisions we have chunked, which are replaced by their chunks. So both arms
    must produce byte-identical records for the 1,025 documents they share, or the
    comparison is measuring the record builder rather than the chunking.

    `uuid` is the parent uuid here: for a whole document the decision IS the unit,
    so the document's id and its parent id are the same value. That also means a
    chunk's `parent_uuid` points at exactly the row Arm A would have returned.
    """
    doc_id = str(record.get("doc_id"))
    title = record.get("title") or ""

    esas_year = _int_or_none(record.get("esas_year"))
    karar_year = _int_or_none(record.get("karar_year"))
    notes = []
    if esas_year is None and karar_year is not None:
        esas_year = karar_year
        notes.append("esas_year_from_karar_year")
    if karar_year is None and esas_year is not None:
        karar_year = esas_year
        notes.append("karar_year_from_esas_year")

    esas_no = _int_or_none(record.get("esas_no"))
    karar_no = _int_or_none(record.get("karar_no"))
    chamber = parse_chamber(title)
    if chamber is None:
        notes.append("no_chamber_in_title")
    parent = parent_uuid_for(source, doc_id)

    return {
        "uuid": parent,
        "filename": filename_for(doc_id),
        "title": title or None,
        "court": parse_court(title, source),
        "high_court": source,
        "chamber": chamber,
        "E_no": f"{esas_year}/{esas_no}" if esas_year and esas_no else None,
        "K_no": f"{karar_year}/{karar_no}" if karar_year and karar_no else None,
        "esas_year": esas_year,
        "karar_year": karar_year,
        "esas_series": esas_no,
        "karar_series": karar_no,

        "parent_uuid": parent,
        "unit": "document",
        "schema_version": SCHEMA_VERSION,
        "text": text,
        # A whole document has no paragraph location -- that absence IS the
        # finding. Arm A structurally cannot answer "which paragraph?", and the
        # validator below is therefore only applied to chunks.
        "paragraph_ids": [],
        "chunk_index": 0,
        "role": None,
        "role_vocabulary": None,
        "content_type": None,
        "confidence": None,
        "case_no": f"{esas_year}/{esas_no}" if esas_year and esas_no else None,
        "decision_date": None,
        "source_type": source,
        "cited_legislations": [],
        "law_refs": [],
        "derivation_notes": notes,
    }


def validate(rec):
    """Every way this record would fail in production. Empty list means it is safe.

    Ordered by how the failure presents: crashes first, then silent drops, then
    filters that quietly never match.
    """
    problems = []

    missing = [f for f in INDEX_RECORD_FIELDS if f not in rec]
    if missing:
        problems.append(f"missing_fields:{','.join(missing)}")

    # Crashes. int(None) raises and 500s the request.
    for f in ("esas_year", "karar_year"):
        if not isinstance(rec.get(f), int):
            problems.append(f"{f}_not_int:{rec.get(f)!r}  (int() would raise)")

    # Silent invisibility. Without a year the record matches neither query bucket.
    if rec.get("esas_year") is None:
        problems.append("esas_year_null: invisible to BOTH the recent and old bucket")

    # 400 from Qdrant: chamber is indexed as an integer, a string looks for a
    # keyword index that does not exist.
    ch = rec.get("chamber")
    if ch is not None and not isinstance(ch, int):
        problems.append(f"chamber_not_int:{ch!r}  (Qdrant returns 400)")

    # The join key. Both stores must carry the same id or hydration finds nothing.
    for f in ("uuid", "parent_uuid"):
        if not rec.get(f):
            problems.append(f"{f}_missing: the two stores cannot be joined")

    if rec.get("high_court") not in OUR_HIGH_COURTS:
        problems.append(f"high_court_unknown:{rec.get('high_court')!r}")

    if not (rec.get("text") or "").strip():
        problems.append("text_empty: nothing to embed or return")

    # Chunk-only requirements. A whole document legitimately has neither -- that
    # absence is exactly what Arm A cannot do and Arm B can, so checking it on a
    # document would fail the baseline for being the baseline.
    if rec.get("unit") == "chunk":
        if not rec.get("paragraph_ids"):
            problems.append("paragraph_ids_empty: pinpoint citation impossible")
        if rec.get("role") is None:
            problems.append("role_null: cannot filter by reasoning part")

    return problems


def production_gap(rec):
    """Which of production's own fields are null. Not failures -- several are
    genuinely absent for regulators (kvkk has no esas number, aym no karar
    number). Reported so the gap is visible instead of assumed."""
    return [f for f in PRODUCTION_FIELDS if rec.get(f) in (None, "")]
