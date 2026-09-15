"""
chunk_generate.py -- LLM chunking pipeline (docs section 15).

One Gemini 2.5 Flash-Lite call per document produces segments + capsules together;
code then owns every field that is arithmetic, a fixed lookup, or must be
byte-identical across separate API calls. Output matches the regex pipeline's
chunks[] + reasoning_capsules[] schema, but with real Turkish reasoning_summary text
and native legislation extraction.

Self-contained: imports nothing outside llm_chunk/. The proven splitting/extraction
logic is copied verbatim into chunk_lib.py.

Usage:
    python chunk_generate.py                      # all 5 sources, DOCS_PER_SOURCE each
    python chunk_generate.py --source kvkk        # one source
    python chunk_generate.py --source aym --limit 1
"""

import argparse
import json
import os
import re
import uuid
from pathlib import Path
from typing import List, Literal, Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field, create_model

import chunk_lib as lib
import prompts

ROOT = Path(__file__).resolve().parents[2]          # llm_chunk/
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "output" / "chunk"

# Generated once with uuid.uuid4() and hardcoded. NEVER regenerate -- doing so
# would change every chunk_id ever produced and break determinism entirely.
CHUNK_NAMESPACE = uuid.UUID("f08fd9b5-14a7-46ef-b7ac-a664a1b45032")

REASONING_SUMMARY_METHOD = "llm_generated"

SOURCES = {
    "aym": {"text_field": "html_content", "granularity": "court_paragraph",
            "role_field": "firac_role", "subject_type": "right"},
    "bam": {"text_field": "content_text", "granularity": "synthetic_segment",
            "role_field": "court_reasoning_role", "subject_type": "civil_or_administrative_case"},
    "danistay": {"text_field": "content_text", "granularity": "synthetic_segment",
                 "role_field": "court_reasoning_role", "subject_type": "administrative_or_tax_dispute"},
    "first_degree": {"text_field": "content_text", "granularity": "synthetic_segment",
                     "role_field": "court_reasoning_role", "subject_type": "civil_or_administrative_case"},
    "kvkk": {"text_field": "html_content", "granularity": "synthetic_segment",
             "role_field": "regulatory_role", "subject_type": "data_controller_violation"},
    # Court of Cassation. The subject_type here is only a FALLBACK: yargitay spans
    # criminal and civil chambers (29 of 45 export rows are Ceza), so the real value
    # is resolved per document by resolve_subject_type(). The court LEVEL stays in
    # source_type -- criminal-vs-civil is subject matter, not court level.
    "yargitay": {"text_field": "html_content", "granularity": "synthetic_segment",
                 "role_field": "court_reasoning_role",
                 "subject_type": "civil_or_administrative_case"},
    # Sources 7 and 8. text_field and granularity are NULL on purpose, not
    # forgotten: every row of both is pending_extraction, so we have never seen a
    # body and cannot know whether the text will land in content_text or
    # html_content. Guessing that is precisely what went wrong with yargitay,
    # whose content_text had no newlines and whose HTML wrapped the whole
    # decision in a single <p> -- both existing strategies returned the entire
    # document as one paragraph. The role_field is safe to fix now because it
    # follows from what the institution IS, not from how its text is laid out.
    "rekabet": {"text_field": None, "granularity": None,
                "role_field": "regulatory_role",
                "subject_type": "other_competition_matter"},
    "uyusmazlik": {"text_field": None, "granularity": None,
                   "role_field": "court_reasoning_role",
                   "subject_type": "jurisdictional_dispute"},
}

# Registered sources that carry no extracted text yet. Confirmed against the
# database, not inferred from our sample: all 10,367 rekabet rows are
# status=pending_extraction and a `content_text <> ''` filter returns zero rows.
# The bodies exist only as PDFs at metadata.data.pdf_url. Everything derivable
# from metadata (case_no, decision_date, subject_type) is wired and tested; the
# paragraph strategy, role values and prompt stay unset until real text exists.
TEXT_PENDING_SOURCES = {"rekabet", "uyusmazlik"}

# Rekabet Kurumu classifies each decision itself, and that classification is a
# TYPE distinction rather than a subject: a merger clearance is a different kind
# of matter from an infringement finding. Same shape as yargitay's per-document
# criminal-vs-civil split. Read from metadata.data.decision_type and NEVER from
# chamber_id -- despite appearances chamber_id is not a chamber here, it merely
# mirrors decision_type, and its numbering already changed between two exports
# (418xxx vs 104xxx), so anything keyed on it breaks on the next one.
REKABET_SUBJECT_TYPE = {
    "Birleşme ve Devralma": "merger_or_acquisition",
    "Rekabet İhlali": "competition_infringement",
    "Menfi Tespit ve Muafiyet": "negative_clearance_or_exemption",
    "Özelleştirme": "privatisation",
    "Diğer": "other_competition_matter",
}

# Documents reserved as worked examples in fewshot/. Never chunked, because a
# document the model was shown cannot also be a document it is scored on.
FEWSHOT_DOC_IDS = {("yargitay", "1221692000")}

# A record is usable when the upstream fetch finished. Two writers populate this
# corpus and they disagree on the word: the scraper writes "fetched", the SQL
# export writes "completed". Accepting only "fetched" silently skipped all 45
# rows of the yargitay export -- zero documents, no error, no clue why.
USABLE_STATUSES = {"fetched", "completed"}

# Yargitay chamber numbering: Hukuk (civil) chambers keep their own number,
# Ceza (criminal) chambers are offset by +23. Holds on all 45 export rows and is
# more reliable than parsing the title string.
CEZA_CHAMBER_ID_MIN = 23

# What `subject_id` IS, per document. For a criminal cassation the subject is the
# OFFENCE, so subject_id "dolandiricilik" under subject_type "criminal_offence"
# answers "every fraud cassation decision" -- the query a lawyer runs. Naming it
# "criminal_case" instead would merely restate source_type and leave subject_id
# meaningless. Only yargitay varies per document; every other source keeps the
# fixed constant in SOURCES.
SUBJECT_TYPE_BY_CHAMBER = {"ceza": "criminal_offence",
                           "hukuk": "civil_or_administrative_case"}

# Set explicitly so a truncated response is reported as truncation rather than
# as a baffling "EOF while parsing a string" from the JSON parser.
#
# 65,536 is Gemini 2.5 Flash-Lite's documented ceiling. Raised from 32,768 after a
# probe: aym document 0ddd6b3c (184,377 chars, 559 paragraphs) failed with
# max_output_tokens_truncated on BOTH attempts, each stopping exactly at the old
# 32,768 cap -- half the available budget was simply unused.
MAX_OUTPUT_TOKENS = 65536

# Output tokens per input character, CALIBRATED from real runs this session:
#   aym 0.471, bam 0.504, danistay 0.502, first_degree 0.591, yargitay 0.902
# Short documents skew high because capsule text is roughly fixed-size overhead.
# 0.50 is deliberately taken from the long-document end of that range, since this
# guard only matters for long documents -- applying the short-document ratio would
# refuse things that would have succeeded.
#
# This is a HEURISTIC, not a guarantee: it refuses the one document known to fail
# (184,377 chars -> ~92,000 predicted) while letting through the next-largest
# (122,573 -> ~61,000), which is untested and may still truncate. finish_reason
# remains the authoritative signal; this only avoids paying twice to learn it.
OUTPUT_TOKENS_PER_CHAR = 0.50

# A model-vs-source text delta this large suggests a missing paragraph ref rather
# than a copying slip, so it is flagged for review instead of merely counted.
MODEL_TEXT_DELTA_CHARS = 120
MODEL_TEXT_DELTA_RATIO = 0.05

# A case number the court actually states looks like "2019/2035". Anything else
# (notably "..." from a redacted DOSYA NO line) is unreadable source, not a
# competing reading of the case number.
WELLFORMED_CASE_NO = re.compile(r"\d{4}/\d+")

# Rekabet Kurumu does not use the courts' esas/karar pair at all; its decisions
# are numbered "06-90/1142-338". Widening the shared regex to accept that would
# quietly loosen the check for all six working sources, so the per-source form is
# kept separate and only rekabet is judged by it.
CASE_NO_FORMS = {"rekabet": re.compile(r"\d{2}-\d+/\d+-\w+")}


def is_wellformed_case_no(value, source):
    """Is this a case number the court could really have printed, or is it
    unreadable source (a redacted "..." line)? The distinction decides whether a
    model/source disagreement is reported as a mismatch or as bad input data."""
    return bool(CASE_NO_FORMS.get(source, WELLFORMED_CASE_NO).fullmatch(value or ""))

DISSENT_ROLES = {"dissent"}
RULING_ROLES = {"conclusion", "outcome"}

# docs 13.1: statute/decree_law/constitution were produced by the regex extractor;
# regulation/directive were recognised but unreachable by regex and are a specific
# target for the LLM. Anything outside this set is the model inventing a category.
LEGISLATION_TYPES = {"statute", "decree_law", "constitution", "regulation", "directive"}

# Turkish ruling patterns used to cross-check the capsule outcome (docs 15.6).
# Ordered most-specific first: "İHLAL EDİLMEDİĞİNE" must be tested before
# "İHLAL EDİLDİĞİNE".
OUTCOME_PATTERNS = {
    "no_violation": [r"İHLÂL\s+EDİLMEDİĞİNE", r"İHLAL\s+EDİLMEDİĞİNE"],
    "violation": [r"İHLÂL\s+EDİLDİĞİNE", r"İHLAL\s+EDİLDİĞİNE"],
    "denied": [r"REDDİNE"],
    "affirmed": [r"ONANMASINA"],
    # Cassation dispositions (yargitay). Without these the outcome cross-check
    # finds no pattern and every yargitay capsule reads as contradicting its own
    # ruling -- the same English-key-vs-Turkish-label failure fixed twice before.
    "reversed": [r"BOZULMASINA", r"BOZULMASI"],
    "corrected_affirmed": [r"DÜZELTİLEREK"],
    "remitted": [r"TEVDİİNE"],
    "abated": [r"DÜŞMESİNE"],
    "remanded": [r"GERİ\s+ÇEVRİLMESİNE"],
}

# The model writes its outcome LABEL in Turkish while the pattern keys above are
# English, so each verdict maps to the label fragments that legitimately express
# it. "ihlal_yok" contains "ihlal", so no_violation is checked first.
OUTCOME_LABEL_FORMS = {
    "no_violation": ("ihlal_yok", "ihlal_olmad", "ihlal_edilmedi", "no_violation",
                     "ihlal_bulunmad"),
    "violation": ("ihlal", "violation"),
    "denied": ("red", "denied", "dismiss", "kabul_edilemez"),
    "affirmed": ("onan", "onama", "affirm", "onandi"),
    "reversed": ("bozma", "bozul", "reversed", "bozuldu"),
    "corrected_affirmed": ("duzelt", "düzelt", "corrected", "onan", "onama"),
    "remitted": ("tevdi", "remit", "gonderil", "gönderil"),
    "abated": ("dusme", "düşme", "abat", "ortadan_kaldir"),
    # Both stems: Turkish drops the vowel in "cevrilme" / "çevrilmesine", so
    # "çevir" does not match it. English forms too -- `outcome` is a free
    # string and the model mixes languages across sources ("ihlal_yok" but
    # "return_for_procedural_action"), so the check must accept both.
    "remanded": ("geri_cevir", "geri_çevir", "geri_cevr", "geri_çevr",
                 "remand", "return", "iade"),
}

ENGLISH_STOPWORDS = {"the", "and", "of", "was", "were", "that", "this", "court",
                     "applicant", "with", "which", "from", "have", "been"}
TURKISH_CHARS = set("çğıöşüÇĞİÖŞÜ")

# Very common Turkish words that survive ASCII-ization, so a diacritic-free Turkish
# sentence is still recognised as Turkish. Listed in both spellings where they differ.
TURKISH_MARKERS = {
    "ve", "ile", "bu", "bir", "icin", "için", "gore", "göre", "olarak", "uyarinca",
    "uyarınca", "nedeniyle", "karar", "karari", "kararı", "kararina", "kararına",
    "dava", "davanin", "davanın", "davaci", "davacı", "mahkeme", "mahkemesi",
    "hakki", "hakkı", "hakkinin", "hakkının", "basvuru", "başvuru", "basvurucu",
    "başvurucu", "ihlal", "reddine", "kabul", "edilmis", "edilmiş", "edilmedigine",
    "edilmediğine", "verilmistir", "verilmiştir", "olmadigina", "olmadığına",
    "yonunden", "yönünden", "gerekce", "gerekçe", "kurul", "kanun", "kanunun",
    "madde", "maddesi", "sayili", "sayılı", "idari", "para", "cezasi", "cezası",
}



# Response schema -- what we ask Gemini for. Deliberately excludes chunk_id,
# canonical_id, char_length, citation_granularity, chunk_label, subject_type and
# reasoning_summary_method: those are code-owned (docs 15.2-15.4).


class CitedLegislation(BaseModel):
    law_no: Optional[str] = None
    # The abbreviation as the court wrote it (T.B.K., HMK, İYUK). Turkish
    # citation is almost entirely by abbreviation, and "T.B.K. madde 56" carries
    # no law number, so without this field the citation is unidentifiable. It
    # is NOT resolved to a number here: BK means Law 818 in a 2010 decision and
    # 6098 in a 2020 one, TMK is usually 4721 but sometimes 3713. Resolution is
    # the legislation store's alias table, date-aware and ambiguity-aware.
    law_short: Optional[str] = None
    law_name: Optional[str] = None
    article_no: Optional[str] = None
    paragraph_no: Optional[str] = None
    law_date: Optional[str] = None
    legislation_type: Optional[str] = None
    confidence: Optional[str] = None
    verbatim_mention: Optional[str] = None


class SegmentBase(BaseModel):
    """`role`, `content_type`, `reasoning_stage`, `confidence` and
    `paragraph_refs` are REQUIRED and enum-constrained where a vocabulary
    exists. Verified necessary: when these were Optional with a None default,
    the model dropped `role` for entire documents at a time -- all 38 bam
    chunks, 7 of 15 first_degree -- and invented `reasoning_stage: "dissent"`,
    which is not one of the three documented stages. An optional field in a
    structured-output schema is an invitation to omit it, and a free string is
    an invitation to invent a value. Constraining the schema makes both
    impossible rather than merely detectable."""
    local_id: str
    paragraph_refs: List[str]
    text: str
    content_type: Literal["reasoning", "ruling"]
    reasoning_stage: Literal["background", "analysis", "outcome"]
    rights: Optional[List[str]] = None       # genuinely null for the four non-aym sources
    confidence: Literal["high", "low"]
    source_type: Optional[str] = None        # echoed for cross-check only
    case_no: Optional[str] = None            # echoed for cross-check only
    # legislation_type stays a free string on purpose: constraining it to the
    # five documented values would silently discard the model's signal that it
    # found a treaty, which is an open schema question (see README).
    cited_legislations: List[CitedLegislation] = Field(default_factory=list)


class Capsule(BaseModel):
    outcome: Optional[str] = None
    conclusion_sentence: Optional[str] = None
    reasoning_summary: Optional[str] = None
    opinion_type: Optional[str] = None
    supporting_local_ids: List[str] = Field(default_factory=list)
    case_no: Optional[str] = None
    subject_id: Optional[str] = None


_RESPONSE_MODELS = {}


def response_model_for(source):
    """Per-source response model: `role` is a Literal over that source's own
    vocabulary, so aym cannot return 'rule_application' and kvkk cannot return
    'facts'. Built once per source and cached."""
    if source not in _RESPONSE_MODELS:
        roles = tuple(prompts.ROLE_VOCAB[source][1])
        segment = create_model(
            f"Segment_{source}", __base__=SegmentBase, role=(Literal[roles], ...))
        _RESPONSE_MODELS[source] = create_model(
            f"DocumentResponse_{source}",
            segments=(List[segment], ...),
            capsules=(List[Capsule], ...))
    return _RESPONSE_MODELS[source]



# Code-owned field construction


def make_chunk_id(doc_id, paragraph_range):
    return str(uuid.uuid5(CHUNK_NAMESPACE, f"{doc_id}-{paragraph_range}"))


def normalise_article_key(article_no):
    """The key form of an article number. `article_no` itself stays AS WRITTEN
    ("Geçici 3", "141/A") -- the legislation store keeps both too. Only the join
    key is normalised, so that "Geçici 3", "geçici madde 3" and "GEÇİCİ MADDE 3"
    collapse to ONE key; today they would make three and silently fracture
    "find every decision citing this article".

        "Geçici 3"          -> "geçici-3"
        "GEÇİCİ MADDE 3"    -> "geçici-3"
        "Ek 5"              -> "ek-5"
        "141/A"             -> "141/a"     <- PLACEHOLDER: the store has not said how a
                                             letter-suffix article appears in article_key
        "23"                -> "23"

    Uses tr_lower, never .lower(): "GEÇİCİ".lower() would produce a dotted-i.
    """
    if not article_no:
        return "unknown"
    s = lib.tr_lower(str(article_no)).strip()
    s = re.sub(r"\bmadde(si|sinin|nin)?\b", " ", s)     # drop the word, keep the number
    s = re.sub(r"\s*/\s*", "/", s)                         # "141 / A" -> "141/a"
    s = re.sub(r"[^a-zçğıöşü0-9/]+", "-", s).strip("-")
    return s or "unknown"


def normalise_law_short(law_short):
    """'T.B.K.' / 'T.B.K' / 'Tbk' -> 'TBK'. Uses tr_upper so 'İyuk' -> 'İYUK',
    not 'IYUK' (Python's .upper() breaks Turkish dotted/dotless i)."""
    if not law_short:
        return None
    s = re.sub(r"[.\s ]+", "", str(law_short))
    s = lib.tr_upper(s)
    return s or None


def is_legislation(c):
    """Is this citation a legal instrument at all, or the court citing case law?

    The rule is deliberately conservative: reject only when the type is outside the
    documented vocabulary AND there is neither a law number nor a law name -- i.e.
    nothing whatsoever identifies a statute, decree-law, constitution, regulation or
    directive. Validated on the real corpus: drops 15 of 15 case-law citations with
    0 false positives, and leaves the one genuine treaty reference (an ECHR protocol,
    which carries a law_name) untouched.
    """
    if c.legislation_type in LEGISLATION_TYPES:
        return True
    return bool(c.law_no or c.law_name)


def make_canonical_id(law_no, article_no, legislation_type, law_name):
    """{law_no}/{article} -- the legislation store's article_key format, so the two
    systems join directly. Falls back to constitution/, then a slug of the law
    name. Returns None when nothing identifies the provision: a law cited only by
    abbreviation has law_short but no key, by design (see CitedLegislation)."""
    article = normalise_article_key(article_no)
    if law_no:
        return f"{re.sub(r'[^0-9]', '', str(law_no)) or law_no}/{article}"
    if legislation_type == "constitution":
        return f"constitution/{article}"
    if law_name:
        slug = lib.tr_lower(law_name).strip()
        slug = re.sub(r"[^a-zçğıöşü0-9]+", "-", slug).strip("-")
        return f"{slug}/{article}"
    return None   # genuinely nothing to identify it by -- don't fabricate a shared key


def paragraph_range(refs):
    """["p6","p7","p8"] -> "p6_to_p8";  ["p6"] -> "p6";  [] -> "unknown"."""
    if not refs:
        return "unknown"
    if len(refs) == 1:
        return refs[0]
    return f"{refs[0]}_to_{refs[-1]}"


def slug_right(right):
    """'Özel hayata ve aile hayatına saygı hakkı' -> 'özel_hayata_..._hakkı'.
    Verified to reproduce the gold files' values exactly. Uses tr_lower, never
    .lower(), which would produce the dotted-i artifact seen in the gold data."""
    s = re.sub(r"\(.*?\)", "", lib.tr_lower(right)).strip()
    return re.sub(r"[^a-zçğıöşü0-9]+", "_", s).strip("_")


def norm_ws(s):
    return " ".join((s or "").split())


def strip_dotted_i(s):
    """Remove the COMBINING DOT ABOVE (U+0307) left behind when 'İ' is lowercased
    badly. The model produces its own snake_case labels and returned
    'i̇hlal_yok' -- i + U+0307 + 'hlal_yok' -- which is a DIFFERENT string
    from 'ihlal_yok', so grouping or filtering by outcome would silently split
    the same verdict into two buckets. Exactly the artifact docs 12.2 describes.

    Removes only U+0307, and deliberately does NOT NFD-decompose first: 'ş'
    decomposes to s + COMBINING CEDILLA (U+0327), so a blanket combining-mark
    strip turns 'karşı_oy' into 'karsı_oy' and corrupts real Turkish. The two
    marks are distinct codepoints, so the targeted removal is safe.

    This is Unicode normalisation, not re-deciding a Gemini-owned value -- the
    same category as computing char_length."""
    return s.replace("̇", "") if s else s



# Source data


def parse_metadata(record):
    """metadata is a JSON *string*. bam/danistay/first_degree have no "data" key at
    all, so callers must tolerate {} rather than assuming metadata["data"] exists."""
    raw = record.get("metadata")
    if not raw:
        return {}
    try:
        return json.loads(raw) if isinstance(raw, str) else raw
    except (ValueError, TypeError):
        return {}


def compute_case_no(record, source):
    if source == "kvkk":
        data = parse_metadata(record).get("data") or {}
        raw = data.get("decision_number_raw") or ""
        case_no = raw.strip().lstrip(":-").strip()
        # 25 records carry a stray leading colon; one holds four comma-separated
        # numbers. karar_year/karar_no are clean integers, so fall back to them.
        if not re.fullmatch(r"\d{4}/\d+", case_no):
            if record.get("karar_year") and record.get("karar_no"):
                return f'{record["karar_year"]}/{record["karar_no"]}'
        return case_no or None
    if source == "rekabet":
        # Rekabet numbers its decisions "06-90/1142-338", which decomposes as
        # {karar_year%100}-{meeting_no}/{esas_no}-{karar_no} -- verified on all
        # 250 rows with zero mismatches. esas_year is null for every one of them,
        # so the generic branch below would return None. Taken verbatim rather
        # than rebuilt from the columns: the raw string is what the Kurul itself
        # prints and what a lawyer would search for.
        raw = (parse_metadata(record).get("data") or {}).get("decision_number_raw") or ""
        return raw.strip() or None
    if record.get("esas_year") and record.get("esas_no"):
        return f'{record["esas_year"]}/{record["esas_no"]}'
    return None


def _iso_from_dotted(s):
    """'15.09.2021' -> '2021-09-15'. bam/first_degree metadata uses DD.MM.YYYY."""
    m = re.fullmatch(r"\s*(\d{1,2})\.(\d{1,2})\.(\d{4})\s*", s or "")
    return f"{int(m.group(3)):04d}-{int(m.group(2)):02d}-{int(m.group(1)):02d}" if m else None


# Rulings close with a formulaic dated sentence:
#   "... 28/12/2022 tarihinde oyçokluğuyla karar verildi."   (danistay)
#   "... 20.02.2018 gününde oybirliğiyle karar verildi."     (yargitay)
# BOTH the separator and the adverb vary, and both variations are load-bearing.
# The earlier pattern required "/" and "tarihinde", which matched only 15 of 45
# yargitay documents: they mostly use "." and split 28 "tarihinde" / 17
# "gününde" as a chamber house style. Accepting both takes yargitay to 45/45
# with no nulls, and leaves every existing source's date unchanged (verified on
# all 12 previously generated documents).
RE_RULING_DATE = re.compile(
    r"(\d{1,2}[./]\d{1,2}[./]\d{4})\s*(?:tarihinde|gününde)", re.IGNORECASE)


def compute_decision_date(record, source, paragraphs):
    """ISO decision date, or (None, reason). The legislation store needs it to pick
    the law version in force and to resolve abbreviations (BK = 818 or 6098
    depending on the year), so a wrong date is worse than a null one -- karar_year
    alone is never promoted to a date.

    Verified per source:  aym/kvkk -> metadata.data.decision_date (ISO already);
    bam/first_degree -> metadata.decision_date (DD.MM.YYYY); danistay -> nothing in
    metadata at all, so the ruling sentence in the text is the only source.
    """
    meta = parse_metadata(record)
    inner = (meta.get("data") or {}).get("decision_date")
    if inner and re.fullmatch(r"\d{4}-\d{2}-\d{2}", str(inner)[:10]):
        return str(inner)[:10], None
    # yargitay stores it flat as karar_tarihi (DD.MM.YYYY, present on all 300).
    top = meta.get("decision_date") or meta.get("karar_tarihi")
    if top:
        iso = _iso_from_dotted(top) or lib._iso_date(top.replace(".", "/"))
        if iso:
            return iso, None
    for para in reversed(paragraphs):                  # the ruling is at the end
        m = RE_RULING_DATE.search(para)
        if m:
            # _iso_date splits on "/" only, but the widened pattern also matches
            # dot-separated dates ("13.01.2026 gününde"). Without normalising the
            # separator it returns None and this loop returned a SILENT null --
            # no date and no flag, the worst of both. Keep scanning on a failed
            # parse instead of giving up on the first match.
            iso = lib._iso_date(m.group(1).replace(".", "/"))
            if iso:
                return iso, None
    return None, "decision_date_unavailable"


def resolve_subject_type(record, source):
    """Per-document subject_type where the source needs it, else the per-source
    constant. Yargitay spans criminal and civil chambers, so a single fixed value
    is wrong for whichever half it does not describe: 29 of 45 export rows are
    Ceza. chamber_id is the primary discriminator, the title is the fallback."""
    default = SOURCES[source]["subject_type"]
    if source == "yargitay":
        cid = record.get("chamber_id")
        if isinstance(cid, int):
            return SUBJECT_TYPE_BY_CHAMBER["ceza" if cid >= CEZA_CHAMBER_ID_MIN else "hukuk"]
        m = re.search(r"\b(Ceza|Hukuk)\s+Dairesi", record.get("title") or "", re.IGNORECASE)
        return SUBJECT_TYPE_BY_CHAMBER.get(lib.tr_lower(m.group(1)), default) if m else default
    if source == "rekabet":
        # An unmapped value falls back to the source default rather than being
        # slugified on the fly: a decision_type we have not seen is new Kurul
        # vocabulary that should be read and mapped deliberately, not invented
        # here under a name nothing else in the corpus uses.
        dtype = (parse_metadata(record).get("data") or {}).get("decision_type")
        return REKABET_SUBJECT_TYPE.get((dtype or "").strip(), default)
    return default


# "SUÇ : Nitelikli hırsızlık" / "Suç : Taksirle yaralama" / "DAVA TÜRÜ : ALACAK".
# Case varies by CHAMBER, not by accident: 3./8./11./14./15./18. Ceza write
# "SUÇ", 10./12./16. Ceza write "Suç". A case-sensitive match finds 13 of 45 and
# silently loses 8 -- the same Turkish-casing trap as the .lower() bug.
RE_SUBJECT_LABEL = re.compile(
    r"^\s*(suç(?:lar)?|dava\s+türü|dava)\s*:\s*(.+)$", re.IGNORECASE)


def candidate_subject(record, source, paragraphs):
    """The subject the document states about itself, if any -- the same mechanism
    as aym's candidate rights: code supplies, Gemini narrows. Present on 30 of 45
    yargitay export rows; the other 15 state nothing and the model derives one.

    Returns (label_kind, raw_value) or (None, None). "DAVA :" is weaker than the
    others -- 9. Hukuk uses it for the full prayer for relief rather than a short
    subject tag -- so it is returned as a hint and never used as a constraint.
    """
    if source != "yargitay":
        return None, None
    for p in paragraphs[:8]:                      # always inside the header block
        m = RE_SUBJECT_LABEL.match(p)
        if m:
            kind = re.sub(r"\s+", "_", lib.tr_lower(m.group(1)))
            return kind, m.group(2).strip()
    return None, None


def candidate_rights(record, source):
    """aym only, and only the individual-application metadata variant: 12 of 200
    records are norm-review decisions with no examination_results at all."""
    if source != "aym":
        return []
    data = parse_metadata(record).get("data") or {}
    out = []
    for e in data.get("examination_results") or []:
        r = e.get("right")
        if r:
            s = slug_right(r)
            if s and s not in out:
                out.append(s)
    return out


def extract_paragraphs_html_br(raw):
    """Paragraphs from HTML that separates them with <br>, not <p>.

    Yargitay HTML wraps the whole decision in ONE <p align=justify> and uses <br>
    between lines, so bs4's find_all("p") returns a single element -- the entire
    document as one paragraph. Its content_text is no better: the <br> tags were
    stripped without inserting whitespace, gluing headings to body text
    ("talep etmistir.II. CEVAPDavali idare..."), so splitlines() also yields one
    paragraph. Both existing strategies fail on this source, hence a third.
    """
    from html import unescape
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(unescape(raw), "html.parser")
    for tag in soup.find_all(["br", "p"]):
        tag.insert_after("\n")
    text = soup.get_text().replace("\xa0", " ")
    return [p for p in (norm_ws(line) for line in text.split("\n")) if p]


# How each source's paragraphs are recovered. Named explicitly rather than
# inferred from the field name, because two sources store HTML and need
# different extractors -- inferring from "html_content" silently picked the
# wrong one for yargitay.
PARAGRAPH_STRATEGY = {
    "aym": "html_p", "kvkk": "html_p",
    "yargitay": "html_br",
    "bam": "lines", "danistay": "lines", "first_degree": "lines",
}


def extract_paragraphs(record, source):
    """The returned list IS the text Gemini sees, and the text chunks are later
    assembled from, so every consumer must reproduce it exactly."""
    field = SOURCES[source]["text_field"]
    if field is None or source not in PARAGRAPH_STRATEGY:
        # Reached only if a text-pending source somehow acquires text without
        # anyone choosing how to split it. A named error here is the difference
        # between "nobody has picked a paragraph strategy for rekabet yet" and a
        # bare KeyError two frames down that reads like a typo.
        raise NotImplementedError(
            f"{source}: no paragraph strategy chosen yet. Every {source} record is "
            f"pending_extraction upstream, so no decision body has ever been seen "
            f"and the strategy cannot be picked from evidence. Read 2-3 real "
            f"decisions, then add entries to SOURCES[{source!r}]['text_field'] and "
            f"PARAGRAPH_STRATEGY -- and mirror them into retrieval_test/corpus.py, "
            f"which must extract byte-identically or grounding checks fail.")
    raw = record.get(field) or ""
    if not raw.strip():
        return []
    strategy = PARAGRAPH_STRATEGY[source]
    if strategy == "html_p":
        return lib.aym_extract_paragraph_texts(raw)
    if strategy == "html_br":
        return extract_paragraphs_html_br(raw)
    # content_text is newline-separated with NO blank lines (verified: bam 36 lines /
    # 19k chars, danistay 62, first_degree 25, and zero occurrences of a blank line in
    # any of them). Splitting on blank lines returned the whole document as a single
    # paragraph, which made paragraph_refs useless. norm_ws also collapses the \xa0
    # non-breaking spaces danistay text is full of.
    return [p for p in (norm_ws(x) for x in raw.splitlines()) if p]


def pick_documents(records, source, n):
    """Filter empty-text records BEFORE taking n, so the picked documents are
    guaranteed usable. kvkk has 11 empty records (pending_extraction/pending_fetch);
    taking the slice first could spend the whole sample on them."""
    field = SOURCES[source]["text_field"]
    if field is None:
        # No text field determined because no row has ever carried text. Counted
        # as "all empty" rather than crashing, so run() can report the real
        # upstream reason instead of a stack trace.
        return [], len(records), 0
    has_text = [r for r in records
                if r.get("status") in USABLE_STATUSES and (r.get(field) or "").strip()]
    usable = [r for r in has_text
              if (source, r.get("doc_id")) not in FEWSHOT_DOC_IDS]
    # Reported separately: "no text upstream" and "reserved as a worked example"
    # are different facts, and yargitay has both (297 pending + 1 example).
    return usable[:n], len(records) - len(has_text), len(has_text) - len(usable)



# Gemini


def build_client():
    from google import genai
    project = os.getenv("GOOGLE_CLOUD_PROJECT")
    location = os.getenv("GOOGLE_CLOUD_LOCATION", "global")
    key_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    if not key_path:
        raise SystemExit("FAILED: GOOGLE_APPLICATION_CREDENTIALS not set. Check llm_chunk/.env")
    if not Path(key_path).is_file():
        raise SystemExit(f"FAILED: key file not found at {key_path}\nFix the path in llm_chunk/.env")
    if not project:
        raise SystemExit("FAILED: GOOGLE_CLOUD_PROJECT not set. Check llm_chunk/.env")
    # vertexai=True is correct for google-genai 1.59.x. Newer Google docs show
    # enterprise=True under the "Gemini Enterprise Agent Platform" rename; that
    # kwarg does not exist in this version.
    return genai.Client(vertexai=True, project=project, location=location)


def call_gemini(client, model, system_instruction, user_content, schema, perturb=False):
    """Primary call is deterministic: temperature 0, fixed seed, no penalties.

    `perturb=True` is used only for the retry, and only because a deterministic
    retry is a no-op -- re-issuing an identical temperature-0 call reproduces
    the identical failure, including a degenerate repetition loop. Greedy
    decoding is precisely the regime where a model can get stuck emitting the
    same citation object until it exhausts the output budget; there is no
    sampling noise to break the cycle. A frequency penalty plus a little
    temperature breaks it. Any document that needed the retry is recorded as
    non-reproducible rather than quietly presented as deterministic output.
    """
    from google.genai import types
    cfg = types.GenerateContentConfig(
        temperature=0.15 if perturb else 0,
        top_p=1,
        seed=42,
        frequency_penalty=0.6 if perturb else None,
        max_output_tokens=MAX_OUTPUT_TOKENS,
        thinking_config=types.ThinkingConfig(thinking_budget=0),
        response_mime_type="application/json",
        response_schema=schema,
        system_instruction=system_instruction,
    )
    return client.models.generate_content(model=model, contents=user_content, config=cfg)


def finish_reason_of(resp):
    try:
        fr = resp.candidates[0].finish_reason
    except (AttributeError, IndexError, TypeError):
        return None
    return getattr(fr, "name", None) or str(fr)



# Post-processing (docs section 6 order)


def resolve_refs(seg, n_paras):
    """paragraph_refs -> validated 1-based paragraph indices, plus bad refs."""
    idxs, bad = [], []
    for ref in seg.paragraph_refs:
        m = re.fullmatch(r"p(\d+)", str(ref).strip())
        if not m:
            bad.append(str(ref))
            continue
        i = int(m.group(1))
        if not 1 <= i <= n_paras:
            bad.append(str(ref))
            continue
        if i not in idxs:
            idxs.append(i)
    return sorted(idxs), bad


def build_chunks(parsed, record, source, doc_id, case_no, paragraphs, decision_date):
    """Returns (chunks, local_id -> [chunk_id], flags).

    `text` is DERIVED by joining the source paragraphs the model referenced, never
    copied from the model's own `text` field. Verified necessary: asked to reproduce a
    5,411-char merge verbatim, the model silently dropped two characters mid-word at a
    paragraph boundary, and elsewhere lowercased headings ("DAVA" -> "Dava"). The
    model's real judgment is WHICH paragraphs belong together; assembling them is
    mechanical, so code owns it and grounding holds by construction rather than being
    checked after the fact. Divergence from the model's own text is still recorded, in
    the same spirit as the legislation cross-check -- surfaced, never silently resolved.
    """
    cfg = SOURCES[source]
    role_field = cfg["role_field"]
    chunks, id_map, flags = [], {}, []
    minor_diffs = []
    n_paras = len(paragraphs)

    for seg in parsed.segments:
        idxs, bad = resolve_refs(seg, n_paras)
        for b in bad:
            flags.append(f"invalid_paragraph_ref:{seg.local_id}:{b}")
        if not idxs:
            flags.append(f"segment_has_no_valid_refs:{seg.local_id}")
            continue

        refs = [f"p{i}" for i in idxs]
        base_range = paragraph_range(refs)
        seg_text = "\n".join(paragraphs[i - 1] for i in idxs)

        # A gap inside the claimed range means the model skipped a paragraph, and
        # because text is assembled from the refs that content is silently lost.
        # Seen for real: danistay 971050200 referenced p30,p32,p33,p34 and dropped
        # 293 characters that were item 2 of a five-item ruling.
        missing = sorted(set(range(idxs[0], idxs[-1] + 1)) - set(idxs))
        if missing:
            dropped = sum(len(paragraphs[m - 1]) for m in missing)
            flags.append(f"paragraph_refs_gap:{seg.local_id}:"
                         f"skipped_{'_'.join('p%d' % m for m in missing)}:{dropped}chars")

        # Divergence between the model's copy and the real paragraph join. Small
        # deltas are the model tidying text while copying (dropped characters,
        # lowercased headings) and do not affect what gets stored, so they are
        # counted rather than flagged. A delta large enough to be a whole paragraph
        # means the refs probably missed one, which DOES change the stored text.
        m_norm, d_norm = norm_ws(seg.text), norm_ws(seg_text)
        if m_norm != d_norm:
            delta = abs(len(m_norm) - len(d_norm))
            if delta >= MODEL_TEXT_DELTA_CHARS or (
                    d_norm and delta / len(d_norm) > MODEL_TEXT_DELTA_RATIO):
                flags.append(f"model_text_differs:{seg.local_id}:"
                             f"{len(m_norm)}vs{len(d_norm)}")
            else:
                minor_diffs.append(seg.local_id)

        if seg.source_type and seg.source_type != source:
            flags.append(f"source_type_mismatch:{seg.local_id}:{seg.source_type}")
        if seg.case_no and case_no and seg.case_no != case_no:
            # Two very different situations, kept apart deliberately. A well-formed
            # but different number means the model read a real alternate case number
            # off the document and the metadata may be wrong -- worth a human look.
            # An ill-formed value means the document redacts its own case number
            # (real example: "DOSYA NO : ..." in bam/726546200, which has 43 such
            # redactions), so the model reported the redaction instead of echoing the
            # value we supplied. That is a source-data property, not a disagreement.
            kind = ("case_no_mismatch" if is_wellformed_case_no(seg.case_no, source)
                    else "case_no_unreadable_in_source")
            flags.append(f"{kind}:{seg.local_id}:{seg.case_no}")

        role = seg.role
        if seg.content_type == "ruling" and role not in RULING_ROLES:
            flags.append(f"content_type_role_inconsistent:{seg.local_id}:{role}")

        # Size cap BEFORE ids are final, so each piece gets its own chunk_id.
        pieces = lib.split_by_size_cap(seg_text)
        for j, piece in enumerate(pieces, 1):
            rng = base_range if len(pieces) == 1 else f"{base_range}_p{j}"
            chunk_id = make_chunk_id(doc_id, rng)
            id_map.setdefault(seg.local_id, []).append(chunk_id)

            cites, seen_cites = [], set()
            for c in seg.cited_legislations:
                # Belt and braces against the repetition loop: the same
                # provision cited identically twice in one segment is noise,
                # and one real response emitted 561 copies of a single
                # Anayasa article before exhausting the output budget.
                fingerprint = (c.law_no, c.article_no, c.paragraph_no,
                               norm_ws(c.verbatim_mention))
                if fingerprint in seen_cites:
                    continue
                seen_cites.add(fingerprint)
                if c.verbatim_mention and norm_ws(c.verbatim_mention) not in norm_ws(piece):
                    # Attach the citation only to the piece it actually appears in;
                    # flag only when it is absent from the whole segment.
                    if norm_ws(c.verbatim_mention) not in norm_ws(seg_text):
                        flags.append(f"verbatim_mention_not_in_text:{seg.local_id}")
                    continue
                canonical = make_canonical_id(
                    c.law_no, c.article_no, c.legislation_type, c.law_name)
                if not is_legislation(c):
                    # Not a legal instrument at all -- almost always the court citing
                    # its own precedent ("Mehmet Serif Ay (B. No: 2012/1181)",
                    # "Ibrahim Er ve digerleri"). Measured: 15 of 89 citations, 17%,
                    # every one of them AYM case law. Storing them here pollutes
                    # "find every decision citing this law" with case-law noise, and
                    # they carry nothing to join on. Dropped, but FLAGGED -- a silent
                    # drop would hide the model ignoring an explicit instruction.
                    #
                    # AYM cites its own precedent constantly and that IS valuable; it
                    # belongs in a cited_decisions field (AYM metadata already carries
                    # referenced_decisions). Out of scope here.
                    flags.append(f"dropped_non_legislation:{seg.local_id}:"
                                 f"{(c.verbatim_mention or '')[:40]}")
                    continue
                if c.legislation_type not in LEGISLATION_TYPES:
                    flags.append(f"unknown_legislation_type:{seg.local_id}:"
                                 f"{c.legislation_type}")
                if canonical is None:
                    # No law_no, not the constitution, and no law_name either -- there
                    # is nothing to join on, so it cannot be deduplicated at all.
                    flags.append(f"citation_not_identifiable:{seg.local_id}:"
                                 f"{c.legislation_type}")
                law_short = normalise_law_short(c.law_short)
                if law_short and not c.law_no:
                    # Cited by abbreviation only. Kept identifiable via law_short;
                    # resolution to a number is the legislation store's job.
                    flags.append(f"abbreviation_without_law_no:{seg.local_id}:{law_short}")
                cites.append({
                    "canonical_id": canonical,
                    "legislation_type": c.legislation_type,
                    "law_no": c.law_no,
                    "law_short": law_short,
                    "law_name": c.law_name,
                    "article_no": c.article_no,
                    "paragraph_no": c.paragraph_no,
                    "verbatim_mention": c.verbatim_mention,
                    "law_date": c.law_date,
                    "confidence": c.confidence,
                })
                if c.article_no and c.confidence == "low":
                    flags.append(f"confidence_contradiction:{seg.local_id}")

            chunk = {
                "chunk_id": chunk_id,
                "chunk_label": f"{doc_id}-{rng}",
                "source_type": source,
                "case_no": case_no,
                "decision_date": decision_date,
                "citation_granularity": cfg["granularity"],
                "source_paragraph_ids": list(refs),
                "text": piece,
                "char_length": len(piece),
                "content_type": seg.content_type,
                "firac_role": role if source == "aym" else None,
                "reasoning_stage": seg.reasoning_stage,
                "rights": (seg.rights or None) if source == "aym" else None,
                "confidence": seg.confidence,
            }
            if role_field != "firac_role":
                chunk[role_field] = role
            chunk["cited_legislations"] = cites
            chunk["_role"] = role          # internal, stripped before writing
            chunks.append(chunk)

    return chunks, id_map, flags, minor_diffs


def build_capsules(parsed, chunks, id_map, source, case_no, cand_rights, decision_date,
                   subject_type):
    cfg = SOURCES[source]
    capsules, flags = [], []
    role_of = {c["chunk_id"]: c.get("_role") for c in chunks}
    ruling_text = " ".join(c["text"] for c in chunks if c.get("content_type") == "ruling")

    for cap in parsed.capsules:
        supporting = []
        for lid in cap.supporting_local_ids:
            if lid not in id_map:
                flags.append(f"unresolved_local_id:{lid}")
                continue
            supporting.extend(id_map[lid])       # extend: a split segment adds all pieces

        if cap.supporting_local_ids and not supporting:
            flags.append("capsule_has_no_resolvable_support")

        opinion = cap.opinion_type or "majority"
        is_dissent_capsule = opinion.startswith("dissent")
        has_dissent_support = any(role_of.get(cid) in DISSENT_ROLES for cid in supporting)
        if is_dissent_capsule != has_dissent_support:
            flags.append(f"opinion_type_support_mismatch:{opinion}")

        if cap.outcome and ruling_text:
            up = lib.tr_upper(ruling_text)
            matched = [k for k, pats in OUTCOME_PATTERNS.items()
                       if any(re.search(p, up) for p in pats)]
            # The label is Turkish ("ihlal_yok"); the pattern keys are English.
            # Comparing them directly reported every aym capsule as contradicting
            # its own ruling, which was a bug in the check, not in the data. Also
            # normalise the dotted-i first, and skip rulings that state both a
            # violation and a non-violation (different rights in one decision),
            # which cannot adjudicate the label either way.
            label = lib.tr_lower(strip_dotted_i(cap.outcome))
            ambiguous = "violation" in matched and "no_violation" in matched
            if matched and not ambiguous and not any(
                    frag in label for k in matched
                    for frag in OUTCOME_LABEL_FORMS.get(k, ())):
                flags.append(f"outcome_vs_ruling_text:{label}:"
                             f"ruling_suggests_{'/'.join(matched)}")

        subject_id = strip_dotted_i(cap.subject_id) or "unspecified"
        if source == "aym" and cand_rights and subject_id not in cand_rights:
            flags.append(f"subject_id_not_in_candidates:{subject_id}")
        # subject_id is a RETRIEVAL field: "unspecified" makes the capsule
        # unfindable by subject, and a topic-query test found 4 of 14 capsules
        # carrying it -- a securities case, a tax-fraud case and a carriage-damage
        # case among them, all of which plainly state what they are about. The one
        # place it is legitimate is an aym norm-review decision, which genuinely
        # has no examination_results and so no right to name.
        if subject_id == "unspecified" and not (source == "aym" and not cand_rights):
            flags.append("subject_id_unspecified")

        if cap.case_no and case_no and cap.case_no != case_no:
            kind = ("capsule_case_no_mismatch"
                    if is_wellformed_case_no(cap.case_no, source)
                    else "capsule_case_no_unreadable_in_source")
            flags.append(f"{kind}:{cap.case_no}")

        # Both free-text capsule fields must be Turkish: Stage 1 retrieval searches
        # the capsule's text against Turkish queries, so English in either one is at
        # best dead weight. This is the mistake docs 7.3 caught only by a retrieval
        # test. Note it diverges deliberately from docs 6, which specified a
        # "plain-English answer" for conclusion_sentence.
        for field, value in (("reasoning_summary", cap.reasoning_summary),
                             ("conclusion_sentence", cap.conclusion_sentence)):
            text = value or ""
            if not text.strip():
                flags.append(f"{field}_empty")
            elif not looks_turkish(text):
                flags.append(f"{field}_not_turkish")

        capsules.append({
            "case_no": case_no,
            "source_type": source,
            "decision_date": decision_date,
            "subject_type": subject_type,
            "subject_id": subject_id,
            "opinion_type": opinion,
            "outcome": strip_dotted_i(cap.outcome),
            "conclusion_sentence": cap.conclusion_sentence,
            "reasoning_summary": cap.reasoning_summary,
            # Code-owned, always. Never read from the model: a system may not
            # self-certify its own audit trail (docs 15.4).
            "reasoning_summary_method": REASONING_SUMMARY_METHOD,
            "supporting_chunk_ids": supporting,
        })
    return capsules, flags


def looks_turkish(text):
    """Cheap language check. The same mistake -- English prose in a Turkish corpus --
    happened once already in this project and was caught only by a retrieval test.

    Turkish-specific characters alone are not a sufficient signal: diacritic-free
    Turkish is common in scraped legal text ("Bankanin kusuru ispatlanamadi ve dava
    reddedildi"), and a one-sentence conclusion_sentence is short enough that their
    absence proves nothing. So an English-word ratio decides against, and either
    diacritics or common Turkish words decide for.
    """
    letters = [c for c in text if c.isalpha()]
    words = re.findall(r"[a-zçğıöşü]+", lib.tr_lower(text))
    if not letters or not words:
        return False
    english_ratio = sum(1 for w in words if w in ENGLISH_STOPWORDS) / len(words)
    if english_ratio >= 0.08:
        return False
    tr_ratio = sum(1 for c in letters if c in TURKISH_CHARS) / len(letters)
    if tr_ratio >= 0.01:
        return True
    return any(w in TURKISH_MARKERS for w in words)


def legislation_crosscheck(chunks):
    """Run the copied regex extractor independently on the same text and record
    disagreement. Never merged, never silently preferred either way (docs 15.3)."""
    gemini_set, regex_set = set(), set()
    for ch in chunks:
        for c in ch.get("cited_legislations") or []:
            gemini_set.add((c.get("law_no"), c.get("article_no")))
        for c in lib.extract_cited_legislations(ch.get("text") or ""):
            regex_set.add((c.get("law_no"), c.get("article_no")))
    return {
        "only_gemini": sorted(f"{a}/{b}" for a, b in gemini_set - regex_set),
        "only_regex": sorted(f"{a}/{b}" for a, b in regex_set - gemini_set),
    }



# Driver


def process_document(client, model, record, source, stats):
    doc_id = record.get("doc_id")
    case_no = compute_case_no(record, source)
    cand = candidate_rights(record, source)
    paragraphs = extract_paragraphs(record, source)
    if not paragraphs:
        return None, None, {"doc_id": doc_id, "case_no": case_no,
                            "reason": "no_paragraphs_extracted", "failed_checks": []}, None

    # Pre-flight size check. Costs nothing and turns a confusing two-attempt
    # truncation into a named, up-front refusal. Verified against a real failure:
    # aym 0ddd6b3c, 184,377 chars / 559 paragraphs, exhausted the output budget
    # twice before reporting anything useful.
    doc_chars = sum(len(p) for p in paragraphs)
    predicted = int(doc_chars * OUTPUT_TOKENS_PER_CHAR)
    if predicted > MAX_OUTPUT_TOKENS:
        return None, None, {
            "doc_id": doc_id, "case_no": case_no,
            "reason": "document_too_long_for_one_call",
            "error": (f"{doc_chars:,} characters in {len(paragraphs)} paragraphs needs "
                      f"roughly {predicted:,} output tokens, over the "
                      f"{MAX_OUTPUT_TOKENS:,} cap. Split the document or raise the cap; "
                      f"not attempted, so no tokens were spent."),
            "failed_checks": [], "raw_response": None}, None

    schema = response_model_for(source)
    subj_kind, subj_value = candidate_subject(record, source, paragraphs)
    system_instruction = prompts.build_system_instruction(
        source, case_no, cand,
        subject_hint=(subj_kind, subj_value) if subj_kind else None)
    user_content = prompts.build_user_content(paragraphs)

    raw_text, parsed, last_err, finish, perturbed = None, None, None, None, False
    for attempt in (1, 2):
        # Attempt 2 perturbs on purpose: an identical temperature-0 retry cannot
        # produce a different result, so the old "retry once" was a no-op for
        # every deterministic failure.
        perturb = attempt == 2
        try:
            resp = call_gemini(client, model, system_instruction, user_content,
                               schema, perturb=perturb)
            raw_text = resp.text
            finish = finish_reason_of(resp)
            stats["input_tokens"] += getattr(resp.usage_metadata, "prompt_token_count", 0) or 0
            stats["output_tokens"] += getattr(resp.usage_metadata, "candidates_token_count", 0) or 0
            if finish == "MAX_TOKENS":
                # Name the real cause. Truncated JSON otherwise surfaces as a
                # baffling "EOF while parsing a string", which hides it.
                raise ValueError(
                    f"response hit max_output_tokens ({MAX_OUTPUT_TOKENS}) and was "
                    f"truncated; {(raw_text or '').count(chr(34) + 'verbatim_mention' + chr(34))} "
                    f"citation objects emitted -- likely a repetition loop")
            parsed = schema.model_validate_json(raw_text)
            perturbed = perturb
            break
        except Exception as e:                   # noqa: BLE001 - any failure retries once
            last_err = f"{type(e).__name__}: {e}"
            if attempt == 2:
                reason = ("max_output_tokens_truncated" if finish == "MAX_TOKENS"
                          else "api_or_parse_failed")
                return None, None, {"doc_id": doc_id, "case_no": case_no,
                                    "reason": reason, "error": last_err,
                                    "finish_reason": finish, "failed_checks": [],
                                    "raw_response": raw_text}, None
    if perturbed:
        stats["perturbed_retries"] += 1
        print(f"  [{source}] {doc_id}  NOTE: needed a perturbed retry "
              f"(temperature 0.15 + frequency penalty) -- this document's output is "
              f"NOT byte-reproducible")

    decision_date, date_flag = compute_decision_date(record, source, paragraphs)
    chunks, id_map, flags, minor_diffs = build_chunks(
        parsed, record, source, doc_id, case_no, paragraphs, decision_date)
    if date_flag:
        flags.append(date_flag)
    stats["minor_text_diffs"] += len(minor_diffs)
    capsules, cap_flags = build_capsules(parsed, chunks, id_map, source, case_no, cand,
                                         decision_date, resolve_subject_type(record, source))
    flags += cap_flags
    disagreement = legislation_crosscheck(chunks)

    for ch in chunks:
        ch.pop("_role", None)

    review = None
    if flags:
        review = {"doc_id": doc_id, "case_no": case_no, "reason": "validation_flags",
                  "failed_checks": flags, "legislation_disagreement": disagreement,
                  "raw_response": json.loads(raw_text) if raw_text else None}
    return chunks, capsules, review, disagreement


def run(sources, limit, only_doc_id=None, out_dir=None):
    load_dotenv(ROOT / ".env")
    model = os.getenv("MODEL", "gemini-2.5-flash-lite")
    per_source = limit if limit is not None else int(os.getenv("DOCS_PER_SOURCE", "2"))
    client = build_client()
    # A probe run must never write into output/chunk/: the 30 retrieval queries
    # are keyed to the 10 doc_ids generated there, so a different document set
    # landing in that directory silently invalidates every one of them.
    out_root = Path(out_dir) if out_dir else OUTPUT_DIR
    if not out_root.is_absolute():
        out_root = ROOT / out_root
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"model    : {model}")
    print(f"project  : {os.getenv('GOOGLE_CLOUD_PROJECT')}  location: {os.getenv('GOOGLE_CLOUD_LOCATION')}")
    print(f"docs/src : {per_source}\n")

    grand = {"ok": 0, "flagged": 0, "failed": 0}
    for source in sources:
        path = DATA_DIR / f"{source}.json"
        if not path.is_file():
            print(f"=== {source} === SKIPPED, no {path.name} in data/")
            continue
        records = json.loads(path.read_text(encoding="utf-8"))
        if only_doc_id:
            # Deliberate bypass of the reserved-example skip-set: the example
            # document has to be generated once in order to become the example.
            wanted = [x.strip() for x in str(only_doc_id).split(",") if x.strip()]
            by_id = {str(r.get("doc_id")): r for r in records}
            picked = [by_id[w] for w in wanted if w in by_id]
            missing = [w for w in wanted if w not in by_id]
            if missing:
                print(f"  !! doc_id(s) not found in {source}: {missing}")
            skipped_empty = skipped_fewshot = 0
            if not picked:
                print(f"=== {source} === doc_id {only_doc_id} not in this source")
                continue
        else:
            picked, skipped_empty, skipped_fewshot = pick_documents(
                records, source, per_source)

        # A registered source with no usable document must say WHY. The bland
        # skip line is the failure mode that hid 45 yargitay documents for a
        # whole round: the export wrote status="completed", the filter accepted
        # only "fetched", and the run reported zero documents with no error and
        # no clue. Never call Gemini for these -- there is nothing to send, and
        # their prompts are placeholders.
        if source in TEXT_PENDING_SOURCES and not picked:
            print(f"=== {source} === BLOCKED: 0 of {len(records)} records have "
                  f"extracted text")
            print(f"    every row is status=pending_extraction; the decision bodies "
                  f"are PDFs at metadata.data.pdf_url.")
            print(f"    Confirmed database-wide, not sampled: all 10,367 rekabet rows "
                  f"are unextracted and a")
            print(f"    `content_text <> ''` filter returns no rows. Blocked upstream, "
                  f"not a pipeline fault.")
            print(f"    Metadata-derived fields (case_no, decision_date, subject_type) "
                  f"are wired and tested;")
            print(f"    paragraph strategy and role values stay unset until real text "
                  f"exists.\n")
            continue

        all_chunks, all_caps, reviews = [], [], []
        stats = {"input_tokens": 0, "output_tokens": 0, "minor_text_diffs": 0,
                 "perturbed_retries": 0}
        only_g, only_r, ok, flagged, failed = 0, 0, 0, 0, 0

        for rec in picked:
            chunks, capsules, review, disagreement = process_document(
                client, model, rec, source, stats)
            if chunks is None:                   # hard failure before post-processing
                reviews.append(review)
                failed += 1
                print(f"  [{source}] {rec.get('doc_id')}  FAILED: {review.get('reason')}")
                continue
            all_chunks.extend(chunks)
            all_caps.extend(capsules)
            only_g += len(disagreement["only_gemini"])
            only_r += len(disagreement["only_regex"])
            if review:
                reviews.append(review)
                flagged += 1
                print(f"  [{source}] {rec.get('doc_id')}  flagged: "
                      f"{', '.join(review['failed_checks'][:4])}")
            else:
                ok += 1
            print(f"  [{source}] {rec.get('doc_id')}  {len(chunks)} chunks, "
                  f"{len(capsules)} capsules")

        # Never replace a good output file with an empty one. When every
        # document in a source fails, the old run is the better artifact and
        # silently clobbering it loses real work -- which is exactly what
        # happened to aym.json once.
        out_path = out_root / f"{source}.json"
        if not all_chunks and out_path.is_file():
            print(f"  !! every {source} document failed -- KEEPING the previous "
                  f"{out_path.name} rather than overwriting it with an empty file")
        else:
            out_path.write_text(
                json.dumps({"chunks": all_chunks, "reasoning_capsules": all_caps},
                           ensure_ascii=False, indent=2), encoding="utf-8")
        review_path = out_root / f"{source}_review.json"
        if reviews:
            review_path.write_text(
                json.dumps(reviews, ensure_ascii=False, indent=2), encoding="utf-8")
        elif review_path.is_file():
            # A clean run must not leave the previous run's flags lying around --
            # a stale review file reads as unresolved problems that no longer exist.
            review_path.unlink()
            print(f"  (removed stale {review_path.name}: this run had no flags)")

        grand["ok"] += ok
        grand["flagged"] += flagged
        grand["failed"] += failed
        print(f"=== {source} ===  {len(picked)} docs | ok {ok} | flagged {flagged} | "
              f"failed {failed} | skipped_empty {skipped_empty} | "
              + (f"reserved_as_example {skipped_fewshot} | " if skipped_fewshot else "") +
              f"{len(all_chunks)} chunks | {len(all_caps)} capsules")
        print(f"  legislation: gemini-only {only_g}, regex-only {only_r}")
        print(f"  minor text diffs (model copy vs source join; stored text unaffected): "
              f"{stats['minor_text_diffs']}")
        if stats["perturbed_retries"]:
            print(f"  perturbed retries: {stats['perturbed_retries']} "
                  f"(those documents are NOT byte-reproducible)")
        print(f"  tokens: in {stats['input_tokens']}, out {stats['output_tokens']}\n")

    print(f"TOTAL: ok {grand['ok']} | flagged {grand['flagged']} | failed {grand['failed']}")
    print(f"Output: {out_root}")


def main():
    ap = argparse.ArgumentParser(description="LLM chunking via Gemini 2.5 Flash-Lite")
    ap.add_argument("--source", choices=sorted(SOURCES), help="only this source")
    ap.add_argument("--limit", type=int, help="documents per source (overrides .env)")
    ap.add_argument("--out-dir", help="write output here instead of output/chunk/. Use for "
                                      "probe runs -- writing a different document set into "
                                      "output/chunk/ invalidates the retrieval queries")
    ap.add_argument("--doc-id", help="generate exactly these documents (comma-separated), "
                                     "even if reserved in FEWSHOT_DOC_IDS. Used to build a "
                                     "worked example, or to choose a test set deliberately "
                                     "rather than taking whichever documents come first")
    args = ap.parse_args()
    run([args.source] if args.source else list(SOURCES), args.limit, args.doc_id,
        args.out_dir)


if __name__ == "__main__":
    main()
