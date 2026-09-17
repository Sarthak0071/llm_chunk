"""
chunker.py -- raw court-decision JSON -> chunks[] + reasoning_capsules[].

Three steps per document, so the same code runs now (direct calls) and later
(Vertex AI batch):
  1. prepare            pure code: paragraphs, code-owned fields, the requests
  2. call_request       the only API step. STRUCTURE returns segments (which
                        paragraphs belong together, their role); CAPSULES
                        returns one capsule per decision and separate opinion,
                        pointing at paragraphs; LAWS returns the citations of
                        every paragraph, one window of paragraphs per request so
                        no answer reaches the output ceiling. None needs
                        another's answer: sent in parallel now, in the same
                        batch job later.
  3. assemble_document  pure code, what the docs (section 15.2) give to code:
                        citations attached by paragraph, ids, dates, case
                        numbers, text assembly, size cap, derived fields, and
                        the mechanical checks -- above all that EVERY paragraph
                        of the document is in exactly one chunk.

    python chunker.py --source kvkk --limit 10
    python chunker.py --doc-id-file ..\\..\\output\\kvkk_first10.json --out-dir output/chunk

Self-contained: imports nothing outside llm_chunk/. Tests live in test_chunk.py.
"""

import argparse
import json
import os
import re
import sys
import uuid
from collections import Counter
from pathlib import Path
from typing import List, Literal, Optional

from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field, create_model

import chunk_lib as lib
import prompts

ROOT = Path(__file__).resolve().parents[2]          # llm_chunk/
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "output" / "chunk"

# Fixed forever: regenerating it would change every chunk_id ever produced.
CHUNK_NAMESPACE = uuid.UUID("f08fd9b5-14a7-46ef-b7ac-a664a1b45032")
CHUNK_SCHEMA_VERSION = 2
REASONING_SUMMARY_METHOD = "llm_generated"

MAX_OUTPUT_TOKENS = 65535          # Gemini 2.5 Flash-Lite's own ceiling (exclusive at 65536); cannot be raised
# LAWS goes in windows so no answer can reach that ceiling: the citations of one
# 561-paragraph norm review needed ~196,000 output tokens as a single answer.
LAWS_WINDOW_PARAS = 60
LAWS_WINDOW_CHARS = 30000
# Structure and capsule answers stay small even for the longest decision (9,849
# output tokens for 561 paragraphs); a document is held back only when its input
# could not fit the model at all.
MAX_DOCUMENT_CHARS = 2_000_000
PARALLEL_REQUESTS = 6              # requests of one document in flight at once
FRAGMENT_CHARS = 40                # a shorter segment joins a same-role neighbour
WORKERS = 4                        # documents processed in parallel
USABLE_STATUSES = {"fetched", "completed"}
TEXT_PENDING_SOURCES = {"rekabet", "uyusmazlik"}  # no decision body extracted upstream yet

SOURCES = {
    "aym": {"granularity": "court_paragraph", "role_field": "firac_role", "subject_type": "right"},
    "bam": {"granularity": "synthetic_segment", "role_field": "court_reasoning_role",
            "subject_type": "civil_or_administrative_case"},
    "danistay": {"granularity": "synthetic_segment", "role_field": "court_reasoning_role",
                 "subject_type": "administrative_or_tax_dispute"},
    "first_degree": {"granularity": "synthetic_segment", "role_field": "court_reasoning_role",
                     "subject_type": "civil_or_administrative_case"},
    "kvkk": {"granularity": "synthetic_segment", "role_field": "regulatory_role",
             "subject_type": "data_controller_violation"},
    "yargitay": {"granularity": "synthetic_segment", "role_field": "court_reasoning_role",
                 "subject_type": "civil_or_administrative_case"},
    "rekabet": {"granularity": None, "role_field": "regulatory_role",
                "subject_type": "other_competition_matter"},
    "uyusmazlik": {"granularity": None, "role_field": "court_reasoning_role",
                   "subject_type": "jurisdictional_dispute"},
}
ROLE_VOCABULARY = {"firac_role": "firac", "court_reasoning_role": "court_reasoning",
                   "regulatory_role": "regulatory"}


# Vocabularies. English keys for machine values, Turkish for content. Every
# closed field is a Literal in the response schema, so an illegal value cannot
# be parsed, let alone stored. The prompt renders the same lists with glosses.


RULING_ROLES = frozenset({"conclusion", "outcome"})
DISSENT_ROLES = frozenset({"dissent"})
CATCHALL_ROLE = {"aym": "unknown", "kvkk": "background", "rekabet": "background"}

STAGE_OF_ROLE = {"facts": "background", "issue": "background", "other": "background",
                 "unknown": "background", "background": "background",
                 "rule": "analysis", "application": "analysis", "rule_application": "analysis",
                 "dissent": "analysis", "analysis": "analysis",
                 "conclusion": "outcome", "outcome": "outcome"}


def derive(role):
    """(content_type, reasoning_stage) as a rule of the role (docs 14.1)."""
    return ("ruling" if role in RULING_ROLES else "reasoning"), STAGE_OF_ROLE[role]


OUTCOME_GLOSS = {
    "violation": "ihlal edildiğine (bireysel başvuru)",
    "no_violation": "ihlal edilmediğine",
    "inadmissible": "kabul edilemez olduğuna",
    "abated": "düşmesine / düşürülmesine",
    "annulled": "iptaline (norm denetimi; Danıştay'da işlemin iptali)",
    "denied": "reddine / esastan reddine / itirazın reddine / şikâyetin reddine / hukuka uygun bulunduğuna",
    "granted": "kabulüne / kaldırılmasına",
    "partially_granted": "kısmen kabul, kısmen ret / kısmen iptal / kısmen bozma",
    "affirmed": "onanmasına",
    "corrected_affirmed": "düzeltilerek onanmasına",
    "reversed": "bozulmasına",
    "remanded": "geri çevrilmesine / iadesine / kaldırılarak mahkemesine gönderilmesine",
    "remitted": "tevdiine",
    "no_jurisdiction": "görevsizlik / yetkisizlik nedeniyle ret; KVKK: talebin Kanun kapsamında değerlendirilemeyeceğine",
    "dismissed_procedural": "usulden ret / süre aşımı / yöntemine uygun olmayan başvuru / dilekçenin reddi",
    "transferred": "dosyanın başka bir mahkemeye veya daireye gönderilmesine (asıl hüküm buysa)",
    "no_decision_needed": "karar verilmesine yer olmadığına",
    "fine_imposed": "idari para cezası uygulanmasına",
    "instruction_issued": "veri sorumlusunun talimatlandırılmasına / hatırlatılmasına / bilgi verilmesine",
    "disciplinary_referral": "Kanun m.18/3: kamu kurumu için sorumlular hakkında işlem yapılmasına",
    "no_action": "yapılacak bir işlem olmadığına / işlem yapılmasına yer olmadığına",
    "procedural_objection": "SADECE karşı oy için: usule, yönteme veya göreve ilişkin muhalefet",
    "other": "hiçbiri uymuyorsa; conclusion_sentence hükmü açıkça yazmalı",
}
# `procedural_objection` was offered to dissents and became an easy exit: 12 of
# 16 dissents took it while arguing the merits. A procedural dissent now names
# the disposition it leads to (dismissed_procedural, no_jurisdiction, remanded).
_TAIL = ("no_decision_needed", "other")
_COURT = ("granted", "partially_granted", "denied", "affirmed", "reversed", "remanded",
          "dismissed_procedural", "no_jurisdiction", "transferred", "abated")
OUTCOME_BY_KIND = {
    "aym_individual_application": ("violation", "no_violation", "inadmissible", "abated",
                                   "dismissed_procedural", "no_jurisdiction") + _TAIL,
    "aym_norm_review": ("annulled", "denied", "partially_granted", "no_jurisdiction",
                        "dismissed_procedural", "remanded") + _TAIL,
    "yargitay_hukuk": ("affirmed", "corrected_affirmed", "reversed", "partially_granted",
                       "remanded", "remitted", "abated", "denied", "transferred",
                       "no_jurisdiction", "dismissed_procedural") + _TAIL,
    "yargitay_ceza": ("affirmed", "corrected_affirmed", "reversed", "partially_granted",
                      "remanded", "remitted", "abated", "denied", "transferred",
                      "no_jurisdiction", "dismissed_procedural") + _TAIL,
    "danistay": ("affirmed", "corrected_affirmed", "reversed", "partially_granted", "denied",
                 "annulled", "granted", "no_jurisdiction", "transferred", "remanded",
                 "dismissed_procedural") + _TAIL,
    "bam": _COURT + _TAIL,
    "first_degree": _COURT + _TAIL,
    "uyusmazlik": _COURT + _TAIL,
    "kvkk": ("fine_imposed", "instruction_issued", "disciplinary_referral", "no_action",
             "denied", "no_jurisdiction") + _TAIL,
    "rekabet": ("fine_imposed", "instruction_issued", "no_action", "denied", "granted") + _TAIL,
}
OPINION_KINDS = ("majority", "dissent", "concurring", "board_decision")
LEGISLATION_TYPES = ("statute", "decree_law", "constitution", "regulation", "directive")
RESPONSE_LEGISLATION_TYPES = LEGISLATION_TYPES + ("treaty", "not_legislation")
CONFIDENCE = ("high", "low")
LAW_SHORT_RE = re.compile(r"^[A-ZÇĞİÖŞÜ]{2,8}$")

# AYM's own structured verdicts, shown to the STRUCTURE call as a hint.
AYM_INDIVIDUAL_OUTCOME = {
    "İhlal": "violation", "İhlal Olmadığı": "no_violation",
    "Açıkça Dayanaktan Yoksunluk": "inadmissible", "Başvuru Yollarının Tüketilmemesi": "inadmissible",
    "Konu Bakımından Yetkisizlik": "inadmissible", "Kişi Bakımından Yetkisizlik": "inadmissible",
    "Zaman Bakımından Yetkisizlik": "inadmissible", "Yer Bakımından Yetkisizlik": "inadmissible",
    "Süre Aşımı": "inadmissible", "Anayasal ve Kişisel Önemin Olmaması": "inadmissible",
    "Başvurunun Reddi": "dismissed_procedural", "Düşme": "abated",
    "İşlemden Kaldırılma": "abated", "İncelenmesine Yer Olmadığı": "no_decision_needed",
}
AYM_NORM_RESULT = {
    "Esas İptal": "annulled", "Esas - Ret": "denied", "İlk - Ret": "denied / dismissed_procedural",
    "İlk - İşin Geri Çevrilmesi": "remanded",
    "Esas - Karar Verilmesine/İncelenmesine Yer Olmadığı": "no_decision_needed",
    "İlk - Karar Verilmesine/İncelenmesine Yer Olmadığı": "no_decision_needed",
}


def cost_usd(tokens_in, tokens_out):
    """USD for a token count. Gemini 2.5 Flash-Lite paid tier: $0.10 input and $0.40
    output per 1M tokens (ai.google.dev pricing, checked 2026-09-17); a batch job
    costs half. PRICE_INPUT_PER_M / PRICE_OUTPUT_PER_M in .env override both."""
    return (tokens_in * float(os.getenv("PRICE_INPUT_PER_M", "0.10"))
            + tokens_out * float(os.getenv("PRICE_OUTPUT_PER_M", "0.40"))) / 1e6


# Text helpers



def norm_ws(s):
    return " ".join((s or "").split())


def slug(s):
    """Turkish-safe snake_case, diacritics kept, U+0307 artifact removed."""
    s = lib.tr_lower((s or "").replace("̇", "")).strip()
    s = re.sub(r"[.'’`]", "", s)
    return re.sub(r"[^a-zçğıöşüâîû0-9]+", "_", s).strip("_")


def slug_right(right):
    """'Özel hayata ve aile hayatına saygı hakkı' -> the gold files' exact slug."""
    s = re.sub(r"\(.*?\)", "", lib.tr_lower(right)).strip()
    return re.sub(r"[^a-zçğıöşü0-9]+", "_", s).strip("_")


ENGLISH_STOPWORDS = {"the", "and", "of", "was", "were", "that", "this", "court", "applicant",
                     "with", "which", "from", "have", "been"}
TURKISH_MARKERS = {"ve", "ile", "bu", "bir", "için", "icin", "göre", "gore", "olarak", "karar",
                   "kararı", "dava", "mahkeme", "hakkı", "hakki", "başvuru", "basvuru", "ihlal",
                   "reddine", "kabul", "verilmiştir", "verilmistir", "kanun", "madde", "sayılı",
                   "sayili", "idari", "para", "cezası", "cezasi", "kurul", "nedeniyle", "uyarınca"}


def looks_turkish(text):
    letters = [c for c in text if c.isalpha()]
    words = re.findall(r"[a-zçğıöşü]+", lib.tr_lower(text))
    if not letters or not words:
        return False
    if sum(1 for w in words if w in ENGLISH_STOPWORDS) / len(words) >= 0.08:
        return False
    if sum(1 for c in letters if c in "çğıöşüÇĞİÖŞÜ") / len(letters) >= 0.01:
        return True
    return any(w in TURKISH_MARKERS for w in words)


def language_of(text):
    """'tr', 'en' or None. Capsule text must be in the decision's own language."""
    if looks_turkish(text):
        return "tr"
    words = re.findall(r"[a-z]+", (text or "").lower())
    if words and sum(1 for w in words if w in ENGLISH_STOPWORDS) / len(words) >= 0.08:
        return "en"
    return None


LOWER_RE = re.compile(r"[a-zçğıöşüâîû]")


def is_heading_line(text):
    """Short line with no lowercase letter: heading, docket label, name. No vocabulary."""
    t = (text or "").strip()
    return len(t) <= 80 and not LOWER_RE.search(t)




# 1. READ -- paragraphs with their boundaries
#
# Measured on 60 records per source: bam/danistay/first_degree content_text has
# one line per paragraph; aym/kvkk/yargitay content_text is ONE line with the
# breaks removed and words glued ("Karar ÖzetiKarar Tarihi:"). Same words in
# both columns; only the HTML keeps the boundaries for those three.


BLOCK_TAGS = ("p", "li", "h1", "h2", "h3", "h4", "tr", "div", "table", "ul", "ol",
              "tbody", "blockquote", "section", "article")


def html_paragraphs(raw):
    """Walk the HTML in document order. Block tags and <br> end a paragraph;
    strings and inline tags accumulate. A table row becomes one line."""
    from html import unescape
    from bs4 import BeautifulSoup, NavigableString, Comment
    soup = BeautifulSoup(unescape(raw), "html.parser")
    out, buf = [], []

    def flush():
        txt = norm_ws(" ".join(buf).replace("\xa0", " "))
        if txt:
            out.append(txt)
        buf.clear()

    def walk(node):
        for child in node.children:
            if isinstance(child, Comment):
                continue
            if isinstance(child, NavigableString):
                buf.append(str(child))
            elif child.name == "br":
                flush()
            elif child.name in BLOCK_TAGS:
                flush()
                walk(child)
                flush()
            else:
                walk(child)
    walk(soup)
    flush()
    return out


def plain_text(s):
    if s and "<" in s and ">" in s:
        from html import unescape
        from bs4 import BeautifulSoup
        return norm_ws(BeautifulSoup(unescape(s), "html.parser").get_text(" "))
    return norm_ws(s)


# Scraper artefacts that sit on their own line in almost every Danıştay record
# ("Karar İçeriği", "ee3", "0", "2023/1999999") and lone punctuation lines.
# Exact tokens only -- no vocabulary. Removed before numbering so they do not
# become chunks; the words of the removed lines are the only words allowed to go.
JUNK_LINE_RE = re.compile(r"^(?:Karar İçeriği|ee3|0|\d{4}/1999999|[.:;\-–…]+)$")
WORD_RE = re.compile(r"\w+")


def lost_words(raw_text, texts, junk=()):
    """{word: count} of raw_text words (as a multiset) that `texts` do not carry.
    The words of `junk` lines are allowed to be missing. Empty = nothing lost."""
    want = Counter(WORD_RE.findall(raw_text or ""))
    want.subtract(Counter(WORD_RE.findall(" ".join(junk))))
    have = Counter(WORD_RE.findall(" ".join(texts)))
    return {w: c - have[w] for w, c in want.items() if c > have[w]}


def raw_column(record):
    """The longer text column as plain text, and its lines (for the junk-line
    allowance). aym/kvkk/yargitay: the HTML; bam/danistay/first_degree: either."""
    ct = record.get("content_text") or ""
    html = record.get("html_content") or ""
    html_text, ct_text = plain_text(html), norm_ws(ct)
    if len(html_text) >= len(ct_text):
        return html_text, (html_paragraphs(html) if html.strip() else [])
    return ct_text, [norm_ws(l) for l in ct.splitlines() if norm_ws(l)]


def paragraphs(record):
    """(paragraphs, lost, plain_content_text). content_text when it has real
    lines, else the HTML walk; junk lines dropped. `lost` is the word-level
    extraction check: every word of the longer raw column must be in the
    paragraphs. If the first reading loses words the other one is tried; what is
    still lost is returned (and flagged), never silently accepted."""
    ct = record.get("content_text") or ""
    html = record.get("html_content") or ""
    lines = [norm_ws(l) for l in ct.splitlines() if norm_ws(l)]
    walked = None

    def walk():
        nonlocal walked
        if walked is None:
            walked = html_paragraphs(html) if html.strip() else []
        return walked

    first = lines if (len(lines) >= 5 or not html.strip()) else walk()
    if len(first) <= 1 and html.strip() and len(walk()) > len(first):
        first = walk()
    raw_text, raw_lines = raw_column(record)
    junk = [q for q in raw_lines if JUNK_LINE_RE.match(q.strip())]

    def clean(ps):
        keep = [q for q in ps if not JUNK_LINE_RE.match(q.strip())]
        return keep, lost_words(raw_text, keep, junk)

    paras, lost = clean(first)
    if lost:
        other = walk() if first is lines else lines
        if other:
            alt, alt_lost = clean(other)
            if sum(alt_lost.values()) < sum(lost.values()):
                paras, lost = alt, alt_lost
    return paras, lost, norm_ws(ct)



# 2. DERIVE -- code-owned fields from metadata (docs 14.1, 15.4)



def parse_metadata(record):
    raw = record.get("metadata")
    if not raw:
        return {}
    try:
        return json.loads(raw) if isinstance(raw, str) else raw
    except (ValueError, TypeError):
        return {}


def compute_case_no(record, source):
    if source == "kvkk":
        raw = ((parse_metadata(record).get("data") or {}).get("decision_number_raw") or "")
        case_no = raw.strip().lstrip(":-").strip()
        if not re.fullmatch(r"\d{4}/\d+", case_no) and record.get("karar_year") and record.get("karar_no"):
            return f'{record["karar_year"]}/{record["karar_no"]}'
        return case_no or None
    if source == "rekabet":
        raw = ((parse_metadata(record).get("data") or {}).get("decision_number_raw") or "")
        return raw.strip() or None
    if record.get("esas_year") and record.get("esas_no"):
        return f'{record["esas_year"]}/{record["esas_no"]}'
    return None


RE_RULING_DATE = re.compile(r"(\d{1,2}[./]\d{1,2}[./]\d{4})\s*(?:tarihinde|gününde)", re.IGNORECASE)


def _iso(d):
    m = re.fullmatch(r"\s*(\d{1,2})[./](\d{1,2})[./](\d{4})\s*", d or "")
    return f"{int(m.group(3)):04d}-{int(m.group(2)):02d}-{int(m.group(1)):02d}" if m else None


def compute_decision_date(record, paras):
    meta = parse_metadata(record)
    inner = (meta.get("data") or {}).get("decision_date")
    if inner and re.fullmatch(r"\d{4}-\d{2}-\d{2}", str(inner)[:10]):
        return str(inner)[:10]
    top = meta.get("decision_date") or meta.get("karar_tarihi")
    if top and _iso(top):
        return _iso(top)
    for p in reversed(paras):                  # the ruling sentence is at the end
        m = RE_RULING_DATE.search(p)
        if m and _iso(m.group(1)):
            return _iso(m.group(1))
    return None


def aym_variant(record):
    data = parse_metadata(record).get("data") or {}
    if data.get("examination_results"):
        return "individual_application"
    if data.get("examined_norms") or data.get("application_type"):
        return "norm_review"
    return "individual_application"


def kind_of(record, source):
    """The KIND of document: prompts and the outcome list key on this."""
    if source == "aym":
        return "aym_" + aym_variant(record)
    if source == "yargitay":
        cid = record.get("chamber_id")
        ceza = (cid >= 23) if isinstance(cid, int) else bool(
            re.search(r"\bCeza\s+Dairesi", record.get("title") or "", re.IGNORECASE))
        return "yargitay_ceza" if ceza else "yargitay_hukuk"
    return source


def resolve_subject_type(record, source, kind):
    if kind == "yargitay_ceza":
        return "criminal_offence"
    if kind == "aym_norm_review":
        return "constitutional_norm_review"
    return SOURCES[source]["subject_type"]


def candidate_rights(record):
    out = []
    for e in (parse_metadata(record).get("data") or {}).get("examination_results") or []:
        s = slug_right(e.get("right") or "")
        if s and s not in out:
            out.append(s)
    return out


def examined_norms(record):
    out = []
    for norm in (parse_metadata(record).get("data") or {}).get("examined_norms") or []:
        for art in norm.get("articles") or []:
            out.append({"law": (norm.get("norm_code_name") or "").strip(),
                        "article": (art.get("article_no") or "").strip(),
                        "clause": art.get("clause"), "result": art.get("review_type_result")})
    return out


def aym_hint(record, kind):
    """The court's own verdicts as text for the structure prompt (structured metadata)."""
    data = parse_metadata(record).get("data") or {}
    lines = []
    if kind == "aym_individual_application":
        for e in data.get("examination_results") or []:
            if e.get("right"):
                lines.append(f"  - {slug_right(e['right'])}: {e.get('outcome')} "
                             f"(-> {AYM_INDIVIDUAL_OUTCOME.get((e.get('outcome') or '').strip(), '?')})")
    elif kind == "aym_norm_review":
        for e in examined_norms(record):
            lines.append(f"  - {e['law']} m.{e['article']}: {e['result']} "
                         f"(-> {AYM_NORM_RESULT.get((e['result'] or '').strip(), '?')})")
    return "\n".join(lines) or None



# 3. SCHEMA -- what the model may return. Closed wherever a vocabulary exists.



class CitedLegislation(BaseModel):
    # No `confidence`: code sets it (article identified -> high), as the docs define it.
    law_no: Optional[str] = None
    law_short: Optional[str] = None
    law_name: Optional[str] = None
    article_no: Optional[str] = None
    paragraph_no: Optional[str] = None
    law_date: Optional[str] = None
    legislation_type: Literal[RESPONSE_LEGISLATION_TYPES]
    verbatim_mention: Optional[str] = None


class ParagraphLaws(BaseModel):
    ref: str
    # Required, no default: an optional list is one the model may leave out, and
    # it did (a KVKK answer omitted citations on all 36 segments of a decision
    # that cites 6698 on nearly every page).
    cited_legislations: List[CitedLegislation]


class LawsResponse(BaseModel):
    """The LAWS call: every paragraph, in order, with the legislation it names."""
    paragraphs: List[ParagraphLaws]


class SegmentBase(BaseModel):
    local_id: str
    paragraph_refs: List[str]
    confidence: Literal[CONFIDENCE]


class CapsuleBase(BaseModel):
    conclusion_sentence: str = Field(min_length=20)
    reasoning_summary: str = Field(min_length=80)
    dissent_authors: List[str] = Field(default_factory=list)
    supporting_paragraph_refs: List[str] = Field(min_length=1)


_MODELS = {}


def structure_model_for(source, kind, candidates=()):
    """STRUCTURE answer: segments with the roles of this source; for AYM individual
    applications `rights` limited to the candidate rights the court's metadata names."""
    key = ("structure", source, kind, tuple(candidates))
    if key not in _MODELS:
        roles = tuple(prompts.ROLE_VOCAB[source][1])
        extra = {"rights": (Optional[List[Literal[tuple(candidates)]]], None)} if candidates else {}
        segment = create_model(f"Segment_{kind}", __base__=SegmentBase, role=(Literal[roles], ...), **extra)
        _MODELS[key] = create_model(f"Structure_{kind}", segments=(List[segment], ...))
    return _MODELS[key]


class RulingItem(BaseModel):
    text: str
    kind: Literal["decision", "cost_or_fee", "forwarding_or_service"]


def _require_ruling_items(schema, _cls):
    """`ruling_items` is REQUIRED in the schema sent to Gemini: the model first lists
    every item of the operative ruling and says which are decisions, then writes
    capsules for those only (it wrote capsules for fees and for TEVDİİNE when the
    rule was prose). The pydantic default stays so earlier saved answers still parse."""
    req = schema.setdefault("required", [])
    if "ruling_items" not in req:
        req.insert(0, "ruling_items")


def capsules_model_for(source, kind, candidates=()):
    """CAPSULES answer: the ruling items, then capsules with the outcomes of this
    kind; for AYM individual applications `subject_id` limited to the candidate rights."""
    key = ("capsules", source, kind, tuple(candidates))
    if key not in _MODELS:
        subject = (Literal[tuple(candidates)], ...) if candidates else (str, Field(min_length=3))
        kinds = OPINION_KINDS if source in ("kvkk", "rekabet") else OPINION_KINDS[:3]
        capsule = create_model(f"Capsule_{kind}", __base__=CapsuleBase,
                               outcome=(Literal[tuple(OUTCOME_BY_KIND[kind])], ...),
                               opinion_type=(Literal[kinds], ...), subject_id=subject)
        _MODELS[key] = create_model(f"Capsules_{kind}",
                                    __config__=ConfigDict(json_schema_extra=_require_ruling_items),
                                    ruling_items=(List[RulingItem], Field(default_factory=list)),
                                    capsules=(List[capsule], ...))
    return _MODELS[key]


def legacy_capsules(raw_response, source, kind, candidates=()):
    """Capsules saved by the earlier design (inside the structure answer, pointing at
    segment local_ids), converted to paragraph refs so saved answers still replay.
    None when there are none."""
    caps = (raw_response or {}).get("capsules")
    if not caps:
        return None
    refs = {sg.get("local_id"): sg.get("paragraph_refs") or [] for sg in raw_response.get("segments", [])}
    converted = []
    for c in caps:
        if "supporting_paragraph_refs" not in c:
            c = dict(c, supporting_paragraph_refs=[r for lid in c.get("supporting_local_ids") or []
                                                   for r in refs.get(lid, [])])
        if c["supporting_paragraph_refs"]:
            converted.append(c)
    return capsules_model_for(source, kind, candidates).model_validate({"capsules": converted})



# Gemini



def http_options(**extra):
    """Quota and server errors retry the SAME request with backoff inside the SDK
    (identical parameters, so the chunking does not change). Only when these
    retries are exhausted does process_document try its perturbed attempt."""
    from google.genai import types
    return types.HttpOptions(retry_options=types.HttpRetryOptions(
        attempts=6, initial_delay=2, max_delay=60, exp_base=2, jitter=1,
        http_status_codes=[408, 429, 500, 502, 503, 504]), **extra)


def build_client():
    from google import genai
    for var in ("GOOGLE_APPLICATION_CREDENTIALS", "GOOGLE_CLOUD_PROJECT"):
        if not os.getenv(var):
            raise SystemExit(f"FAILED: {var} not set. Check llm_chunk/.env")
    if not Path(os.getenv("GOOGLE_APPLICATION_CREDENTIALS")).is_file():
        raise SystemExit("FAILED: credentials file not found; fix the path in llm_chunk/.env")
    return genai.Client(vertexai=True, project=os.getenv("GOOGLE_CLOUD_PROJECT"),
                        location=os.getenv("GOOGLE_CLOUD_LOCATION", "global"),
                        http_options=http_options())


def call_gemini(client, model, system, content, schema, perturb=False, max_tokens=None):
    """Temperature 0 and a fixed seed: the same document chunks identically.
    `perturb` is only for the retry -- an identical deterministic retry is a no-op."""
    from google.genai import types
    cfg = types.GenerateContentConfig(
        temperature=0.4 if perturb else 0, top_p=1, seed=42,
        frequency_penalty=1.0 if perturb else None,
        max_output_tokens=max_tokens or MAX_OUTPUT_TOKENS,
        thinking_config=types.ThinkingConfig(thinking_budget=0),
        response_mime_type="application/json", response_schema=schema,
        system_instruction=system)
    return client.models.generate_content(model=model, contents=content, config=cfg)


def schema_preflight(client, model, pairs, stats=None):
    """Before any document is sent: one ~20-token call per (source, kind) with the
    real structure, capsules and laws schemas. A schema the API rejects
    (400 "too many states", unsupported keyword) fails HERE with the API's own
    message, not on document 1 of 80. Cost: well under a cent."""
    from google.genai import types
    problems = []
    for source, kind in sorted(set(pairs)):
        cands = ("mülkiyet_hakkı",) if kind == "aym_individual_application" else ()
        for label, schema in (("structure", structure_model_for(source, kind, cands)),
                              ("capsules", capsules_model_for(source, kind, cands)),
                              ("laws", LawsResponse)):
            try:
                resp = client.models.generate_content(
                    model=model, contents="ping",
                    config=types.GenerateContentConfig(
                        temperature=0, max_output_tokens=16,
                        thinking_config=types.ThinkingConfig(thinking_budget=0),
                        response_mime_type="application/json", response_schema=schema,
                        system_instruction="Reply with any valid instance."))
                if stats is not None:
                    _usage(resp, stats, "preflight")
            except Exception as exc:                            # noqa: BLE001
                msg = str(exc)
                if "400" in msg or "INVALID_ARGUMENT" in msg or "schema" in msg.lower():
                    problems.append(f"{source}/{kind} {label} schema rejected: {msg[:300]}")
                # other errors (quota, network) are not schema problems; the run reports them per document
    return problems


def _usage(resp, stats, name):
    u = getattr(resp, "usage_metadata", None)
    tin, tout = getattr(u, "prompt_token_count", 0) or 0, getattr(u, "candidates_token_count", 0) or 0
    stats["input_tokens"] += tin
    stats["output_tokens"] += tout
    stats[f"{name}_input_tokens"] += tin
    stats[f"{name}_output_tokens"] += tout


def _finish(resp):
    try:
        return getattr(resp.candidates[0].finish_reason, "name", None)
    except (AttributeError, IndexError, TypeError):
        return None



# 4. REPAIR -- mechanical, before any id exists



def _ref_idxs(refs, n):
    """Valid paragraph numbers of `refs`, deduplicated, in DOCUMENT order."""
    out = set()
    for r in refs:
        m = re.fullmatch(r"p(\d+)", str(r).strip())
        if m and 1 <= int(m.group(1)) <= n:
            out.add(int(m.group(1)))
    return sorted(out)


def _first_ref(seg, n):
    return (_ref_idxs(seg.paragraph_refs, n) or [10 ** 9])[0]


def normalise_segments(segments, paras):
    """A paragraph listed twice keeps its FIRST segment; segments are put in
    document order; a segment under FRAGMENT_CHARS that is not a heading joins
    the previous segment if it has the SAME role, else the next one if that has
    the same role, else it stays its own chunk (a short ruling line such as
    "karar verilmiştir." must never become part of the reasoning). Returns
    (segments, alias, log): alias maps a merged/emptied local_id to the
    local_id that now holds its paragraphs, so capsule pointers still resolve."""
    n, owner, out, alias, log = len(paras), {}, [], {}, []
    for seg in segments:
        keep, stolen = [], set()
        for i in _ref_idxs(seg.paragraph_refs, n):
            if i in owner:
                stolen.add(owner[i])
                log.append(f"duplicate_ref_removed:{seg.local_id}:p{i}")
            else:
                owner[i] = seg.local_id
                keep.append(f"p{i}")
        if not keep:
            if stolen:
                alias[seg.local_id] = sorted(stolen)[0]
            log.append(f"segment_emptied:{seg.local_id}")
            continue
        out.append(seg.model_copy(update={"paragraph_refs": keep}))
    out.sort(key=lambda sg: _first_ref(sg, n))

    def joined(a, b):
        update = {"paragraph_refs": [f"p{i}" for i in _ref_idxs(list(a.paragraph_refs) + list(b.paragraph_refs), n)]}
        if "rights" in type(a).model_fields:
            update["rights"] = list(dict.fromkeys((a.rights or []) + (b.rights or []))) or None
        return a.model_copy(update=update)

    merged = []
    for k, seg in enumerate(out):
        text = " ".join(paras[i - 1] for i in _ref_idxs(seg.paragraph_refs, n))
        if len(text) < FRAGMENT_CHARS and not is_heading_line(text):
            if merged and merged[-1].role == seg.role:
                alias[seg.local_id] = merged[-1].local_id
                log.append(f"fragment_merged:{seg.local_id}->{merged[-1].local_id}:{text[:30]}")
                merged[-1] = joined(merged[-1], seg)
                continue
            if k + 1 < len(out) and out[k + 1].role == seg.role:
                alias[seg.local_id] = out[k + 1].local_id
                log.append(f"fragment_merged:{seg.local_id}->{out[k + 1].local_id}:{text[:30]}")
                out[k + 1] = joined(out[k + 1], seg)
                continue
        merged.append(seg)
    # The operative ruling is ONE chunk: consecutive ruling segments (heading, each
    # numbered item, costs, the closing line) are joined -- one chunk per line split
    # a first-instance HÜKÜM into 6 chunks. Structural, no vocabulary.
    final = []
    for seg in merged:
        if final and final[-1].role in RULING_ROLES and seg.role in RULING_ROLES \
                and _first_ref(seg, n) == _ref_idxs(final[-1].paragraph_refs, n)[-1] + 1:
            alias[seg.local_id] = final[-1].local_id
            log.append(f"ruling_segments_joined:{seg.local_id}->{final[-1].local_id}")
            final[-1] = joined(final[-1], seg)
            continue
        final.append(seg)
    return final, alias, log



# 5. LAWS -- the citations of every paragraph, from the second request


def laws_windows(paras):
    """[(first, last), ...]: consecutive paragraph numbers covering every paragraph
    exactly once, each window at most LAWS_WINDOW_PARAS paragraphs and
    LAWS_WINDOW_CHARS characters (one longer paragraph is a window of its own)."""
    out, start, size = [], 1, 0
    for i, para in enumerate(paras, 1):
        if i > start and (i - start >= LAWS_WINDOW_PARAS or size + len(para) > LAWS_WINDOW_CHARS):
            out.append((start, i - 1))
            start, size = i, 0
        size += len(para)
    if paras:
        out.append((start, len(paras)))
    return out


def citations_by_paragraph(laws, n, log, window=None):
    """{paragraph number: [CitedLegislation, ...]} from one LAWS answer (None when
    that call failed). `window` (first, last) is the part that request answered for:
    refs outside it are dropped; paragraphs it did not list are counted, never guessed."""
    out = {}
    if laws is None:
        return out
    lo, hi = window or (1, n)
    answered, invalid = set(), 0
    for item in laws.paragraphs:
        idx = _ref_idxs([item.ref], n)
        if not idx or not lo <= idx[0] <= hi:
            invalid += 1
            continue
        answered.add(idx[0])
        if item.cited_legislations:
            out.setdefault(idx[0], []).extend(item.cited_legislations)
    span = f" in p{lo}-p{hi}" if window else ""
    if invalid:
        log.append(f"laws_invalid_ref:{invalid}{span}")
    if len(answered) < hi - lo + 1:
        log.append(f"laws_paragraphs_not_answered:{hi - lo + 1 - len(answered)} of {hi - lo + 1}{span}")
    return out



# 6. ASSEMBLE -- code-owned fields



def make_chunk_id(doc_id, rng):
    return str(uuid.uuid5(CHUNK_NAMESPACE, f"{doc_id}-{rng}"))


def normalise_article_key(article_no):
    if not article_no:
        return "unknown"
    s = lib.tr_lower(str(article_no)).strip()
    s = re.sub(r"\bmadde(si|sinin|nin)?\b", " ", s)
    s = re.sub(r"\s*/\s*", "/", s)
    return re.sub(r"[^a-zçğıöşü0-9/]+", "-", s).strip("-") or "unknown"


def make_canonical_id(law_no, article_no, legislation_type, law_name, law_short=None):
    art = normalise_article_key(article_no)
    if law_no:
        return f"{re.sub(r'[^0-9]', '', str(law_no)) or law_no}/{art}"
    if legislation_type == "constitution":
        return f"constitution/{art}"
    if law_name:
        return f"{re.sub(r'[^a-zçğıöşü0-9]+', '-', lib.tr_lower(law_name)).strip('-')}/{art}"
    if law_short:
        return f"{lib.tr_lower(law_short)}/{art}"
    return None


def normalise_law_short(s):
    """'T.B.K.' -> 'TBK'; None when not shaped like an abbreviation."""
    if not s:
        return None
    s = lib.tr_upper(re.sub(r"[.\s ]+", "", str(s)))
    return s if LAW_SHORT_RE.match(s) else None


def fold_opinion(kind, authors):
    if kind not in ("dissent", "concurring"):
        return kind
    names = [slug(a) for a in authors or [] if slug(a)]
    return f"{kind}:{'+'.join(names)}" if names else kind


def is_separate(opinion_type):
    return (opinion_type or "").split(":")[0] in ("dissent", "concurring")


def fill_uncovered(segments, paras, source, log):
    """Every paragraph the model left out goes into a catch-all segment, so the
    stored chunks always cover the whole document. Consecutive gaps form one
    segment. Logged as `uncovered_filled`, never rejected."""
    from types import SimpleNamespace
    n = len(paras)
    covered = {i for seg in segments for i in _ref_idxs(seg.paragraph_refs, n)}
    gaps, run_ = [], []
    for i in range(1, n + 1):
        if i in covered:
            if run_:
                gaps.append(run_)
                run_ = []
        else:
            run_.append(i)
    if run_:
        gaps.append(run_)
    out = list(segments)
    if gaps:
        role = CATCHALL_ROLE.get(source, "other")
        out += [SimpleNamespace(local_id=f"fill_{k}", paragraph_refs=[f"p{i}" for i in g],
                                role=role, confidence="low", rights=None)
                for k, g in enumerate(gaps, 1)]
        spans = ",".join(f"p{g[0]}" + (f"-p{g[-1]}" if len(g) > 1 else "") for g in gaps)
        log.append(f"uncovered_filled:{sum(len(g) for g in gaps)} paragraph(s) in {len(gaps)} "
                   f"catch-all segment(s): {spans}")
    out.sort(key=lambda sg: _first_ref(sg, n))          # chunks are stored in document order
    for sg in out:
        idxs = _ref_idxs(sg.paragraph_refs, n)
        if idxs and idxs[-1] - idxs[0] + 1 != len(idxs):
            log.append(f"noncontiguous_segment:{sg.local_id}:p{idxs[0]}-p{idxs[-1]} holds {len(idxs)}")
    return out


def fallback_segments(paras, source):
    """The model gave no usable answer: consecutive paragraphs grouped up to the
    size cap, catch-all role, low confidence. The record is stored, flagged
    `fallback_no_model`, and can be rerun by id."""
    from types import SimpleNamespace
    groups, cur, size = [], [], 0
    for i, p in enumerate(paras, 1):
        if cur and size + 1 + len(p) > lib.MAX_CHUNK_CHARS:
            groups.append(cur)
            cur, size = [], 0
        cur.append(i)
        size += len(p) + (1 if size else 0)
    if cur:
        groups.append(cur)
    role = CATCHALL_ROLE.get(source, "other")
    return [SimpleNamespace(local_id=f"fallback_{k}", paragraph_refs=[f"p{i}" for i in g], role=role,
                            confidence="low", rights=None)
            for k, g in enumerate(groups, 1)]


_QUOTES = str.maketrans({"’": "'", "‘": "'", "ʼ": "'", "`": "'", "´": "'", "′": "'",
                         "“": '"', "”": '"', "„": '"', "«": '"', "»": '"'})


def ground_form(s):
    """Text as compared for citation grounding: one apostrophe and one quote
    form, Turkish upper case, single spaces."""
    return lib.tr_upper(norm_ws(s).translate(_QUOTES))


def _number_in(num, text):
    return bool(num) and re.search(rf"(?<!\d){re.escape(str(num))}(?!\d)", text) is not None


def _blank(v):
    """'' and whitespace are null (null policy: an empty field is null, never '')."""
    v = norm_ws(v) if isinstance(v, str) else v
    return v or None


def _article_in(article_no, text):
    """The article is stated in `text`: as written ("353/1-b-1") or its leading number."""
    if not article_no:
        return False
    lead = re.match(r"\d+", article_no)
    return _number_in(article_no, text) or bool(lead and _number_in(lead.group(0), text))


def clean_citation(c, para, doc_up, counts):
    """One LAWS-answer citation, checked against ITS OWN paragraph -> the stored dict,
    or None. Every stored field is what the decision states, or null."""
    if c.legislation_type == "not_legislation":
        counts["dropped_non_legislation"] += 1
        return None
    if c.legislation_type == "treaty":
        counts["dropped_treaty"] += 1                  # no stored home yet (README)
        return None
    law_no, law_short, law_name = _blank(c.law_no), _blank(c.law_short), _blank(c.law_name)
    article_no, paragraph_no = _blank(c.article_no), _blank(c.paragraph_no)
    law_date, quote = _blank(c.law_date), _blank(c.verbatim_mention)
    para_up = ground_form(para)
    law_digits = re.sub(r"\D", "", law_no or "")
    quote_exact = bool(quote) and ground_form(quote) in para_up
    # Grounded in the paragraph the answer put it on: the exact quote, the article
    # number, or "<law number> sayılı". Otherwise the citation is not stored.
    if not (quote_exact or _article_in(article_no, para)
            or (law_digits and re.search(rf"(?<!\d){law_digits}\s*SAYILI", para_up))):
        counts["citation_not_in_its_paragraph"] += 1
        return None
    if quote and not quote_exact:
        quote = None                                   # never store a quote the text does not contain
        counts["verbatim_not_exact_cleared"] += 1
    # Docs 13.1: law_no only as the court states it. The Constitution has none (a
    # stray number is the article); a number the decision never writes is not stored.
    if c.legislation_type == "constitution" and law_no:
        article_no = article_no or law_no
        law_no = None
        counts["constitution_law_no_cleared"] += 1
    # A law number written in the citation's own quote ("3572 sayılı ... Kararname")
    # is stated, not inferred: filled when the answer left law_no empty.
    if not law_no and quote_exact and c.legislation_type != "constitution":
        in_quote = set(re.findall(r"(?<![\d/])(\d{3,5})\s+sayılı", quote, re.IGNORECASE))
        if len(in_quote) == 1:
            law_no = law_digits = in_quote.pop()
            counts["law_no_from_quote"] += 1
    if law_no and not _number_in(law_digits or law_no, doc_up):
        law_no = None
        counts["law_no_not_in_document_cleared"] += 1
    # law_short: an abbreviation written in capitals in this paragraph (HMK, T.B.K.).
    # "Kanun" / "KANUN" is a back-reference word, not an abbreviation.
    if law_short and (LOWER_RE.search(law_short)
                      or re.sub(r"[.\s]", "", law_short) not in re.sub(r"[.\s]", "", para)):
        law_short = None
        counts["law_short_rejected"] += 1
    law_short = normalise_law_short(law_short)
    # law_name: a name the decision writes. One word ("Kanun", "Anayasa") is not a
    # name; an abbreviation the model expanded (T.B.K. -> Türk Borçlar Kanunu) is not stated.
    if law_name and (len(law_name.split()) < 2 or ground_form(law_name) not in doc_up):
        law_name = None
        counts["law_name_cleared"] += 1
    if c.legislation_type == "constitution" and not article_no:
        counts["constitution_without_article_dropped"] += 1   # also catches "Anayasa Mahkemesi"
        return None
    if not (law_no or law_short or law_name or article_no or c.legislation_type == "constitution"):
        counts["citation_without_law_or_article_dropped"] += 1  # "Bu Kanun"
        return None
    return {"canonical_id": make_canonical_id(law_no, article_no, c.legislation_type, law_name, law_short),
            "legislation_type": c.legislation_type, "law_no": law_no, "law_short": law_short,
            "law_name": law_name, "article_no": article_no, "paragraph_no": paragraph_no,
            "verbatim_mention": quote, "law_date": law_date, "confidence": "high" if article_no else "low"}


def build_chunks(segments, alias, source, doc_id, case_no, paras, decision_date, log, cites_by_para=None):
    """A chunk's citations are the LAWS answer's citations of its own paragraphs."""
    role_field = SOURCES[source]["role_field"]
    n, doc_up = len(paras), ground_form(" ".join(paras))
    cites_by_para = cites_by_para or {}
    chunks, id_map, counts = [], {}, Counter()
    for seg in segments:
        idxs = _ref_idxs(seg.paragraph_refs, n)
        refs = [f"p{i}" for i in idxs]
        rng = refs[0] if len(refs) == 1 else f"{refs[0]}_to_{refs[-1]}"
        text = "\n".join(paras[i - 1] for i in idxs)
        content_type, stage = derive(seg.role)
        # The citations of this segment's paragraphs, each checked against its own paragraph.
        cites, seen = [], set()
        for i in idxs:
            for c in cites_by_para.get(i, []):
                key = (i, c.law_no, c.law_short, c.article_no, c.paragraph_no, norm_ws(c.verbatim_mention))
                if key in seen:
                    continue
                seen.add(key)
                cleaned = clean_citation(c, paras[i - 1], doc_up, counts)
                if cleaned:
                    cites.append((i, cleaned))
        pieces = lib.split_by_size_cap(text)
        # Each citation lands in exactly ONE piece: the piece of its own paragraph that
        # holds its quote, else its article number, else where that paragraph starts.
        # (Matching the law number first put a quote from piece 2 on piece 1.)
        gpieces = [ground_form(pc) for pc in pieces]
        ends, pos = [], 0
        for g in gpieces:
            ends.append(pos + len(g))
            pos += len(g) + 1
        starts, pos = {}, 0
        for i in idxs:
            starts[i] = pos
            pos += len(ground_form(paras[i - 1])) + 1

        def piece_at(offset):
            return next((k_ for k_, end in enumerate(ends) if offset <= end), len(pieces) - 1)

        joined = " ".join(gpieces)

        def piece_of(i, cite):
            first = piece_at(starts[i])
            span = range(first, piece_at(starts[i] + len(ground_form(paras[i - 1]))) + 1)
            quote = ground_form(cite["verbatim_mention"] or "")
            whole = next((k_ for k_ in span if quote and quote in gpieces[k_]), None)
            if whole is not None:
                return whole
            # The 2,000-character split cut the quote in two (6 of 271 citations on
            # the 30-document run): the piece where the quote starts.
            at = joined.find(quote, max(0, starts[i] - 50)) if quote else -1
            if at >= 0:
                return piece_at(at)
            return next((k_ for k_ in span if _article_in(cite["article_no"], pieces[k_])), first)

        assign = {}
        for i, c in cites:
            assign.setdefault(piece_of(i, c), []).append(c)
        for j, piece in enumerate(pieces, 1):
            piece_rng = rng if len(pieces) == 1 else f"{rng}_p{j}"
            # One entry per provision per chunk: the same article named in two
            # paragraphs of one segment comes back twice from the LAWS answer.
            piece_cites, provisions = [], set()
            for c in assign.get(j - 1, []):
                prov = (c["canonical_id"] or c["law_short"] or c["law_name"] or ground_form(c["verbatim_mention"] or ""),
                        c["article_no"], c["paragraph_no"])
                if prov in provisions:
                    counts["same_provision_merged"] += 1
                    continue
                provisions.add(prov)
                piece_cites.append(c)
            chunk_id = make_chunk_id(doc_id, piece_rng)
            id_map.setdefault(seg.local_id, []).append(chunk_id)
            chunk = {
                "chunk_id": chunk_id, "chunk_label": f"{doc_id}-{piece_rng}",
                "source_type": source, "case_no": case_no, "decision_date": decision_date,
                "citation_granularity": SOURCES[source]["granularity"],
                "source_paragraph_ids": refs, "text": piece, "char_length": len(piece),
                "content_type": content_type,
                "firac_role": seg.role if source == "aym" else None,
                "reasoning_stage": stage,
                "rights": (getattr(seg, "rights", None) or None) if source == "aym" else None,
                "confidence": seg.confidence,
            }
            if role_field != "firac_role":
                chunk[role_field] = seg.role
            chunk["role"] = seg.role
            chunk["role_vocabulary"] = ROLE_VOCABULARY[role_field]
            chunk["schema_version"] = CHUNK_SCHEMA_VERSION
            chunk["cited_legislations"] = piece_cites
            chunks.append(chunk)
    for old in alias:                                  # merged/emptied segments
        new, hops = alias[old], {old}
        while new in alias and new not in id_map and new not in hops:   # A -> B -> C chains
            hops.add(new)
            new = alias[new]
        id_map.setdefault(old, []).extend(i for i in id_map.get(new, []) if i not in id_map[old])
    log += [f"{k}:{v}" for k, v in counts.items()]
    return chunks, id_map


def build_capsules(capsules, chunks, n, source, case_no, decision_date, subject_type, log):
    """Capsules point at paragraphs; a capsule rests on every chunk holding one of them."""
    out = []
    by_para = {}
    for c in chunks:
        for i in _ref_idxs(c["source_paragraph_ids"], n):
            by_para.setdefault(i, []).append(c["chunk_id"])
    role_of_chunk = {c["chunk_id"]: c["role"] for c in chunks}
    # The same outcome with the same conclusion sentence is one capsule written twice
    # (a 561-paragraph norm review repeated two; a joint dissent came back again
    # under one of its authors): kept once, support and authors merged.
    merged = {}
    for cap in capsules:
        key = (cap.outcome, norm_ws(cap.conclusion_sentence))
        if key in merged:
            first = merged[key]
            merged[key] = first.model_copy(update={
                "supporting_paragraph_refs": list(dict.fromkeys(list(first.supporting_paragraph_refs)
                                                                + list(cap.supporting_paragraph_refs))),
                "dissent_authors": list(dict.fromkeys(list(first.dissent_authors) + list(cap.dissent_authors)))})
            log.append(f"duplicate_capsule_merged:{cap.outcome}:{norm_ws(cap.conclusion_sentence)[:40]}")
        else:
            merged[key] = cap
    for cap in merged.values():
        idxs = _ref_idxs(cap.supporting_paragraph_refs, n)
        bad = [r for r in cap.supporting_paragraph_refs if not _ref_idxs([r], n)]
        if bad:
            log.append(f"dangling_support_dropped:{','.join(map(str, bad[:5]))}")   # a paragraph that does not exist
        support = list(dict.fromkeys(cid for i in idxs for cid in by_para.get(i, [])))
        # Docs 15.4: code cross-checks opinion_type against the roles of the
        # supporting chunks. A majority capsule never rests on dissent chunks; a
        # dissent capsule rests on dissent chunks when it has any. Repaired
        # mechanically and logged, instead of stored inconsistent.
        sep = cap.opinion_type in ("dissent", "concurring")
        if role_of_chunk and support:
            dis = [i for i in support if role_of_chunk.get(i) in DISSENT_ROLES]
            if not sep and dis:
                log.append(f"support_repaired:majority_dropped_{len(dis)}_dissent_chunks")
                support = [i for i in support if i not in dis]
            elif sep and dis and len(dis) < len(support):
                log.append(f"support_repaired:{cap.opinion_type}_kept_{len(dis)}_dissent_chunks")
                support = dis
        out.append({
            "case_no": case_no, "source_type": source, "decision_date": decision_date,
            "subject_type": subject_type, "subject_id": slug(cap.subject_id) or "unspecified",
            "opinion_type": fold_opinion(cap.opinion_type, cap.dissent_authors),
            "outcome": cap.outcome, "conclusion_sentence": cap.conclusion_sentence,
            "reasoning_summary": cap.reasoning_summary,
            "reasoning_summary_method": REASONING_SUMMARY_METHOD,
            "supporting_chunk_ids": support,
        })
    return out



# 7. CHECK -- mechanics only: is every paragraph stored once, is nothing invented
# or empty, do roles and capsules agree structurally.


ARTICLE_NUM_RE = re.compile(r"\b(\d+)\s*(?:\.|['’ʼ]?\s*[IU]NC[IU]|['’ʼ]?\s*NC[IU])?\s*(?:/\s*[A-Z]\s*)?MADDE|\bMADDE\s+(\d+)")
LAW_NUM_RE = re.compile(r"(?<![\d/])\b(\d{2,5})\s+SAYILI\s+(?:[^\s.;,]+\s+){0,12}?(?:KANUN|YASA|KHK|KARARNAME)")
LAW_NAMED_RE = re.compile(r"(?<![\d/])\b(\d{3,5})\s+sayılı\s+(?:[^\s.;,]+\s+){0,12}?(?:Kanun|Yasa|KHK|Kararname)",
                          re.IGNORECASE)
CASE_NUM_RE = re.compile(r"\b(\d{4}/\d+)\b")
_ORDINALS = {"BIRINCI": "1", "IKINCI": "2", "UCUNCU": "3", "DORDUNCU": "4", "BESINCI": "5",
             "ALTINCI": "6", "YEDINCI": "7", "SEKIZINCI": "8", "DOKUZUNCU": "9", "ONUNCU": "10"}
_DEACCENT = str.maketrans("çğıöşüÇĞİÖŞÜâîûÂÎÛ", "cgiosuCGIOSUaiuAIU")


def norm_upper(s):
    return lib.tr_upper(s or "").translate(_DEACCENT)


def stated_numbers(text):
    """{'m33','k6698','e2019/12951'}: articles, law numbers, case numbers."""
    up = re.sub(r"\b(" + "|".join(_ORDINALS) + r")\s+(MADDE|FIKRA)",
                lambda m: _ORDINALS[m.group(1)] + ". " + m.group(2), norm_upper(text))
    out = {"m" + (a or b) for a, b in ARTICLE_NUM_RE.findall(up)}
    out |= {"k" + x for x in LAW_NUM_RE.findall(up)}
    out |= {"e" + x for x in CASE_NUM_RE.findall(up)}
    return out


def doc_of(chunk):
    return chunk["chunk_label"].split("-p", 1)[0]


def verify_document(chunks, capsules, paras):
    """Returns (issues, soft): lists of 'kind:detail' strings. Nothing is rejected; issues are for review."""
    hard, soft, n = [], [], len(paras)
    count, seen_ranges = [0] * (n + 1), set()
    for c in chunks:
        key = tuple(c["source_paragraph_ids"])
        if key in seen_ranges:                      # size-cap pieces share refs
            continue
        seen_ranges.add(key)
        for i in _ref_idxs(c["source_paragraph_ids"], n):
            count[i] += 1
    dup = [i for i in range(1, n + 1) if count[i] > 1]
    if dup:
        hard.append(f"paragraph_overlap:{','.join('p%d' % i for i in dup[:8])}")
    missing = [i for i in range(1, n + 1) if count[i] == 0]
    content = [i for i in missing if len(paras[i - 1]) >= FRAGMENT_CHARS and not is_heading_line(paras[i - 1])]
    minor = [i for i in missing if i not in content]
    if content:
        chars = sum(len(paras[i - 1]) for i in content)
        (hard if chars > 200 or any(len(paras[i - 1]) >= 100 for i in content) else soft).append(
            f"uncovered_content:{len(content)} paragraph(s), {chars} chars: "
            f"{','.join('p%d' % i for i in content[:10])}")
    if minor:
        soft.append(f"uncovered_minor:{','.join('p%d' % i for i in minor[:10])}")
    for c in chunks:
        if c["char_length"] < FRAGMENT_CHARS and not is_heading_line(c["text"]):
            soft.append(f"fragment_chunk:{c['chunk_label']}:{c['text'][:30]!r}")
    # citation recall, mechanical: a law NUMBER named in the text ("6698 sayılı")
    # that no citation carries. Reported, so an empty cited_legislations on a
    # document that names laws cannot pass unnoticed.
    # A law is "<number> sayılı ... Kanun/KHK/Kararname". Not preceded by "YYYY/"
    # (a decision number: "2020/935 sayılı Karar") and not "sayılı Resmî Gazete".
    named = set(LAW_NAMED_RE.findall(" ".join(paras)))
    cited = {str(x["law_no"]).strip() for c in chunks for x in c["cited_legislations"] if x["law_no"]}
    if named - cited:
        soft.append(f"laws_named_not_cited:{','.join(sorted(named - cited))}")
    role_of = {c["chunk_id"]: c["role"] for c in chunks}
    text_of = {c["chunk_id"]: c["text"] for c in chunks}
    doc_nums = stated_numbers(" ".join(paras))
    doc_text = " ".join(paras)
    doc_lang = language_of(" ".join(paras))
    if chunks and capsules and all(is_separate(c["opinion_type"]) for c in capsules):
        hard.append("no_majority_capsule")
    if chunks and not any(c["role"] in RULING_ROLES for c in chunks):
        # Structural: a decision has an operative ruling somewhere. Reported,
        # never used to drop the document.
        hard.append("no_ruling_chunk")
    dissent_chunks = sum(1 for c in chunks if c["role"] in DISSENT_ROLES)
    if dissent_chunks and not any(is_separate(c["opinion_type"]) for c in capsules):
        # Structural: a dissent segment is a separate opinion, and every separate
        # opinion gets a capsule. One without the other is a wrong role or a lost capsule.
        hard.append(f"dissent_chunk_without_opinion_capsule:{dissent_chunks}")
    if chunks:
        # A ruling chunk followed by content and then by the real ruling is usually
        # the case history quoting an earlier decision ("Karar sonucu: Daire ... bozmuş").
        catchall = CATCHALL_ROLE.get(chunks[0]["source_type"], "other")
        seq = [(c["role"], c["chunk_label"]) for c in sorted(chunks, key=lambda c: int(c["source_paragraph_ids"][0][1:]))]
        last = max((k for k, (r, _) in enumerate(seq) if r in RULING_ROLES), default=None)
        for k, (r, label) in enumerate(seq[:last] if last is not None else []):
            if r in RULING_ROLES and any(r2 not in RULING_ROLES and r2 not in DISSENT_ROLES and r2 != catchall
                                         for r2, _ in seq[k + 1:last]):
                soft.append(f"ruling_chunk_before_content:{label}")
    for c in capsules:
        ids = c["supporting_chunk_ids"]
        roles = {role_of.get(i) for i in ids}
        if not ids:
            hard.append(f"capsule_without_support:{c['opinion_type']}")
        if is_separate(c["opinion_type"]) and not roles & DISSENT_ROLES:
            hard.append(f"separate_opinion_unsupported:{c['opinion_type']}")
        if not is_separate(c["opinion_type"]) and roles & DISSENT_ROLES:
            hard.append(f"majority_on_dissent:{c['outcome']}")
        for field in ("reasoning_summary", "conclusion_sentence"):
            if not (c[field] or "").strip():
                hard.append(f"{field}_empty:{c['opinion_type']}")
            elif doc_lang and language_of(c[field]) != doc_lang:
                hard.append(f"{field}_not_in_decision_language:{language_of(c[field])}!={doc_lang}:"
                            f"{c['opinion_type']}")
        sup_nums = stated_numbers(" ".join(text_of.get(i, "") for i in ids))
        for tok in sorted(stated_numbers(c["reasoning_summary"] + " " + c["conclusion_sentence"])):
            # Article numbers also count when the decision writes them in a list
            # ("334 ve devamı maddelerinde"), which the pattern does not read as an article.
            if tok not in doc_nums and not (tok[0] == "m" and _number_in(tok[1:], doc_text)):
                (hard if tok[0] in "ke" else soft).append(f"number_not_in_document:{tok}")
            elif tok not in sup_nums:
                soft.append(f"number_not_in_support:{tok}")
        if c["outcome"] == "other":
            soft.append(f"outcome_other:{c['opinion_type']}")
        if len(c["conclusion_sentence"] or "") > 400:
            soft.append(f"conclusion_sentence_long:{len(c['conclusion_sentence'])} chars:{c['opinion_type']}")
        if c["opinion_type"].startswith("dissent"):
            # Compared with the majority on the SAME subject: in a multi-provision
            # decision a dissent may rightly share an outcome with another provision.
            # A dissent's subject is often the majority's plus a suffix
            # ("..._m3_1_e" -> "..._m3_1_e_karsi_oy"): the longest shared subject wins.
            majority = [x for x in capsules if not is_separate(x["opinion_type"])]
            sid = c["subject_id"]
            related = [x for x in majority if x["subject_id"] == sid or sid.startswith(x["subject_id"] + "_")
                       or x["subject_id"].startswith(sid + "_")]
            longest = max((len(x["subject_id"]) for x in related), default=0)
            same = {x["outcome"] for x in related if len(x["subject_id"]) == longest}
            if not related and len({x["outcome"] for x in majority}) == 1:
                same = {majority[0]["outcome"]}
            if c["outcome"] in same:
                soft.append(f"dissent_same_outcome_as_majority:{c['outcome']}:{c['subject_id']}")
        if c["subject_id"] == "unspecified":
            hard.append("subject_id_unspecified")
    return hard, soft



# Driver



def extraction_issues(lost):
    if not lost:
        return []
    top = sorted(lost.items(), key=lambda x: -x[1])[:8]
    return [f"extraction_word_loss:{sum(lost.values())} word(s): " + ", ".join(f"{w}x{c}" for w, c in top)]


def assemble(paras, source, doc_id, case_no, decision_date, subject_type, segments, alias,
             capsules, cites_by_para, log):
    """Everything after the model calls, no API: fill uncovered paragraphs, attach
    citations by paragraph, assemble chunks and capsules, check.
    assemble_document, test_chunk --stress, --replay and --rebuild all run this."""
    segments = fill_uncovered(segments, paras, source, log)
    chunks, _ = build_chunks(segments, alias, source, doc_id, case_no, paras, decision_date, log, cites_by_para)
    caps = build_capsules(list(capsules), chunks, len(paras), source, case_no, decision_date, subject_type, log)
    issues, soft = verify_document(chunks, caps, paras)
    return chunks, caps, issues, soft, log


def mechanical_document(paras, source, doc_id, case_no, decision_date, reason, lost, cites_by_para=None):
    """No structure answer, still every paragraph stored -- with the LAWS answers'
    citations when those calls succeeded. No capsule."""
    log = [f"fallback_no_model:{reason}"]
    segments = fallback_segments(paras, source)
    chunks, _ = build_chunks(segments, {}, source, doc_id, case_no, paras, decision_date, log, cites_by_para)
    issues, soft = verify_document(chunks, [], paras)
    return chunks, [], [f"fallback_no_model:{reason}"] + extraction_issues(lost) + issues, soft, log


# The three steps. prepare and assemble_document are pure code; only call_request
# talks to the API, so batch mode replaces call_request and nothing else.


def prepare(record, source):
    """Record -> job: paragraphs, code-owned fields and the requests (structure,
    capsules, laws_1..laws_k). A request is plain data (name, system text, content
    text, response model, limit note, window): sent by call_request now, written
    into a batch file later. A job without requests is stored mechanically."""
    doc_id = str(record.get("doc_id"))
    paras, lost, plain_ct = paragraphs(record)
    job = {"doc_id": doc_id, "source": source, "case_no": compute_case_no(record, source),
           "paras": paras, "lost": lost, "requests": {}}
    if not paras:
        return job
    kind = kind_of(record, source)
    cands = candidate_rights(record) if kind == "aym_individual_application" else []
    vocab = {"outcomes": OUTCOME_BY_KIND[kind], "gloss": OUTCOME_GLOSS,
             "opinion_kinds": OPINION_KINDS if source in ("kvkk", "rekabet") else OPINION_KINDS[:3],
             "legislation_types": RESPONSE_LEGISLATION_TYPES}
    # A worked-example document gets no worked example: it would be shown its own answer.
    with_example = not prompts.is_example_document(source, job["case_no"], doc_id)
    chars = sum(len(p) for p in paras)
    job.update(kind=kind, decision_date=compute_decision_date(record, paras),
               subject_type=resolve_subject_type(record, source, kind),
               predicted=int(chars * 0.35),
               base_review={"doc_id": doc_id, "case_no": job["case_no"], "kind": kind,
                            "words_lost": sum(lost.values()), "paragraphs": len(paras),
                            "worked_example": with_example})
    if chars > MAX_DOCUMENT_CHARS:
        job["skip_reason"] = "document_too_long"
        return job
    # Both views (owner's decision): numbered paragraphs always; the plain
    # content_text as a reference copy only when it holds words the paragraphs lack.
    plain = plain_ct if plain_ct and lost else None
    content = prompts.build_user_content(paras, plain_copy=plain)
    requests = {
        "structure": {"name": "structure", "content": content, "predicted": job["predicted"],
                      "system": prompts.build_structure_instruction(source, kind, job["case_no"], cands, vocab,
                                                                    with_example=with_example),
                      "schema": structure_model_for(source, kind, cands),
                      "limit_note": prompts.STRUCTURE_LIMIT_NOTE},
        "capsules": {"name": "capsules", "content": content, "predicted": job["predicted"],
                     "system": prompts.build_capsules_instruction(
                         source, kind, job["case_no"], cands,
                         examined_norms(record) if kind == "aym_norm_review" else [], vocab,
                         with_example=with_example,
                         metadata_hint=aym_hint(record, kind) if source == "aym" else None),
                     "schema": capsules_model_for(source, kind, cands),
                     "limit_note": prompts.CAPSULES_LIMIT_NOTE},
    }
    laws_system = prompts.build_laws_instruction(source)
    for k, (lo, hi) in enumerate(laws_windows(paras), 1):
        requests[f"laws_{k}"] = {"name": "laws", "window": [lo, hi], "system": laws_system,
                                 "content": prompts.build_laws_content(paras, lo, hi, plain_copy=plain),
                                 "predicted": int(sum(len(p) for p in paras[lo - 1:hi]) * 0.35),
                                 "schema": LawsResponse, "limit_note": prompts.LAWS_LIMIT_NOTE}
    job["requests"] = requests
    return job


def call_request(client, model, req, predicted, stats):
    """One request, up to three attempts: as prepared; perturbed (an identical
    deterministic retry repeats the failure); and, only after a truncated answer,
    with the request's limit note (a list that looped until the output budget ran
    out). Quota and server errors are retried inside the SDK first (http_options).
    Returns (parsed answer or None, raw text, error, notes)."""
    raw, err, finish, notes = None, None, None, []
    predicted = req.get("predicted", predicted)
    for attempt in (1, 2, 3):
        if attempt == 3 and finish != "MAX_TOKENS":
            break
        system = req["system"] + (f"\n\n## LIMIT\n{req['limit_note']}" if attempt == 3 else "")
        finish = None
        try:
            resp = call_gemini(client, model, system, req["content"], req["schema"], perturb=attempt == 2,
                               max_tokens=min(MAX_OUTPUT_TOKENS, max(8192, predicted * 3)) if attempt >= 2 else None)
            raw, finish = resp.text, _finish(resp)
            _usage(resp, stats, req["name"])
            if finish == "MAX_TOKENS":
                raise ValueError("response truncated at max_output_tokens")
            return req["schema"].model_validate_json(raw), raw, None, notes
        except Exception as e:                           # noqa: BLE001
            err = f"{type(e).__name__}: {str(e)[:300]}"
            label = req["name"] + (f"[p{req['window'][0]}-p{req['window'][1]}]" if req.get("window") else "")
            notes.append(f"retry:{label}:attempt_{attempt}_failed:{err[:80]}")
    return None, raw, ("truncated: " if finish == "MAX_TOKENS" else "") + (err or "no answer"), notes


def assemble_document(job, answers, raws, errors, notes=()):
    """Job + answers -> (chunks, capsules, review), no API. `answers` maps request
    key to the parsed answer, None when that call failed. Nothing is rejected:
      no structure answer    every paragraph stored in catch-all chunks, flagged fallback
      no capsules answer     chunks without capsules; issue capsules_call_failed
      a laws window failed   that window's paragraphs without citations; issue laws_call_failed"""
    paras, source, doc_id = job["paras"], job["source"], job["doc_id"]
    if not paras:
        return None, None, {"doc_id": doc_id, "case_no": job["case_no"], "reason": "no_text",
                            "error": "the record has no decision text; nothing to chunk"}
    n, log = len(paras), list(notes)
    structure, capsules = answers.get("structure"), answers.get("capsules")
    laws_keys = [k for k in job["requests"] if k.startswith("laws")]
    cites = {}
    for k in laws_keys:
        window = tuple(job["requests"][k]["window"])
        for i, found in citations_by_paragraph(answers.get(k), n, log, window).items():
            cites.setdefault(i, []).extend(found)

    def as_json(k):
        return json.loads(raws[k]) if answers.get(k) is not None and raws.get(k) else None
    review = dict(job["base_review"], raw_response=as_json("structure"), raw_capsules=as_json("capsules"),
                  raw_laws=[{"window": job["requests"][k]["window"], "answer": as_json(k)} for k in laws_keys])
    failed_raw = {k: (raws.get(k) or "")[:4000] for k in job["requests"] if answers.get(k) is None and raws.get(k)}
    if failed_raw:
        review["raw_failed"] = failed_raw
    head = []
    if structure is None:
        reason = job.get("skip_reason") or "api_or_parse_failed"
        segments, alias = fallback_segments(paras, source), {}
        log.append(f"fallback_no_model:{reason}")
        head.append(f"fallback_no_model:{reason}")
        review.update(fallback=reason, error=errors.get("structure"))
    else:
        segments, alias, nlog = normalise_segments(structure.segments, paras)
        log += nlog
    chunks, caps, issues, soft, log = assemble(paras, source, doc_id, job["case_no"], job["decision_date"],
                                               job["subject_type"], segments, alias,
                                               capsules.capsules if capsules is not None else [], cites, log)
    if job["requests"] and capsules is None:
        head.append(f"capsules_call_failed:{(errors.get('capsules') or '')[:80]}")
    for k in laws_keys:
        if answers.get(k) is None:
            lo, hi = job["requests"][k]["window"]
            head.append(f"laws_call_failed:p{lo}-p{hi}:{(errors.get(k) or '')[:60]}")
    review.update(issues=head + extraction_issues(job["lost"]) + issues, soft=soft, log=log)
    return chunks, caps, review


LANGUAGE_NAME = {"tr": "Turkish", "en": "English"}
RETRY_SOFT = ("dissent_same_outcome_as_majority",)


def problem_count(review):
    """What a corrective retry must reduce: hard issues plus the soft notes it targets."""
    return len(review.get("issues") or []) + sum(1 for x in review.get("soft") or [] if x.startswith(RETRY_SOFT))


def corrections_for(review, chunks):
    """{request key: correction note} for model slips code can see in an assembled
    document. Each named request is sent ONCE more with the slip stated (F3); a call
    that failed outright was already retried by call_request."""
    if review.get("fallback"):
        return {}
    notes = {}

    def add(key, text):
        notes.setdefault(key, [])
        if text not in notes[key]:
            notes[key].append(text)
    for issue in review.get("issues") or []:
        if "_not_in_decision_language:" in issue:
            lang = issue.split("!=")[-1].split(":")[0]
            add("capsules", "conclusion_sentence and reasoning_summary must be written in the language of the "
                            f"decision ({LANGUAGE_NAME.get(lang, lang)}); some of yours were not.")
        elif issue == "no_ruling_chunk":
            add("structure", "No segment has the conclusion/outcome role, but every decision has an operative "
                             "ruling. Find it and give that segment the ruling role.")
        elif issue.startswith("dissent_chunk_without_opinion_capsule"):
            idxs = sorted({int(r[1:]) for c in chunks if c["role"] in DISSENT_ROLES for r in c["source_paragraph_ids"]})
            runs, spans = [], []
            for i in idxs:
                if runs and i == runs[-1][-1] + 1:
                    runs[-1].append(i)
                else:
                    runs.append([i])
            spans = ", ".join(f"p{r[0]}" + (f"-p{r[-1]}" if len(r) > 1 else "") for r in runs)
            add("capsules", f"Paragraphs {spans} are a separate opinion, but no dissent or concurring capsule was "
                            "written. Write one capsule for each separate opinion, citing only its own paragraphs "
                            "and naming its authors.")
        elif issue == "no_majority_capsule":
            add("capsules", "No majority (or board_decision) capsule was written; every decision has one for its "
                            "operative ruling.")
        elif issue.startswith(("capsule_without_support", "separate_opinion_unsupported", "majority_on_dissent")):
            add("capsules", "Some supporting_paragraph_refs do not fit their capsule: a majority capsule cites the "
                            "ruling and its reasoning, a separate opinion cites only its own paragraphs.")
        elif issue == "subject_id_unspecified":
            add("capsules", "subject_id must name the subject in dispute, never 'unspecified'.")
    for note in review.get("soft") or []:
        if note.startswith("dissent_same_outcome_as_majority"):
            add("capsules", "A dissent capsule has the same outcome as the majority on the same subject. A "
                            "dissent's outcome is what the DISSENTER would have decided: a dissenter who finds "
                            "the provision unconstitutional -> annulled, who would keep it -> denied, who would "
                            "find a violation -> violation. Check every dissent capsule's outcome against its own "
                            "conclusion_sentence.")
    return {k: " ".join(v) for k, v in notes.items()}


def process_document(client, model, record, source, stats):
    """(chunks, capsules, review) for one record: prepare, all requests in parallel,
    assemble; then, for model slips code can see, one corrective retry of the
    request concerned, kept only when it leaves fewer issues. chunks is None only
    for a record with no text."""
    from concurrent.futures import ThreadPoolExecutor
    job = prepare(record, source)
    answers, raws, errors, notes, usage = {}, {}, {}, [], {}
    if job["requests"]:
        usage = {key: Counter() for key in job["requests"]}
        with ThreadPoolExecutor(max_workers=min(PARALLEL_REQUESTS, len(job["requests"]))) as pool:
            futures = {key: pool.submit(call_request, client, model, req, job["predicted"], usage[key])
                       for key, req in job["requests"].items()}
        for key, fut in futures.items():
            answers[key], raws[key], errors[key], n = fut.result()
            notes += n
    chunks, caps, review = assemble_document(job, answers, raws, errors, notes)
    fixes = corrections_for(review, chunks) if chunks is not None else {}
    if fixes:
        answers2, raws2, errors2, notes2 = dict(answers), dict(raws), dict(errors), list(notes)
        for key, note in fixes.items():
            req = dict(job["requests"][key])
            req["system"] += ("\n\n## Correction\nYour previous answer for THIS document had this problem: " + note
                              + " Produce the whole answer again, fixing it and changing nothing else.")
            usage[f"{key}_correction"] = Counter()
            parsed, raw, err, n = call_request(client, model, req, job["predicted"], usage[f"{key}_correction"])
            notes2 += n
            if parsed is not None:
                answers2[key], raws2[key], errors2[key] = parsed, raw, None
        chunks2, caps2, review2 = assemble_document(job, answers2, raws2, errors2, notes2)
        improved = problem_count(review2) < problem_count(review)
        if improved:
            chunks, caps, review = chunks2, caps2, review2
        review["log"].append(f"corrective_retry:{'+'.join(fixes)}:{'improved' if improved else 'not_improved_first_kept'}")
    for u in usage.values():
        stats.update(u)
    if usage:
        review["tokens"] = {key: {"in": u["input_tokens"], "out": u["output_tokens"]} for key, u in usage.items()}
        review["cost_usd"] = round(cost_usd(sum(u["input_tokens"] for u in usage.values()),
                                            sum(u["output_tokens"] for u in usage.values())), 6)
    return chunks, caps, review


def write_output(out_root, source, chunks, caps, reviews, ran=None):
    """Write one source's output. `ran` (the doc_ids of a targeted run) merges:
    documents not in `ran` keep their chunks, capsules and reviews untouched;
    documents in `ran` are replaced. Without `ran` the file is this run's."""
    out_path = out_root / f"{source}.json"
    review_path = out_root / f"{source}_review.json"
    c, k, rv = chunks, caps, reviews
    if ran is not None and out_path.is_file():
        old = json.loads(out_path.read_text(encoding="utf-8"))
        kept_old = [x for x in old.get("chunks", []) if doc_of(x) not in ran]
        kept_ids = {x["chunk_id"] for x in kept_old}
        c = kept_old + chunks
        # keep only capsules of documents NOT in this run; this run's own
        # capsules are re-added from `caps` (re-appending them from the file
        # duplicated them on every write).
        k = [x for x in old.get("reasoning_capsules", [])
             if any(i in kept_ids for i in x["supporting_chunk_ids"])] + caps
    if ran is not None:
        rv = _merge(review_path, reviews, ran, lambda x: str(x.get("doc_id")))
    if c or not out_path.is_file():
        out_path.write_text(json.dumps({"chunks": c, "reasoning_capsules": k},
                                       ensure_ascii=False, indent=2), encoding="utf-8")
    review_path.write_text(json.dumps(rv, ensure_ascii=False, indent=2), encoding="utf-8")


def _merge(path, new_items, ran_ids, key):
    if not path.is_file():
        return new_items
    old = json.loads(path.read_text(encoding="utf-8"))
    return [x for x in old if key(x) not in ran_ids] + new_items


def run(sources, limit, doc_ids=None, out_dir=None):
    load_dotenv(ROOT / ".env")
    model = os.getenv("MODEL", "gemini-2.5-flash-lite")
    client = build_client()
    out_root = Path(out_dir) if out_dir else OUTPUT_DIR
    out_root = out_root if out_root.is_absolute() else ROOT / out_root
    out_root.mkdir(parents=True, exist_ok=True)
    print(f"model {model} | project {os.getenv('GOOGLE_CLOUD_PROJECT')} | out {out_root}")
    # Fail fast on schemas the API will not accept: one tiny call per kind.
    pairs = []
    for source in sources:
        path = DATA_DIR / f"{source}.json"
        if not path.is_file() or source in TEXT_PENDING_SOURCES:
            continue
        for r in json.loads(path.read_text(encoding="utf-8")):
            if doc_ids and str(r.get("doc_id")) not in doc_ids.get(source, []):
                continue
            if r.get("status") in USABLE_STATUSES:
                pairs.append((source, kind_of(r, source)))
    pre = Counter()
    problems = schema_preflight(client, model, pairs, pre)
    if problems:
        for pr in problems:
            print("  SCHEMA REJECTED BY API: " + pr)
        raise SystemExit("stopped before sending any document: fix the schema above")
    print(f"schema preflight: {len(set(pairs))} kind(s) accepted by the API | "
          f"cost ${cost_usd(pre['input_tokens'], pre['output_tokens']):.4f}\n")
    total, total_tokens = Counter(), Counter(pre)
    spent = cost_usd(pre["input_tokens"], pre["output_tokens"])
    for source in sources:
        path = DATA_DIR / f"{source}.json"
        if not path.is_file() or source in TEXT_PENDING_SOURCES:
            print(f"=== {source} === skipped (no data file or no extracted text upstream)")
            continue
        records = json.loads(path.read_text(encoding="utf-8"))
        usable = [r for r in records if r.get("status") in USABLE_STATUSES
                  and ((r.get("content_text") or "").strip() or (r.get("html_content") or "").strip())]
        if doc_ids:
            by_id = {str(r.get("doc_id")): r for r in records}
            picked = [by_id[d] for d in doc_ids.get(source, []) if d in by_id]
        else:
            # Worked-example documents are chunked like any other, without their example.
            picked = usable[:limit]
        ran = {str(r.get("doc_id")) for r in picked}
        chunks, caps, reviews = [], [], []
        stats = Counter()
        counts = Counter()

        def flush():
            """Write after EVERY document; a targeted run merges by doc_id."""
            write_output(out_root, source, chunks, caps, reviews, ran if doc_ids else None)

        from concurrent.futures import ThreadPoolExecutor, as_completed

        def work(rec):
            local = Counter()
            ch, cp, review = process_document(client, model, rec, source, local)
            return rec, ch, cp, review, local

        pool = ThreadPoolExecutor(max_workers=WORKERS)
        futures = [pool.submit(work, rec) for rec in picked]
        for fut in as_completed(futures):
            rec, ch, cp, review, local = fut.result()
            stats.update(local)
            doc = str(rec.get("doc_id"))
            if ch is None:
                reviews.append(review)
                counts["no_text"] += 1
                print(f"  [{source}] {doc}  NO TEXT: {review.get('error', '')[:100]}")
            else:
                chunks += ch
                caps += cp
                reviews.append(review)
                counts["ok"] += 1
                counts["with_issues"] += bool(review["issues"])
                counts["fallback"] += bool(review.get("fallback"))
                notes = review["issues"] + [x for x in review["log"] if x.startswith(
                    ("retry:", "laws_", "corrective_retry", "uncovered_filled", "dangling_support"))] + review["soft"]
                spent += review.get("cost_usd", 0.0)
                print(f"  [{source}] {doc}  {len(ch)} chunks, {len(cp)} capsules | "
                      f"${review.get('cost_usd', 0.0):.4f} (run so far ${spent:.4f})"
                      + (f"  | {'; '.join(n[:90] for n in notes[:3])}" if notes else ""))
            flush()
        pool.shutdown(wait=True)
        total.update(counts)
        total_tokens.update(stats)
        print(f"=== {source} === {len(picked)} docs | stored {counts['ok']} (with issues to review "
              f"{counts['with_issues']}; stored WITHOUT a model answer {counts['fallback']}) | "
              f"no text {counts['no_text']} | {len(chunks)} chunks | {len(caps)} capsules")
        retries = sum(1 for rv in reviews for x in rv.get("log", []) if x.startswith("retry:"))
        print(f"  output tokens: structure {stats['structure_output_tokens']:,} | capsules "
              f"{stats['capsules_output_tokens']:,} | laws "
              f"{stats['laws_output_tokens']:,} | retried attempts {retries} | "
              f"tokens in {stats['input_tokens']:,} out {stats['output_tokens']:,} | "
              f"cost ${cost_usd(stats['input_tokens'], stats['output_tokens']):.4f}\n")
    print(f"TOTAL stored {total['ok']} | with issues {total['with_issues']} | "
          f"without a model answer (rerun by id) {total['fallback']} | no text {total['no_text']}")
    grand = cost_usd(total_tokens["input_tokens"], total_tokens["output_tokens"])
    docs = max(total["ok"], 1)
    print(f"COST ${grand:.4f} for {total['ok']} document(s), ${grand / docs:.4f} per document "
          f"(schema check included) | tokens in {total_tokens['input_tokens']:,} out "
          f"{total_tokens['output_tokens']:,} | the same work as a batch job: ${grand / 2:.4f}")


def main():
    ap = argparse.ArgumentParser(description="raw court JSON -> chunks, LLM first")
    ap.add_argument("--source", choices=sorted(SOURCES))
    ap.add_argument("--limit", type=int, default=int(os.getenv("DOCS_PER_SOURCE", "2")))
    ap.add_argument("--doc-id-file", help="JSON {source: [doc_id, ...]}; merges into existing output")
    ap.add_argument("--out-dir", help="default output/chunk")
    args = ap.parse_args()
    doc_ids = None
    if args.doc_id_file:
        doc_ids = json.loads(Path(args.doc_id_file).read_text(encoding="utf-8"))
    sources = [args.source] if args.source else ([s for s in SOURCES if doc_ids.get(s)] if doc_ids
                                                  else [s for s in SOURCES if s not in TEXT_PENDING_SOURCES])
    run(sources, args.limit, doc_ids, args.out_dir)


if __name__ == "__main__":
    main()
