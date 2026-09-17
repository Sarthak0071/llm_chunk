"""
chunker.py -- raw court-decision JSON -> chunks[] + reasoning_capsules[].

One Gemini call reads the decision and returns segments (which paragraphs belong
together, their role, citations) and capsules (outcome, summary, dissents). A
second call re-reads the result and returns only fixes. Code does what the docs
(section 15.2) give to code: ids, dates, case numbers, text assembly, size cap,
derived fields, and the mechanical checks -- above all that EVERY paragraph of
the document is in exactly one chunk.

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
from pydantic import BaseModel, Field, create_model

import chunk_lib as lib
import prompts

ROOT = Path(__file__).resolve().parents[2]          # llm_chunk/
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "output" / "chunk"

# Fixed forever: regenerating it would change every chunk_id ever produced.
CHUNK_NAMESPACE = uuid.UUID("f08fd9b5-14a7-46ef-b7ac-a664a1b45032")
CHUNK_SCHEMA_VERSION = 2
REASONING_SUMMARY_METHOD = "llm_generated"

MAX_OUTPUT_TOKENS = 65535          # Gemini 2.5 Flash-Lite ceiling is exclusive at 65536
MIN_COVERAGE = 0.85                # paragraphs must carry this share of the record's text
FRAGMENT_CHARS = 40                # a shorter segment is merged into its neighbour
WORKERS = 4                        # documents processed in parallel (two calls each, in order)
USABLE_STATUSES = {"fetched", "completed"}
FEWSHOT_DOC_IDS = {("yargitay", "1221692000")}   # worked examples are never chunked
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

# AYM's own structured verdicts, used only as a HINT to the audit call.
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


def is_heading_line(text):
    """Short line with no lowercase letter: heading, docket label, name. No vocabulary."""
    t = (text or "").strip()
    return len(t) <= 80 and not re.search(r"[a-zçğıöşüâîû]", t)



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
# become chunks; coverage is measured after removal against the raw text.
JUNK_LINE_RE = re.compile(r"^(?:Karar İçeriği|ee3|0|\d{4}/1999999|[.:;\-–…]+)$")


def paragraphs(record):
    """(paragraphs, coverage, plain_content_text). content_text when it has
    real lines, else the HTML walk; junk lines dropped; coverage = share of the
    longer column's text that the paragraphs carry."""
    ct = record.get("content_text") or ""
    html = record.get("html_content") or ""
    lines = [norm_ws(l) for l in ct.splitlines() if norm_ws(l)]
    if len(lines) >= 5 or not html.strip():
        paras = lines
    else:
        paras = html_paragraphs(html)
    if len(paras) <= 1 and html.strip():
        alt = html_paragraphs(html)
        if len(alt) > len(paras):
            paras = alt
    paras = [q for q in paras if not JUNK_LINE_RE.match(q.strip())]
    full = max(len(plain_text(html)), len(norm_ws(ct)))
    got = len(norm_ws(" ".join(paras)))
    return paras, (round(got / full, 3) if full else 0.0), norm_ws(ct)



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
    """The KIND of document: prompts, outcome list and audit key on this."""
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
    """The court's own verdicts as text for the audit prompt (structured metadata)."""
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
    law_no: Optional[str] = None
    law_short: Optional[str] = None
    law_name: Optional[str] = None
    article_no: Optional[str] = None
    paragraph_no: Optional[str] = None
    law_date: Optional[str] = None
    legislation_type: Literal[RESPONSE_LEGISLATION_TYPES]
    confidence: Literal[CONFIDENCE]
    verbatim_mention: Optional[str] = None


class SegmentBase(BaseModel):
    local_id: str
    paragraph_refs: List[str]
    confidence: Literal[CONFIDENCE]
    # No maxItems here: Gemini rejects the schema ("too many states") when a
    # list bound meets the enum fields. Runaway citation loops are handled by
    # the third generation attempt instead (see process_document.generate).
    cited_legislations: List[CitedLegislation] = Field(default_factory=list)


class CapsuleBase(BaseModel):
    conclusion_sentence: str = Field(min_length=20)
    reasoning_summary: str = Field(min_length=80)
    dissent_authors: List[str] = Field(default_factory=list)
    supporting_local_ids: List[str] = Field(min_length=1)


_MODELS = {}


def response_model_for(source, kind, candidates=()):
    """Per-document model: roles of this source, outcomes of this kind, and for
    AYM individual applications `rights`/`subject_id` limited to the candidate
    rights the court's metadata names."""
    key = (source, kind, tuple(candidates))
    if key not in _MODELS:
        roles = tuple(prompts.ROLE_VOCAB[source][1])
        seg_extra, cap_extra = {}, {}
        if candidates:
            seg_extra["rights"] = (Optional[List[Literal[tuple(candidates)]]], None)
            cap_extra["subject_id"] = (Literal[tuple(candidates)], ...)
        else:
            cap_extra["subject_id"] = (str, Field(min_length=3))
        segment = create_model(f"Segment_{kind}", __base__=SegmentBase,
                               role=(Literal[roles], ...), **seg_extra)
        kinds = OPINION_KINDS if source in ("kvkk", "rekabet") else OPINION_KINDS[:3]
        capsule = create_model(f"Capsule_{kind}", __base__=CapsuleBase,
                               outcome=(Literal[tuple(OUTCOME_BY_KIND[kind])], ...),
                               opinion_type=(Literal[kinds], ...), **cap_extra)
        _MODELS[key] = create_model(f"Response_{kind}", segments=(List[segment], ...),
                                    capsules=(List[capsule], ...))
    return _MODELS[key]


def audit_model_for(source, kind, citations=True):
    roles = tuple(prompts.ROLE_VOCAB[source][1])
    outcomes = tuple(OUTCOME_BY_KIND[kind])
    RoleFix = create_model("RoleFix", local_id=(str, ...), role=(Literal[roles], ...),
                           reason=(str, Field(min_length=5)))
    OutcomeFix = create_model("OutcomeFix", capsule_index=(int, ...),
                              outcome=(Literal[outcomes], ...), reason=(str, Field(min_length=5)))
    Missing = create_model("MissingCapsule", outcome=(Literal[outcomes], ...), subject=(str, ...),
                           reason=(str, Field(min_length=5)))
    MissingCitation = create_model("MissingCitation", __base__=CitedLegislation, local_id=(str, ...))
    fields = dict(role_fixes=(List[RoleFix], Field(default_factory=list)),
                  outcome_fixes=(List[OutcomeFix], Field(default_factory=list)),
                  missing_capsules=(List[Missing], Field(default_factory=list)))
    if citations:
        fields["missing_citations"] = (List[MissingCitation], Field(default_factory=list))
    return create_model(f"Audit_{kind}{'' if citations else '_lite'}", **fields)



# Gemini



def build_client():
    from google import genai
    for var in ("GOOGLE_APPLICATION_CREDENTIALS", "GOOGLE_CLOUD_PROJECT"):
        if not os.getenv(var):
            raise SystemExit(f"FAILED: {var} not set. Check llm_chunk/.env")
    if not Path(os.getenv("GOOGLE_APPLICATION_CREDENTIALS")).is_file():
        raise SystemExit("FAILED: credentials file not found; fix the path in llm_chunk/.env")
    return genai.Client(vertexai=True, project=os.getenv("GOOGLE_CLOUD_PROJECT"),
                        location=os.getenv("GOOGLE_CLOUD_LOCATION", "global"))


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


def schema_preflight(client, model, pairs):
    """Before any document is sent: one ~20-token call per (source, kind) with the
    real response schema and the real audit schema. A schema the API rejects
    (400 "too many states", unsupported keyword) fails HERE with the API's own
    message, not on document 1 of 80. Cost: well under a cent."""
    from google.genai import types
    problems = []
    for source, kind in sorted(set(pairs)):
        cands = ("mülkiyet_hakkı",) if kind == "aym_individual_application" else ()
        for label, schema in (("response", response_model_for(source, kind, cands)),
                              ("audit", audit_model_for(source, kind))):
            try:
                client.models.generate_content(
                    model=model, contents="ping",
                    config=types.GenerateContentConfig(
                        temperature=0, max_output_tokens=16,
                        thinking_config=types.ThinkingConfig(thinking_budget=0),
                        response_mime_type="application/json", response_schema=schema,
                        system_instruction="Reply with any valid instance."))
            except Exception as exc:                            # noqa: BLE001
                msg = str(exc)
                if "400" in msg or "INVALID_ARGUMENT" in msg or "schema" in msg.lower():
                    problems.append(f"{source}/{kind} {label} schema rejected: {msg[:300]}")
                # other errors (quota, network) are not schema problems; the run reports them per document
    return problems


def _usage(resp, stats):
    u = getattr(resp, "usage_metadata", None)
    stats["input_tokens"] += getattr(u, "prompt_token_count", 0) or 0
    stats["output_tokens"] += getattr(u, "candidates_token_count", 0) or 0


def _finish(resp):
    try:
        return getattr(resp.candidates[0].finish_reason, "name", None)
    except (AttributeError, IndexError, TypeError):
        return None



# 4. REPAIR -- mechanical, before any id exists



def _ref_idxs(refs, n):
    out = []
    for r in refs:
        m = re.fullmatch(r"p(\d+)", str(r).strip())
        if m and 1 <= int(m.group(1)) <= n and int(m.group(1)) not in out:
            out.append(int(m.group(1)))
    return out


def normalise_segments(segments, paras):
    """A paragraph listed twice keeps its FIRST segment; a segment under
    FRAGMENT_CHARS that is not a heading joins the previous one. Returns
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
    merged = []
    for seg in out:
        text = " ".join(paras[i - 1] for i in _ref_idxs(seg.paragraph_refs, n))
        if merged and len(text) < FRAGMENT_CHARS and not is_heading_line(text):
            prev = merged[-1]
            merged[-1] = prev.model_copy(update={
                "paragraph_refs": list(prev.paragraph_refs) + list(seg.paragraph_refs),
                "cited_legislations": list(prev.cited_legislations) + list(seg.cited_legislations)})
            alias[seg.local_id] = prev.local_id
            log.append(f"fragment_merged:{seg.local_id}->{prev.local_id}:{text[:30]}")
        else:
            merged.append(seg)
    return merged, alias, log



# 5. AUDIT -- the second read. The model decides; code applies.



def audit(client, model, paras, segments, capsules, source, kind, record, stats):
    schema = audit_model_for(source, kind)
    system = prompts.build_audit_instruction(source, kind, prompts.ROLE_VOCAB[source][1],
                                             OUTCOME_BY_KIND[kind], OUTCOME_GLOSS)
    content = prompts.build_audit_content(paras, segments, capsules,
                                          aym_hint(record, kind) if source == "aym" else None,
                                          no_ruling=not any(sg.role in RULING_ROLES for sg in segments))
    for attempt in (1, 2):
        # A truncated audit is a citation loop on a very long decision; the
        # retry drops the citation list and keeps the role/outcome review.
        sch = schema if attempt == 1 else audit_model_for(source, kind, citations=False)
        sys_ = system if attempt == 1 else system + "\n\nDo not return missing_citations this time."
        resp = call_gemini(client, model, sys_, content, sch, max_tokens=32768)
        _usage(resp, stats)
        if _finish(resp) != "MAX_TOKENS":
            return sch.model_validate_json(resp.text)
    raise ValueError("audit response truncated twice")


def apply_audit(segments, capsules, verdict, source=None):
    log, by_id = [], {s.local_id: k for k, s in enumerate(segments)}
    segments, capsules = list(segments), list(capsules)
    for fix in verdict.role_fixes:
        k = by_id.get(fix.local_id)
        if k is None or segments[k].role == fix.role:
            continue
        # The prompt limits role fixes to four clear cases; the model does not
        # respect that on its own (61 changes on one Danıştay decision), so code
        # does: into the catch-all role, into a ruling role, into dissent, or a
        # kvkk stage change. Everything else is the first pass's call.
        allowed = (fix.role in RULING_ROLES or fix.role in DISSENT_ROLES
                   or fix.role in CATCHALL_ROLE.values() or fix.role == "other"
                   or source in ("kvkk", "rekabet"))
        if not allowed:
            log.append(f"audit_fix_out_of_scope:{fix.local_id}:{segments[k].role}->{fix.role}")
            continue
        # Structural guard, no vocabulary: a decision keeps at least one
        # ruling-role segment. On Yargıtay the second read moved the paragraph
        # carrying BOZULMASINA / REDDİNE out of `conclusion`, leaving no ruling.
        if segments[k].role in RULING_ROLES and fix.role not in RULING_ROLES and \
                sum(1 for sg in segments if sg.role in RULING_ROLES) == 1:
            log.append(f"audit_fix_refused:{fix.local_id}:{segments[k].role}->{fix.role}:"
                       f"would leave no ruling segment")
            continue
        log.append(f"audited_role:{fix.local_id}:{segments[k].role}->{fix.role}:{fix.reason[:70]}")
        segments[k] = segments[k].model_copy(update={"role": fix.role})
    for fix in verdict.outcome_fixes:
        if fix.outcome == "other":
            log.append(f"audit_fix_refused:outcome[{fix.capsule_index}]->other")
            continue
        if 0 <= fix.capsule_index < len(capsules) and capsules[fix.capsule_index].outcome != fix.outcome:
            log.append(f"audited_outcome:[{fix.capsule_index}]:{capsules[fix.capsule_index].outcome}"
                       f"->{fix.outcome}:{fix.reason[:70]}")
            capsules[fix.capsule_index] = capsules[fix.capsule_index].model_copy(
                update={"outcome": fix.outcome})
    for m in verdict.missing_capsules:
        log.append(f"audit_missing_capsule:{m.outcome}:{slug(m.subject)}:{m.reason[:70]}")
    added = 0
    for mc in getattr(verdict, "missing_citations", []):
        k = by_id.get(mc.local_id)
        if k is None:
            continue
        cite = CitedLegislation(**{f: getattr(mc, f) for f in CitedLegislation.model_fields})
        segments[k] = segments[k].model_copy(
            update={"cited_legislations": list(segments[k].cited_legislations) + [cite]})
        added += 1
    if added:
        log.append(f"audited_citations_added:{added}")
    return segments, capsules, (log or ["audit_agreed"])



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


def make_canonical_id(law_no, article_no, legislation_type, law_name):
    art = normalise_article_key(article_no)
    if law_no:
        return f"{re.sub(r'[^0-9]', '', str(law_no)) or law_no}/{art}"
    if legislation_type == "constitution":
        return f"constitution/{art}"
    if law_name:
        return f"{re.sub(r'[^a-zçğıöşü0-9]+', '-', lib.tr_lower(law_name)).strip('-')}/{art}"
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
    if not gaps:
        return list(segments)
    role = CATCHALL_ROLE.get(source, "other")
    fillers = [SimpleNamespace(local_id=f"fill_{k}", paragraph_refs=[f"p{i}" for i in g],
                               role=role, confidence="low", cited_legislations=[], rights=None)
               for k, g in enumerate(gaps, 1)]
    spans = ",".join(f"p{g[0]}" + (f"-p{g[-1]}" if len(g) > 1 else "") for g in gaps)
    log.append(f"uncovered_filled:{sum(len(g) for g in gaps)} paragraph(s) in {len(gaps)} "
               f"catch-all segment(s): {spans}")
    out = list(segments) + fillers
    out.sort(key=lambda sg: (_ref_idxs(sg.paragraph_refs, n) or [10 ** 9])[0])
    return out


def build_chunks(segments, alias, source, doc_id, case_no, paras, decision_date, log):
    role_field = SOURCES[source]["role_field"]
    n, doc_up = len(paras), lib.tr_upper(norm_ws(" ".join(paras)))
    chunks, id_map, counts = [], {}, Counter()
    for seg in segments:
        idxs = _ref_idxs(seg.paragraph_refs, n)
        refs = [f"p{i}" for i in idxs]
        rng = refs[0] if len(refs) == 1 else f"{refs[0]}_to_{refs[-1]}"
        text = "\n".join(paras[i - 1] for i in idxs)
        content_type, stage = derive(seg.role)
        cites, seen = [], set()
        for c in seg.cited_legislations:
            key = (c.law_no, c.article_no, c.paragraph_no, norm_ws(c.verbatim_mention))
            if key in seen:
                continue
            seen.add(key)
            if c.legislation_type == "not_legislation":
                counts["dropped_non_legislation"] += 1
                continue
            if c.legislation_type == "treaty":
                counts["dropped_treaty"] += 1          # no stored home yet (README)
                continue
            if c.verbatim_mention and lib.tr_upper(norm_ws(c.verbatim_mention)) not in doc_up \
                    and not (c.article_no and re.search(rf"(?<!\d){re.escape(c.article_no)}(?!\d)", text)):
                counts["citation_not_in_document"] += 1
                continue
            short = normalise_law_short(c.law_short)
            if c.law_short and not short:
                counts["law_short_rejected"] += 1
            cites.append({"canonical_id": make_canonical_id(c.law_no, c.article_no, c.legislation_type, c.law_name),
                          "legislation_type": c.legislation_type, "law_no": c.law_no,
                          "law_short": short, "law_name": c.law_name, "article_no": c.article_no,
                          "paragraph_no": c.paragraph_no, "verbatim_mention": c.verbatim_mention,
                          "law_date": c.law_date, "confidence": c.confidence})
        pieces = lib.split_by_size_cap(text)

        def in_piece(cite, piece):
            """The piece that carries the citation: its quote, else its law number,
            else its article number; a citation matching no piece goes to the first."""
            up = lib.tr_upper(norm_ws(piece))
            if cite["verbatim_mention"] and lib.tr_upper(norm_ws(cite["verbatim_mention"])) in up:
                return True
            if cite["law_no"] and str(cite["law_no"]) in piece:
                return True
            return bool(cite["article_no"] and re.search(rf"(?<!\d){re.escape(cite['article_no'])}(?!\d)", piece))

        # Each citation lands in exactly ONE piece: the first that contains it,
        # else the last. Attaching a segment's citations to every piece produced
        # 312 entries for 55 real citations on one document.
        assign = {}
        for c in cites:
            idx = next((k_ for k_, piece in enumerate(pieces) if in_piece(c, piece)), len(pieces) - 1)
            assign.setdefault(idx, []).append(c)
        for j, piece in enumerate(pieces, 1):
            piece_rng = rng if len(pieces) == 1 else f"{rng}_p{j}"
            piece_cites = assign.get(j - 1, [])
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
    for old, new in alias.items():                     # merged/emptied segments
        id_map.setdefault(old, []).extend(id_map.get(new, []))
    log += [f"{k}:{v}" for k, v in counts.items()]
    return chunks, id_map


def build_capsules(capsules, id_map, source, case_no, decision_date, subject_type, log,
                   role_of_chunk=None):
    out = []
    role_of_chunk = role_of_chunk or {}
    for cap in capsules:
        support = []
        for lid in cap.supporting_local_ids:
            if lid in id_map:
                support.extend(id_map[lid])
            else:
                log.append(f"dangling_support_dropped:{lid}")   # a segment the model never wrote
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



# 7. CHECK -- mechanics only. Whether a role or outcome is RIGHT was the audit's
# question; here: is every paragraph stored once, is nothing invented or empty.


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
    if chunks and capsules and all(is_separate(c["opinion_type"]) for c in capsules):
        hard.append("no_majority_capsule")
    if chunks and not any(c["role"] in RULING_ROLES for c in chunks):
        # Structural: a decision has an operative ruling somewhere. Reported,
        # never used to drop the document.
        hard.append("no_ruling_chunk")
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
            elif not looks_turkish(c[field]):
                hard.append(f"{field}_not_turkish:{c['opinion_type']}")
        sup_nums = stated_numbers(" ".join(text_of.get(i, "") for i in ids))
        for tok in sorted(stated_numbers(c["reasoning_summary"] + " " + c["conclusion_sentence"])):
            if tok not in doc_nums:
                (hard if tok[0] in "ke" else soft).append(f"number_not_in_document:{tok}")
            elif tok not in sup_nums:
                soft.append(f"number_not_in_support:{tok}")
        if c["outcome"] == "other":
            soft.append(f"outcome_other:{c['opinion_type']}")
        if c["subject_id"] == "unspecified":
            hard.append("subject_id_unspecified")
    return hard, soft



# Driver



class Failed(Exception):
    def __init__(self, review):
        super().__init__(review.get("error"))
        self.review = review


def process_document(client, model, record, source, stats):
    doc_id = str(record.get("doc_id"))
    paras, coverage, plain_ct = paragraphs(record)
    case_no = compute_case_no(record, source)
    if not paras or coverage < MIN_COVERAGE:
        return None, None, {"doc_id": doc_id, "case_no": case_no, "reason": "input_incomplete",
                            "error": f"paragraphs carry {coverage:.0%} of the record's text "
                                     f"(minimum {MIN_COVERAGE:.0%}); nothing sent"}
    kind = kind_of(record, source)
    cands = candidate_rights(record) if kind == "aym_individual_application" else []
    schema = response_model_for(source, kind, cands)
    decision_date = compute_decision_date(record, paras)
    subject_type = resolve_subject_type(record, source, kind)
    vocab = {"outcomes": OUTCOME_BY_KIND[kind], "gloss": OUTCOME_GLOSS,
             "opinion_kinds": OPINION_KINDS if source in ("kvkk", "rekabet") else OPINION_KINDS[:3],
             "legislation_types": RESPONSE_LEGISLATION_TYPES}
    prompt_args = dict(source=source, kind=kind, case_no=case_no, candidate_rights=cands,
                       examined=examined_norms(record) if kind == "aym_norm_review" else [],
                       vocab=vocab)
    # Both views (owner's decision): numbered paragraphs always; the plain
    # content_text as a reference copy only when it holds words the paragraphs lack.
    extra = plain_ct if plain_ct and coverage < 0.999 else None
    content = prompts.build_user_content(paras, plain_copy=extra)
    predicted = int(sum(len(p) for p in paras) * 0.35)
    if predicted > MAX_OUTPUT_TOKENS:
        return None, None, {"doc_id": doc_id, "case_no": case_no, "reason": "document_too_long",
                            "error": f"~{predicted:,} output tokens predicted; nothing sent"}

    def generate(system):
        raw, finish, err = None, None, None
        for attempt in (1, 2, 3):
            try:
                # Attempt 3 exists for one failure mode: a citation loop that
                # fills the output budget twice (one KVKK answer carried 298
                # citation objects). The instruction caps citations; the schema
                # cannot (Gemini rejects maxItems next to enums).
                sys_ = system if attempt < 3 else system + (
                    "\n\n## LIMIT\nYour previous answers overflowed. List at most 10 "
                    "cited_legislations per segment, each provision once, verbatim_mention "
                    "under 120 characters.")
                resp = call_gemini(client, model, sys_, content, schema, perturb=attempt == 2,
                                   max_tokens=min(MAX_OUTPUT_TOKENS, max(8192, predicted * 3)) if attempt >= 2 else None)
                raw, finish = resp.text, _finish(resp)
                _usage(resp, stats)
                if finish == "MAX_TOKENS":
                    raise ValueError("response truncated at max_output_tokens")
                return schema.model_validate_json(raw), raw
            except Exception as e:                       # noqa: BLE001
                err = f"{type(e).__name__}: {e}"
                if attempt == 3 or (attempt == 2 and finish != "MAX_TOKENS"):
                    raise Failed({"doc_id": doc_id, "case_no": case_no,
                                  "reason": "truncated" if finish == "MAX_TOKENS" else "api_or_parse_failed",
                                  "error": err, "raw_response": raw})

    def finish(parsed):
        log = []
        segments, alias, repair_log = normalise_segments(parsed.segments, paras)
        log += repair_log
        capsules = list(parsed.capsules)
        try:
            verdict = audit(client, model, paras, segments, capsules, source, kind, record, stats)
            segments, capsules, audit_log = apply_audit(segments, capsules, verdict, source)
            stats["audits"] += 1
            stats["audit_fixes"] += sum(1 for f in audit_log if f.startswith("audited_"))
            log += audit_log
        except Exception as exc:                          # noqa: BLE001
            log.append(f"audit_failed:{type(exc).__name__}:{str(exc)[:80]}")
        segments = fill_uncovered(segments, paras, source, log)
        chunks, id_map = build_chunks(segments, alias, source, doc_id, case_no, paras, decision_date, log)
        caps = build_capsules(capsules, id_map, source, case_no, decision_date, subject_type, log,
                              role_of_chunk={c["chunk_id"]: c["role"] for c in chunks})
        issues, soft = verify_document(chunks, caps, paras)
        return chunks, caps, issues, soft, log

    try:
        parsed, raw = generate(prompts.build_system_instruction(**prompt_args))
    except Failed as f:
        return None, None, f.review
    # Nothing is rejected: the model decides, code repairs mechanically (gaps
    # filled, duplicates removed, dangling pointers dropped) and REPORTS the rest.
    chunks, caps, issues, soft, log = finish(parsed)
    review = {"doc_id": doc_id, "case_no": case_no, "kind": kind, "coverage": coverage,
              "paragraphs": len(paras), "issues": issues, "soft": soft, "log": log,
              "raw_response": json.loads(raw) if raw else None}
    return chunks, caps, review


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
    problems = schema_preflight(client, model, pairs)
    if problems:
        for pr in problems:
            print("  SCHEMA REJECTED BY API: " + pr)
        raise SystemExit("stopped before sending any document: fix the schema above")
    print(f"schema preflight: {len(set(pairs))} kind(s) accepted by the API\n")
    total = Counter()
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
            picked = [r for r in usable if (source, str(r.get("doc_id"))) not in FEWSHOT_DOC_IDS][:limit]
        ran = {str(r.get("doc_id")) for r in picked}
        chunks, caps, reviews = [], [], []
        stats = Counter()
        counts = Counter()

        def flush():
            """Write after EVERY document; a targeted run merges by doc_id."""
            out_path = out_root / f"{source}.json"
            c, k, rv = chunks, caps, reviews
            if doc_ids and out_path.is_file():
                old = json.loads(out_path.read_text(encoding="utf-8"))
                kept_old = [x for x in old.get("chunks", []) if doc_of(x) not in ran]
                kept_ids = {x["chunk_id"] for x in kept_old}
                c = kept_old + chunks
                # keep only capsules of documents NOT in this run; this run's own
                # capsules are re-added from `caps` (re-appending them from the file
                # duplicated them on every write).
                k = [x for x in old.get("reasoning_capsules", [])
                     if any(i in kept_ids for i in x["supporting_chunk_ids"])] + caps
            if doc_ids:
                rv = _merge(out_root / f"{source}_review.json", reviews, ran, lambda x: str(x.get("doc_id")))
            if c or not out_path.is_file():
                out_path.write_text(json.dumps({"chunks": c, "reasoning_capsules": k},
                                               ensure_ascii=False, indent=2), encoding="utf-8")
            (out_root / f"{source}_review.json").write_text(
                json.dumps(rv, ensure_ascii=False, indent=2), encoding="utf-8")

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
                counts["failed"] += 1
                print(f"  [{source}] {doc}  FAILED: {review['reason']}: {review.get('error', '')[:100]}")
            else:
                chunks += ch
                caps += cp
                reviews.append(review)
                counts["ok"] += 1
                counts["with_issues"] += bool(review["issues"])
                notes = review["issues"] + [x for x in review["log"] if x.startswith(
                    ("audited_", "audit_missing", "audit_failed", "uncovered_filled", "dangling_support"))] + review["soft"]
                print(f"  [{source}] {doc}  {len(ch)} chunks, {len(cp)} capsules"
                      + (f"  | {'; '.join(n[:90] for n in notes[:3])}" if notes else ""))
            flush()
        pool.shutdown(wait=True)
        total.update(counts)
        print(f"=== {source} === {len(picked)} docs | stored {counts['ok']} (with issues to review "
              f"{counts['with_issues']}) | failed {counts['failed']} | {len(chunks)} chunks | {len(caps)} capsules")
        print(f"  audit: {stats['audits']} second reads, {stats['audit_fixes']} fixes | "
              f"tokens in {stats['input_tokens']:,} out {stats['output_tokens']:,}\n")
    print(f"TOTAL stored {total['ok']} | with issues {total['with_issues']} | failed {total['failed']}")


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
