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

# Generated once with uuid.uuid4() and hardcoded. NEVER regenerate -- doing so
# would change every chunk_id ever produced and break determinism entirely.
CHUNK_NAMESPACE = uuid.UUID("f08fd9b5-14a7-46ef-b7ac-a664a1b45032")

REASONING_SUMMARY_METHOD = "llm_generated"

# Stamped on every chunk so a stored chunk says which shape it is, and producer and
# consumer can move independently. Bump when a field is added, renamed or removed.
CHUNK_SCHEMA_VERSION = 2

# Which vocabulary a chunk's `role` came from. The per-source field name is kept
# for backward compatibility, but `role` + `role_vocabulary` is the indexable form:
# one field to filter on, with the distinction preserved beside it.
ROLE_VOCABULARY = {
    "firac_role": "firac",
    "court_reasoning_role": "court_reasoning",
    "regulatory_role": "regulatory",
}

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
#
# 65,535 and NOT 65,536. The documented ceiling is exclusive: the API rejects the
# request outright with
#   "maxOutputTokens value of 65536 but the supported range is
#    from 1 (inclusive) to 65536 (exclusive)"
# It is a 400 before any generation, so it costs nothing and returns no tokens --
# which also means it looks exactly like a parse failure in the logs. The earlier
# runs in this project used 65536 successfully, so the validation tightened at
# some point; pinning the off-by-one here rather than trusting the documented
# number a second time.
MAX_OUTPUT_TOKENS = 65535

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
#
# RECALIBRATED after REQUEST_SEGMENT_TEXT was turned off. The 0.50 above was
# measured while the model echoed every segment's text back, which was ~38% of
# all output. Without the echo, four documents spanning 5.5k to 32k characters
# measured 0.117, 0.254, 0.265 and 0.267 -- a median of 0.26 and a notably stable
# one across a 6x size range.
#
# 0.35 rather than 0.27: this guard refuses documents up front, so being wrong in
# the generous direction costs one truncated call while being wrong in the strict
# direction silently drops a document nobody ever looks at again. Four documents
# is a small calibration set and deserves the margin.
OUTPUT_TOKENS_PER_CHAR = 0.35

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


# docs 13.1: the five STORED legislation types. The response schema offers two
# more -- `treaty` and `not_legislation` -- which never reach storage: see
# RESPONSE_LEGISLATION_TYPES and build_chunks.

# Turkish ruling patterns used to cross-check the capsule outcome (docs 15.6).
# Now keyed by the closed `outcome` enum and matched on norm_upper() text,
# so the label IS the key and the old label-fragment table is gone. Kept under
# this name because the README and docs refer to it.

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




# =============================================================================
# Vocabularies -- every closed value the pipeline uses, in one place.
#
# `outcome`, `opinion_type` and `legislation_type` were free strings in the
# response schema. An LLM given a string field returns a string: 208 capsules
# produced 80 spellings of `outcome`, dissent capsules carried the opinion type
# ("karsi_oy") in the outcome slot, and `legislation_type` leaked "other" and
# "case" despite a prompt saying "five values only". Now the response schema
# carries a Literal per field (Gemini enforces `enum` at generation time,
# pydantic re-validates on parse), the prompt renders the same list with a
# Turkish gloss, and verify_document checks stored output against it.
#
# LANGUAGE POLICY (decided with the owner): English for closed machine keys --
# outcome, opinion kind, legislation type, roles, stages. Turkish for everything
# that carries content: subject_id, rights, dissent authors, summaries.
#
# Turkish text is ALWAYS matched on norm_upper(): tr_upper first (Python's
# .upper() mangles i/ı), then a diacritic strip, so "İHLÂL EDİLDİĞİNE",
# "İHLAL EDİLDİĞİNE" and an ASCII-scraped "IHLAL EDILDIGINE" match one pattern.
# =============================================================================

_DEACCENT = str.maketrans("çğıöşüÇĞİÖŞÜâîûÂÎÛ", "cgiosuCGIOSUaiuAIU")



def deaccent(s):
    return (s or "").translate(_DEACCENT)


def norm_upper(s):
    """tr_upper then deaccent: the ONE form every Turkish pattern is matched on."""
    return deaccent(lib.tr_upper(s or ""))


def slug(s):
    """Turkish-safe snake_case slug, diacritics kept (matches slug_right in the
    generator). Strips the U+0307 artifact first. '' when nothing survives."""
    if not s:
        return ""
    s = s.replace("̇", "")
    s = lib.tr_lower(s).strip()
    s = re.sub(r"[.'’`]", "", s)
    s = re.sub(r"[^a-zçğıöşüâîû0-9]+", "_", s).strip("_")
    return s


# --------------------------------------------------------------------------
# Archetypes: the KIND of document, resolved by code from metadata. Prompts,
# outcome subsets and the gate's invariants key on this, not on the source.
# --------------------------------------------------------------------------

ARCHETYPES = (
    "aym_individual_application", "aym_norm_review",
    "bam", "danistay", "first_degree",
    "kvkk", "rekabet",
    "yargitay_hukuk", "yargitay_ceza",
    "uyusmazlik",
)

ARCHETYPE_SOURCE = {a: (a.split("_")[0] if a.startswith(("aym_", "yargitay_")) else a)
                    for a in ARCHETYPES}

# --------------------------------------------------------------------------
# Roles -> derived fields. Docs 14.1: content_type and reasoning_stage are
# "set by rule from the role". They were being asked of the model, which
# produced 35 conclusion/reasoning chunks and 24 conclusion/analysis ones.
# --------------------------------------------------------------------------

RULING_ROLES = frozenset({"conclusion", "outcome"})
DISSENT_ROLES = frozenset({"dissent"})
CATCHALL_ROLES = frozenset({"unknown", "other"})

# rule -> analysis follows the hand-verified gold (fewshot/aym.json), not the
# model's majority pairing (125 background vs 62 analysis on the last run).
STAGE_OF_ROLE = {
    "facts": "background", "issue": "background", "other": "background",
    "unknown": "background", "background": "background",
    "rule": "analysis", "application": "analysis", "rule_application": "analysis",
    "dissent": "analysis", "analysis": "analysis",
    "conclusion": "outcome", "outcome": "outcome",
}


def derive(role):
    """(content_type, reasoning_stage) from role. Raises on an unknown role,
    which cannot happen for schema-validated output."""
    return ("ruling" if role in RULING_ROLES else "reasoning"), STAGE_OF_ROLE[role]


# --------------------------------------------------------------------------
# Outcome
# --------------------------------------------------------------------------

# value -> Turkish gloss shown to the model in the prompt.
OUTCOME_GLOSS = {
    "violation": "ihlal edildiğine (bireysel başvuru)",
    "no_violation": "ihlal edilmediğine",
    "inadmissible": "kabul edilemez olduğuna",
    "abated": "düşmesine / düşürülmesine",
    "annulled": "iptaline (norm denetimi; Danıştay'da işlemin iptali)",
    "denied": "reddine / esastan reddine / itirazın reddine / şikâyetin reddine",
    "granted": "kabulüne / kaldırılmasına",
    "partially_granted": "kısmen kabul, kısmen ret / kısmen iptal / kısmen bozma",
    "affirmed": "onanmasına",
    "corrected_affirmed": "düzeltilerek onanmasına",
    "reversed": "bozulmasına",
    "remanded": "geri çevrilmesine / iadesine / kaldırılarak mahkemesine gönderilmesine",
    "remitted": "tevdiine",
    "no_jurisdiction": "görevsizlik / yetkisizlik nedeniyle ret",
    "dismissed_procedural": "usulden ret / süre aşımı / yöntemine uygun olmayan başvuru",
    "transferred": "dosyanın başka bir mahkemeye veya daireye gönderilmesine (asıl hüküm buysa)",
    "no_decision_needed": "karar verilmesine yer olmadığına",
    "fine_imposed": "idari para cezası uygulanmasına",
    "instruction_issued": "veri sorumlusunun talimatlandırılmasına",
    "no_action": "yapılacak bir işlem olmadığına / işlem yapılmasına yer olmadığına",
    "disciplinary_referral": "Kanun m.18/3: kamu kurumu veri sorumlusu için sorumlular hakkında "
                             "disiplin işlemi yapılmasına ve sonucun Kurula bildirilmesine",
    "procedural_objection": "SADECE karşı oy için: usule, yönteme veya göreve ilişkin muhalefet",
    "other": "hiçbiri uymuyorsa; conclusion_sentence hükmü açıkça yazmalı",
}

OUTCOME_ALL = tuple(OUTCOME_GLOSS)

_DISSENT_ONLY = ("procedural_objection",)
_COMMON_TAIL = ("no_decision_needed", "other") + _DISSENT_ONLY

OUTCOME_BY_ARCHETYPE = {
    "aym_individual_application": (
        "violation", "no_violation", "inadmissible", "abated", "dismissed_procedural",
        "no_jurisdiction") + _COMMON_TAIL,
    "aym_norm_review": (
        "annulled", "denied", "partially_granted", "no_jurisdiction",
        "dismissed_procedural", "remanded") + _COMMON_TAIL,
    "yargitay_hukuk": (
        "affirmed", "corrected_affirmed", "reversed", "partially_granted", "remanded",
        "remitted", "abated", "denied", "transferred", "no_jurisdiction",
        "dismissed_procedural") + _COMMON_TAIL,
    "yargitay_ceza": (
        "affirmed", "corrected_affirmed", "reversed", "partially_granted", "remanded",
        "remitted", "abated", "denied", "transferred", "no_jurisdiction",
        "dismissed_procedural") + _COMMON_TAIL,
    "danistay": (
        "affirmed", "corrected_affirmed", "reversed", "partially_granted", "denied",
        "annulled", "granted", "no_jurisdiction", "transferred", "remanded",
        "dismissed_procedural") + _COMMON_TAIL,
    "bam": (
        "granted", "partially_granted", "denied", "affirmed", "reversed", "remanded",
        "dismissed_procedural", "no_jurisdiction", "transferred", "abated") + _COMMON_TAIL,
    "first_degree": (
        "granted", "partially_granted", "denied", "dismissed_procedural",
        "no_jurisdiction", "transferred", "remanded", "abated") + _COMMON_TAIL,
    "uyusmazlik": (
        "granted", "partially_granted", "denied", "dismissed_procedural",
        "no_jurisdiction", "transferred", "remanded", "abated") + _COMMON_TAIL,
    "kvkk": ("fine_imposed", "instruction_issued", "disciplinary_referral", "no_action",
             "denied") + _COMMON_TAIL,
    "rekabet": ("fine_imposed", "instruction_issued", "no_action", "denied",
                "granted") + _COMMON_TAIL,
}

# Every `outcome` value observed on the 208-capsule run, mapped to the enum.
# None = no honest mapping (an opinion type in the outcome slot, "unspecified",
# a label naming the violation rather than the disposition). Used to re-score
# output produced before the schema was closed and to render the worked
# examples; NEVER in the request path -- there the Literal makes an illegal
# value impossible. Context-dependent values are resolved by legacy_outcome().
_LEGACY_ANY = {
    "iptal": "annulled", "onama": "affirmed", "bozma": "reversed",
    "yetkisizlik_nedeniyle_red": "no_jurisdiction", "yetkisizlik": "no_jurisdiction",
    "yetki_yönünden_red": "no_jurisdiction", "yetki_yonunden_red": "no_jurisdiction",
    "karsi_oy": None, "karşı_oy": None, "unspecified": None, "aykırı": None, "aykiri": None,
    "dava_reddine_karar_verildi": "denied", "iptal_gerekliliği": "annulled",
    "iptal_gerekliligi": "annulled", "usule_muhalefet": "procedural_objection",
    "yontemine_uygun_degil": "dismissed_procedural", "ihlal_yok": "no_violation",
    "ihlal": "violation", "i̇hlal": "violation",
    "kabul_edilemezlik": "inadmissible", "istem_reddine": "denied", "itiraz_reddi": "denied",
    "iptal_talebinin_reddi": "denied", "iptal_talebi_reddi": "denied",
    "dosya_geri_cevirme": "remanded", "görev_ve_yetki_itirazi": "procedural_objection",
    "gorev_ve_yetki_itirazi": "procedural_objection", "iptal_edilmemesi": "denied",
    "yer_olmadığı": "no_decision_needed", "yer_olmadigi": "no_decision_needed",
    "basvurunun_geri_cevrilmesi": "remanded", "basvurunun_reddi": "denied",
    "basvuru_reddi": "denied", "dosya_iadesi": "remanded",
    "işin_geri_çevrilmesi": "remanded", "isin_geri_cevrilmesi": "remanded",
    "iptal_karsi_oy": "annulled", "esastan_inceleme_gerekliliği": "procedural_objection",
    "esastan_inceleme_gerekliligi": "procedural_objection",
    "yeniden_karar_verilmesine_yer_olmadigi": "no_decision_needed",
    "anayasa_denetimi_yapilmali": "procedural_objection",
    "konusu_kalmayan_istem": "no_decision_needed", "iade": "remanded",
    "anayasa_aykirilik": "annulled", "yontemde_ayrisik_oy": "procedural_objection",
    "yöntemine_uygun_olmayan_basvuru": "dismissed_procedural",
    "yontemine_uygun_olmayan_basvuru": "dismissed_procedural",
    "yontemine_uygun_olmama": "dismissed_procedural",
    # bam
    "denied_on_merits": "denied", "appeal_dismissed": "denied", "appeal_upheld": "granted",
    "case_transferred": "transferred", "transfer_to_another_court": "transferred",
    "partially_upheld": "partially_granted",
    "partially_upheld_partially_reversed": "partially_granted",
    "remanded_for_further_investigation": "remanded",
    # danistay
    "appeal_denied_lower_ruling_affirmed": "affirmed",
    "appeal_granted_lower_ruling_reversed": "reversed",
    "competent_court_determined_as_ankara_administrative_court": "transferred",
    "jurisdiction_denied_case_remanded": "no_jurisdiction",
    "license_revocation_and_fine_upheld": "affirmed",
    "tax_penalty_cancellation": "reversed",
    # first_degree
    "denied_procedural": "dismissed_procedural", "claim_granted": "granted",
    "accept_partially": "partially_granted", "partial_acceptance": "partially_granted",
    "denied": "denied", "denied_claim": "denied", "denied_no_fault": "denied",
    "denied_loss_certificate": "denied",
    # kvkk
    "administrative_fine_imposed": "fine_imposed",
    "administrative_fine_for_data_security_breach": "fine_imposed",
    "administrative_fine_for_data_security_violations": "fine_imposed",
    "data_security_measures_required": "instruction_issued",
    "data_deletion_request_denied": "denied", "no_action_needed": "no_action",
    "aydinlatma_ve_acik_rizanin_alinmamasi": None, "data_sharing_violation": None,
    "veri_guvenligi_ihlali": None,
    # yargitay
    "geri_cevirme": "remanded", "dosyanin_yerel_mahkemeye_geri_cevrilmesi": "remanded",
    "görev_belirleme": "transferred", "gorev_belirleme": "transferred",
    "red_ve_onama": "affirmed", "reddetme": "denied", "reddi_takdiren_para_cezasi": "denied",
    "vazgecme_nedeniyle_incelemeksizin_iade": "remitted",
}


def legacy_outcome(value, archetype=None):
    """Map an observed free-text outcome to the enum, or None when there is no
    honest mapping. A value already in the enum passes through."""
    if value is None:
        return None
    v = value.replace("̇", "").strip()
    if v in OUTCOME_ALL:
        return v
    if v == "red":
        return "denied"
    return _LEGACY_ANY.get(v, _LEGACY_ANY.get(lib.tr_lower(v)))


# --------------------------------------------------------------------------
# Opinion type
# --------------------------------------------------------------------------

OPINION_KINDS = ("majority", "dissent", "concurring", "board_decision")
OPINION_KINDS_BY_ARCHETYPE = {
    a: (("majority", "dissent", "concurring", "board_decision")
        if a in ("kvkk", "rekabet") else ("majority", "dissent", "concurring"))
    for a in ARCHETYPES
}
SEPARATE_OPINION_KINDS = ("dissent", "concurring")


def fold_opinion(kind, authors):
    """Response fields -> the stored opinion_type string.
      majority / board_decision           -> as is
      dissent + ["Göksu"]                 -> "dissent:göksu"
      dissent + ["Tanyıldız","Özden"]     -> "dissent:tanyıldız+özden"
      dissent + []                        -> "dissent"     (unsigned/redacted)
    `concurring:<author>` is a value approved as an addition to this field.
    Existing readers test startswith("dissent") and keep working."""
    if kind not in SEPARATE_OPINION_KINDS:
        return kind
    names = [slug(a) for a in (authors or []) if slug(a)]
    return f"{kind}:{'+'.join(names)}" if names else kind


def is_separate_opinion(opinion_type):
    return bool(opinion_type) and opinion_type.split(":")[0] in SEPARATE_OPINION_KINDS


# --------------------------------------------------------------------------
# Legislation type
# --------------------------------------------------------------------------

# Stored vocabulary (docs 13.1). Unchanged.
LEGISLATION_TYPES = ("statute", "decree_law", "constitution", "regulation", "directive")
# What the model may answer. `treaty` and `not_legislation` are response-only:
# not_legislation is dropped (case law, doctrine, party names -- previously the
# model invented "other"/"case" for these); treaty is dropped WITH a counted
# flag until the owner decides whether to widen the stored vocabulary (README).
RESPONSE_LEGISLATION_TYPES = LEGISLATION_TYPES + ("treaty", "not_legislation")

CONFIDENCE = ("high", "low")

LAW_SHORT_RE = re.compile(r"^[A-ZÇĞİÖŞÜ]{2,8}$")

# --------------------------------------------------------------------------
# AYM metadata -> accepted outcomes (independent cross-check, gate I7b)
# --------------------------------------------------------------------------

AYM_NORM_RESULT_ACCEPT = {
    "Esas İptal": {"annulled", "partially_granted"},
    "Esas - Ret": {"denied", "partially_granted"},
    "İlk - Ret": {"denied", "dismissed_procedural", "no_jurisdiction"},
    "İlk - İşin Geri Çevrilmesi": {"remanded"},
    "Esas - Karar Verilmesine/İncelenmesine Yer Olmadığı": {"no_decision_needed"},
    "İlk - Karar Verilmesine/İncelenmesine Yer Olmadığı": {"no_decision_needed"},
}

AYM_INDIVIDUAL_OUTCOME = {
    "İhlal": "violation",
    "İhlal Olmadığı": "no_violation",
    "Açıkça Dayanaktan Yoksunluk": "inadmissible",
    "Başvuru Yollarının Tüketilmemesi": "inadmissible",
    "Konu Bakımından Yetkisizlik": "inadmissible",
    "Kişi Bakımından Yetkisizlik": "inadmissible",
    "Zaman Bakımından Yetkisizlik": "inadmissible",
    "Yer Bakımından Yetkisizlik": "inadmissible",
    "Süre Aşımı": "inadmissible",
    "Anayasal ve Kişisel Önemin Olmaması": "inadmissible",
    "Başvurunun Reddi": "dismissed_procedural",
    "Düşme": "abated",
    "İşlemden Kaldırılma": "abated",
    "İncelenmesine Yer Olmadığı": "no_decision_needed",
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
    # Closed. `treaty` is the model's honest signal for the ECHR and its
    # protocols; `not_legislation` is the escape hatch for case law, doctrine
    # and party names, which the model used to file under an invented "other"
    # or "case". Neither reaches storage (build_chunks drops both, treaty with
    # a counted flag) -- see RESPONSE_LEGISLATION_TYPES.
    legislation_type: Literal[RESPONSE_LEGISLATION_TYPES]
    confidence: Literal[CONFIDENCE]
    verbatim_mention: Optional[str] = None


class SegmentBase(BaseModel):
    """`role`, `confidence` and `paragraph_refs` are REQUIRED and
    enum-constrained. Verified necessary: when these were Optional with a None
    default, the model dropped `role` for entire documents at a time -- all 38
    bam chunks, 7 of 15 first_degree -- and invented `reasoning_stage:
    "dissent"`. An optional field in a structured-output schema is an invitation
    to omit it, and a free string is an invitation to invent a value.

    `content_type` and `reasoning_stage` are NOT asked of the model any more:
    docs 14.1 defines both as a rule of the role, and asking produced 35
    conclusion/reasoning chunks. They are derived in build_chunks (derive).
    `rights` is added per document by response_model_for, as a Literal over the
    candidate list, and only for AYM individual applications."""
    local_id: str
    paragraph_refs: List[str]
    confidence: Literal[CONFIDENCE]
    source_type: Optional[str] = None        # echoed for cross-check only
    case_no: Optional[str] = None            # echoed for cross-check only
    cited_legislations: List[CitedLegislation] = Field(default_factory=list)


class CapsuleBase(BaseModel):
    """`outcome`, `opinion_type` and (for AYM individual applications)
    `subject_id` are added per archetype by response_model_for as Literals.
    `dissent_authors` replaces the free-text "dissent:<surname>" suffix; code
    folds kind + authors back into the stored opinion_type string
    (fold_opinion), so the stored schema does not change."""
    # REQUIRED, with a floor. The first closed-schema run returned a capsule
    # with "" in both text fields and no supporting ids while the segments were
    # complete -- the same "optional means omit" failure recorded above for
    # segments. min_length maps to minLength/minItems in the response schema,
    # and pydantic rejects a violation on parse, which takes the retry path.
    conclusion_sentence: str = Field(min_length=20)
    reasoning_summary: str = Field(min_length=80)
    dissent_authors: List[str] = Field(default_factory=list)
    supporting_local_ids: List[str] = Field(min_length=1)
    case_no: Optional[str] = None


_RESPONSE_MODELS = {}


# Whether to make the model echo each segment's TEXT back to us.
#
# It used to be required, and it was expensive and dangerous for no benefit:
#
#   COST      echoed text was ~38% of all output tokens. We already hold that
#             text -- the stored `text` is assembled by code from paragraph_refs,
#             precisely because the model's copy proved unreliable (it dropped
#             characters mid-word and lowercased headings).
#   FAILURE   a 1964 decision looped inside the echo, repeating one sentence of
#             quoted statute until it exhausted the entire 65,535-token budget,
#             twice, having emitted only four segments.
#   BENEFIT   one diagnostic, `model_text_differs`, which is largely redundant
#             with the paragraph-ref validation that already runs.
#
# Set True to restore the echo and the diagnostic with it.
REQUEST_SEGMENT_TEXT = False


def response_model_for(source, archetype=None, candidates=()):
    """Per-DOCUMENT response model, cached by (source, archetype, candidates).

    `role` is a Literal over the source's vocabulary, so aym cannot return
    'rule_application' and kvkk cannot return 'facts'. `outcome` is a Literal
    over the archetype's subset (OUTCOME_BY_ARCHETYPE), so a norm review
    cannot return 'onama' and a dissent cannot return 'karsi_oy'.
    `opinion_type` is a Literal over the opinion kinds. For an AYM individual
    application `rights` and `subject_id` are Literals over the candidate
    rights the metadata names; for every other document `rights` is not asked
    at all and `subject_id` is a free Turkish slug that code normalises.
    """
    archetype = archetype or source
    candidates = tuple(candidates or ())
    key = (source, archetype, candidates)
    if key not in _RESPONSE_MODELS:
        roles = tuple(prompts.ROLE_VOCAB[source][1])
        seg_extra = {"text": (str, ...)} if REQUEST_SEGMENT_TEXT else {}
        cap_extra = {}
        if candidates:
            seg_extra["rights"] = (Optional[List[Literal[candidates]]], None)
            cap_extra["subject_id"] = (Literal[candidates], ...)
        else:
            cap_extra["subject_id"] = (Optional[str], None)
        segment = create_model(
            f"Segment_{archetype}", __base__=SegmentBase, role=(Literal[roles], ...),
            **seg_extra)
        outcomes = tuple(OUTCOME_BY_ARCHETYPE.get(archetype)
                         or OUTCOME_BY_ARCHETYPE[source])
        kinds = tuple(OPINION_KINDS_BY_ARCHETYPE.get(archetype, OPINION_KINDS))
        capsule = create_model(
            f"Capsule_{archetype}", __base__=CapsuleBase,
            outcome=(Literal[outcomes], ...),
            opinion_type=(Literal[kinds], ...),
            **cap_extra)
        _RESPONSE_MODELS[key] = create_model(
            f"DocumentResponse_{archetype}",
            segments=(List[segment], ...),
            capsules=(List[capsule], ...))
    return _RESPONSE_MODELS[key]



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
    not 'IYUK' (Python's .upper() breaks Turkish dotted/dotless i).

    Returns None for anything not shaped like a law abbreviation (2-8 Turkish
    capitals, no digits): the model once filed a court name,
    'İCRAHUKUKMAHKEMESİ', in this slot. Gemini ignores JSON-schema `pattern`
    and a raising pydantic validator would reject a whole document for one
    abbreviation, so the shape is enforced here and the caller flags the drop
    (law_short_rejected)."""
    if not law_short:
        return None
    s = re.sub(r"[.\s ]+", "", str(law_short))
    s = lib.tr_upper(s)
    if not LAW_SHORT_RE.match(s):
        return None
    return s or None


def citation_is_grounded(c, text):
    """Is this citation supported by the text, even when the quoted phrase is not?

    A `verbatim_mention` that is not a literal substring is not automatically an
    invention. Turkish decisions enumerate provisions in one breath -- "Anayasa'nın
    2., 6., 10. ... ve 161. maddelerine" -- so a per-article citation can only ever
    be a reconstruction. Requiring a literal quote discarded 23 real constitutional
    citations from a single decision.

    Grounding is the weaker, honest test: the article number must appear in the
    text, AND so must something identifying the law -- its number, its
    abbreviation, or a distinctive word of its name. Both present means the text
    really does cite that provision, whatever words the model chose to quote.
    """
    body = norm_ws(text)
    if not body:
        return False

    article = (c.article_no or "").strip()
    if article:
        # Word-boundary match so article 8 is not satisfied by "1985" or "89".
        if not re.search(rf"(?<!\d){re.escape(article)}(?!\d)", body):
            return False

    identifiers = []
    if c.law_no:
        identifiers.append(re.escape(str(c.law_no).strip()))
    if c.law_short:
        identifiers.append(re.escape(c.law_short.strip().replace(".", r"\.?")))
    if c.law_name:
        # A distinctive word from the law's name: the longest one, which avoids
        # matching on "Kanun" or "Hakkında" that appear in every statute title.
        words = [w for w in re.split(r"\W+", c.law_name) if len(w) > 5]
        if words:
            identifiers.append(re.escape(max(words, key=len)))
    if c.legislation_type == "constitution":
        identifiers.append("Anayasa")

    if not identifiers:
        return False
    return any(re.search(i, body, re.IGNORECASE) for i in identifiers)


def is_legislation(c):
    """Is this citation a legal instrument the store can hold?

    With `legislation_type` a closed Literal, the answer is the type itself. The
    old heuristic ("outside the vocabulary AND no law_no AND no law_name")
    admitted anything carrying a law_name, which is how 'Anayasa Mahkemesi
    Kararlar Dergisi' and 40 other case-law references were stored as
    legislation under an invented type. `not_legislation` is now the model's
    explicit answer for those; `treaty` is honest but has no stored home yet.
    """
    return c.legislation_type in LEGISLATION_TYPES


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
    if source == "aym":
        # AYM's two document kinds have different subjects. An individual
        # application is about a RIGHT; a norm review is about a STATUTE, and
        # labelling 65 statute reviews "right" would promise a subject they do
        # not contain -- and make them indistinguishable from the other kind.
        return ("right" if aym_variant(record) == "individual_application"
                else "constitutional_norm_review")
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


# The section headings each source's prompt names. Kept here rather than imported
# from preflight so the generator has no dependency on a diagnostic tool, and
# stored without diacritics because the same heading is scraped as both "GEREKÇE"
# and "GEREKCE". Sources whose prompt is written around function rather than
# markers have an empty list and never trigger the fallback.
PROMPT_MARKERS = {
    "aym": ["OLAY VE OLGULAR", "ILGILI HUKUK", "GENEL ILKELER", "DEGERLENDIRME",
            "HUKUM", "KARSIOY", "KARSI OY", "INCELEME VE GEREKCE"],
    "bam": ["DAVA", "CEVAP", "ILK DERECE", "ISTINAF", "GEREKCE", "HUKUM"],
    "danistay": ["ISTEMIN KONUSU", "YARGILAMA SURECI", "TEMYIZ EDEN",
                 "HUKUKI DEGERLENDIRME", "KARAR SONUCU", "KARSI OY"],
    "first_degree": ["DAVA", "GEREGI DUSUNULDU", "HUKUM", "GEREKCE"],
    "kvkk": [], "yargitay": [], "rekabet": [], "uyusmazlik": [],
}

def count_markers(paragraphs, source):
    """How many of this source's expected headings appear. None when the source's
    prompt does not rely on headings at all, so the caller can tell "no markers
    expected" apart from "markers expected and missing"."""
    expected = PROMPT_MARKERS.get(source) or []
    if not expected:
        return None
    body = " ".join(paragraphs).translate(_DEACCENT).upper()
    return sum(1 for m in expected if m.translate(_DEACCENT).upper() in body)


def aym_variant(record):
    """Which KIND of AYM decision this is. They are two different documents.

    An INDIVIDUAL APPLICATION (bireysel başvuru) has an applicant who says a right
    was violated; it carries examination_results naming those rights, and FIRAC
    plus `rights` describes it well.

    A NORM REVIEW (norm denetimi) has no applicant and no violated right. A court
    or parliamentary group asks whether a STATUTE is constitutional. It carries
    examined_norms instead -- the provisions under review, with the outcome for
    each -- and a prompt about applicants and rights does not describe it.

    Both were always present, but the proportions inverted: 94% of the earlier
    200-document sample were individual applications, while a year-spread export
    returned 65 norm reviews and zero individual applications. Detecting the kind
    per record is the only thing that survives that.
    """
    data = parse_metadata(record).get("data") or {}
    if data.get("examination_results"):
        return "individual_application"
    if data.get("examined_norms") or data.get("application_type"):
        return "norm_review"
    return "individual_application"


def archetype_of(record, source):
    """The KIND of document, resolved by code from metadata already read.

    The corpus has more kinds than sources: a 1963 norm review and a 2023
    individual application both arrive as `aym`; a fraud cassation and a
    land-registry cassation both arrive as `yargitay`. Prompts, the outcome
    vocabulary offered to the model and the content gate's invariants all key
    on this value, so the same instruction is never sent to two different
    kinds of document. Returns one of ARCHETYPES.
    """
    if source == "aym":
        return "aym_" + aym_variant(record)
    if source == "yargitay":
        subject = resolve_subject_type(record, source)
        return "yargitay_ceza" if subject == "criminal_offence" else "yargitay_hukuk"
    return source


def examined_norms(record):
    """The statutes under constitutional review, from AYM norm-review metadata.

    Present on 64 of 65 norm reviews and completely unused until now. It is the
    norm-review equivalent of candidate_rights: code supplies the candidates, the
    model picks the one the decision turns on. Without it `subject_id` was being
    set to "unspecified" on every one of these documents, which makes them
    unfindable by subject -- the exact failure banned for every other source.
    """
    data = parse_metadata(record).get("data") or {}
    out = []
    for norm in data.get("examined_norms") or []:
        law = (norm.get("norm_code_name") or "").strip()
        for art in norm.get("articles") or []:
            art_no = (art.get("article_no") or "").strip()
            if not (law or art_no):
                continue
            out.append({"law": law, "article": art_no,
                        "clause": art.get("clause"),
                        "result": art.get("review_type_result"),
                        "against": art.get("basis_constitutional_provisions") or []})
    return out


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


BLOCK_TAGS = ("p", "li", "h1", "h2", "h3", "tr", "div")


def extract_paragraphs_html_blocks(raw):
    """Paragraphs from HTML where the substance lives in lists and tables.

    KVKK decision summaries put the complaint points, the data controller's
    defence, the Board's evaluations and the decision itself in <ul><li>, and
    the docket (Karar Tarihi / Karar No / Konu Özeti) in <table><td>. The
    <p>-only extractor kept the six connective sentences between those lists
    and dropped everything else: measured over all 193 usable KVKK records,
    191 lost more than 200 characters and the median loss was 6,597 characters
    -- roughly 80% of every document. The "hollow documents" this pipeline had
    been diagnosing upstream were produced here.

    Recursive walk in document order. An element with no block descendants is
    one paragraph (a <tr> becomes "Karar No : 2023/2007", cells joined by a
    space). A container's own inline text -- an <em> sentence sitting directly
    inside a <li> that also holds a nested <ul>, seen in one record -- is
    flushed as its own paragraph before the nested blocks, so nothing is lost
    to nesting. Verified to recover the whole document: 1.000 coverage on 193
    of 193 records.
    """
    from html import unescape
    from bs4 import BeautifulSoup, NavigableString, Comment
    soup = BeautifulSoup(unescape(raw), "html.parser")
    out = []

    def flush(buf):
        txt = norm_ws(" ".join(buf).replace(" ", " "))
        if txt:
            out.append(txt)

    def walk(node):
        buf = []
        for child in node.children:
            if isinstance(child, Comment):
                continue
            if isinstance(child, NavigableString):
                buf.append(str(child))
            elif child.name in BLOCK_TAGS or child.find(BLOCK_TAGS):
                flush(buf)
                buf = []
                if child.find(BLOCK_TAGS):
                    walk(child)                    # container: recurse
                else:
                    flush([child.get_text(separator=" ")])   # leaf block
            else:
                buf.append(child.get_text(separator=" "))    # inline: em, strong, a, span
        flush(buf)

    walk(soup)
    return out


# How each source's paragraphs are recovered. Named explicitly rather than
# inferred from the field name, because two sources store HTML and need
# different extractors -- inferring from "html_content" silently picked the
# wrong one for yargitay, and the <p>-only walk silently dropped 80% of kvkk.
PARAGRAPH_STRATEGY = {
    "aym": "html_p", "kvkk": "html_blocks",
    "yargitay": "html_br",
    "bam": "lines", "danistay": "lines", "first_degree": "lines",
}


def _split_with(raw, strategy):
    """One strategy applied to one string. No source knowledge, no fallbacks."""
    if not (raw or "").strip():
        return []
    if strategy == "html_p":
        return lib.aym_extract_paragraph_texts(raw)
    if strategy == "html_br":
        return extract_paragraphs_html_br(raw)
    if strategy == "html_blocks":
        return extract_paragraphs_html_blocks(raw)
    return [p for p in (norm_ws(x) for x in raw.splitlines()) if p]


def extract_paragraphs(record, source):
    """The returned list IS the text Gemini sees, and the text chunks are later
    assembled from, so every consumer must reproduce it exactly.

    RESCUE, NOT OPTIMISE. The configured strategy is tried first and kept whenever
    it produces a real split. Only when it collapses to a single paragraph do we
    look at the other column -- and then we take whichever alternative recovers
    the most paragraphs.

    This exists because the scraper changed format mid-corpus. bam and danistay
    records used to store newline-separated text in content_text with html_content
    empty; newer records of the SAME sources store a flattened content_text with
    no newlines at all and put the structure in html_content as <br> tags, exactly
    as yargitay does. Both shapes are present in danistay today -- 2021, 2023 and
    2024 are old-style while the rest are new-style -- so no fixed per-source
    choice can be right.

    A pre-flight check found this on 20 of 98 documents in a new export, before
    any of them were sent to the model. Unrescued, each would have become a single
    chunk with a useless paragraph reference.

    The rescue only fires on a collapse, never on a working split, so behaviour on
    every document already chunked is unchanged -- which regression_check verifies.
    """
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
    strategy = PARAGRAPH_STRATEGY[source]
    paras = _split_with(record.get(field) or "", strategy)
    if len(paras) > 1:
        return paras

    # Collapsed to one paragraph (or none). Try the other column and the other
    # strategies, keeping whichever recovers the most. Ordered so the result is
    # deterministic when two candidates tie.
    other = "content_text" if field == "html_content" else "html_content"
    best = paras
    for cand_field, cand_strategy in ((other, "html_br"), (other, "html_p"),
                                      (other, "lines"), (field, "html_br"),
                                      (field, "lines")):
        alt = _split_with(record.get(cand_field) or "", cand_strategy)
        if len(alt) > len(best):
            best = alt
    return best


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


def call_gemini(client, model, system_instruction, user_content, schema,
                perturb=False, max_tokens=None):
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
        # The retry perturbs HARDER than it used to. At temperature 0.15 with a
        # 0.6 penalty a degenerate loop reproduced itself exactly -- both attempts
        # filled the entire 65,535-token budget and emitted two citations between
        # them. If the first sampling regime could not escape the cycle, a barely
        # different one will not either.
        temperature=0.4 if perturb else 0,
        top_p=1,
        seed=42,
        frequency_penalty=1.0 if perturb else None,
        max_output_tokens=max_tokens or MAX_OUTPUT_TOKENS,
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


FRAGMENT_CHARS = 40


def normalise_segments(segments, paragraphs):
    """Two MECHANICAL repairs on the model's segment list, before any chunk id
    exists. Both are things code can decide exactly, so they are fixed here
    rather than sent back to the model or rejected:

    1. A paragraph listed in two segments is kept in the FIRST and removed from
       the later one (seen: "...Kararı ile," as the last line of the analysis
       block and again as the first line of the outcome block). Stored text is
       then never duplicated.
    2. A segment whose whole text is under FRAGMENT_CHARS and is not a docket
       line is merged into the previous segment ("karar verilmiştir." closing
       the outcome list; "değerlendirmelerinden hareketle;" closing the
       evaluations). A connective one-liner is not a retrievable unit.

    Returns (segments, repair_flags). Segment objects are pydantic models; the
    repaired list holds shallow copies with `paragraph_refs` rewritten."""
    n = len(paragraphs)
    flags, owner, out = [], {}, []          # owner: paragraph index -> local_id
    for seg in segments:
        refs, stolen_from = [], set()
        for r in seg.paragraph_refs:
            m = re.fullmatch(r"p(\d+)", str(r).strip())
            if m and 1 <= int(m.group(1)) <= n:
                i = int(m.group(1))
                if i in owner:
                    flags.append(f"duplicate_ref_removed:{seg.local_id}:{r}")
                    stolen_from.add(owner[i])
                    continue
                owner[i] = seg.local_id
            refs.append(r)
        if not refs and seg.paragraph_refs:
            # Every paragraph already belonged to an earlier segment. A capsule
            # citing this segment is re-pointed to the one that kept the text
            # (build_chunks reads the "->" alias), so support never dangles.
            target = sorted(stolen_from)[0] if stolen_from else None
            flags.append(f"segment_emptied_by_dedupe:{seg.local_id}->{target}")
            continue
        out.append(seg.model_copy(update={"paragraph_refs": refs}))

    merged = []
    for seg in out:
        idxs = [int(str(r)[1:]) for r in seg.paragraph_refs if re.fullmatch(r"p\d+", str(r))]
        text = " ".join(paragraphs[i - 1] for i in idxs)
        up = norm_upper(text).strip()
        is_fragment = (len(text) < FRAGMENT_CHARS and idxs and merged
                       and not _is_heading_line(text))
        if is_fragment:
            prev = merged[-1]
            merged[-1] = prev.model_copy(
                update={"paragraph_refs": list(prev.paragraph_refs) + list(seg.paragraph_refs),
                        "cited_legislations": list(prev.cited_legislations) + list(seg.cited_legislations)})
            flags.append(f"fragment_merged:{seg.local_id}->{prev.local_id}:{text[:30]}")
        else:
            merged.append(seg)
    return merged, flags


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

    segments, repair_flags = normalise_segments(parsed.segments, paragraphs)
    flags += repair_flags
    for seg in segments:
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
        # Only possible when the model was asked to echo the text. With
        # REQUEST_SEGMENT_TEXT off there is nothing to compare against -- the
        # grounding guarantee then rests entirely on paragraph_refs resolving,
        # which is checked above and is the stronger of the two signals anyway.
        model_text = getattr(seg, "text", None)
        if model_text is not None:
            m_norm, d_norm = norm_ws(model_text), norm_ws(seg_text)
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
        # Docs 14.1: both are a rule of the role. Deriving them here makes the
        # 35 conclusion/reasoning and 24 conclusion/analysis combinations of the
        # last run impossible rather than merely flagged.
        content_type, reasoning_stage = derive(role)

        # Size cap BEFORE ids are final, so each piece gets its own chunk_id.
        pieces = lib.split_by_size_cap(seg_text)
        for j, piece in enumerate(pieces, 1):
            rng = base_range if len(pieces) == 1 else f"{base_range}_p{j}"
            chunk_id = make_chunk_id(doc_id, rng)
            id_map.setdefault(seg.local_id, []).append(chunk_id)
            for f in repair_flags:
                if f.startswith(("fragment_merged:", "segment_emptied_by_dedupe:")) and                         f.split(":")[1].endswith("->" + seg.local_id):
                    id_map.setdefault(f.split(":")[1].split("->")[0], []).append(chunk_id)

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
                    # Attach the citation to the piece it appears in. When the
                    # mention is absent from the whole segment, decide between a
                    # RECONSTRUCTION and a HALLUCINATION before discarding it.
                    #
                    # Courts enumerate: "Anayasa'nın 2., 6., 10., 42., 88., 89.,
                    # 104., 123., 130., 131. ve 161. maddelerine". Citing article
                    # 89 from that sentence CANNOT be a contiguous quote, so the
                    # model reconstructs "Anayasa'nın 89." -- correct, and
                    # previously dropped. In one smoke-tested decision that threw
                    # away 23 genuine constitutional citations.
                    if norm_ws(c.verbatim_mention) not in norm_ws(seg_text):
                        if citation_is_grounded(c, seg_text):
                            if not citation_is_grounded(c, piece):
                                continue          # belongs to a different piece
                            flags.append(
                                f"verbatim_mention_reconstructed:{seg.local_id}")
                        else:
                            flags.append(
                                f"verbatim_mention_not_in_text:{seg.local_id}")
                            continue
                    else:
                        continue
                if c.legislation_type == "treaty":
                    # Honest signal (ECHR, its protocols) with no stored home:
                    # the documented vocabulary has five values and widening it is
                    # the owner's call (README). Dropped WITH a flag so the loss is
                    # counted rather than hidden under an invented type.
                    flags.append(f"dropped_treaty:{seg.local_id}:"
                                 f"{(c.law_name or c.verbatim_mention or '')[:40]}")
                    continue
                if not is_legislation(c):
                    # `not_legislation`: the court citing its own precedent
                    # ("Mehmet Serif Ay (B. No: 2012/1181)"), doctrine, a party
                    # name. Measured on the earlier run: 17% of citations, every
                    # one AYM case law. Dropped but FLAGGED -- a silent drop would
                    # hide the model ignoring an explicit instruction. AYM
                    # precedent IS valuable and belongs in a cited_decisions field
                    # (the metadata already carries referenced_decisions); out of
                    # scope here.
                    flags.append(f"dropped_non_legislation:{seg.local_id}:"
                                 f"{(c.verbatim_mention or '')[:40]}")
                    continue
                canonical = make_canonical_id(
                    c.law_no, c.article_no, c.legislation_type, c.law_name)
                if canonical is None:
                    # No law_no, not the constitution, and no law_name either -- there
                    # is nothing to join on, so it cannot be deduplicated at all.
                    flags.append(f"citation_not_identifiable:{seg.local_id}:"
                                 f"{c.legislation_type}")
                law_short = normalise_law_short(c.law_short)
                if c.law_short and not law_short:
                    flags.append(f"law_short_rejected:{seg.local_id}:{c.law_short[:30]}")
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
                "content_type": content_type,
                "firac_role": role if source == "aym" else None,
                "reasoning_stage": reasoning_stage,
                # Present in the response model only for AYM individual
                # applications (a Literal over the candidate rights); null
                # everywhere else, never omitted.
                "rights": (getattr(seg, "rights", None) or None) if source == "aym" else None,
                "confidence": seg.confidence,
            }
            if role_field != "firac_role":
                chunk[role_field] = role
            # The SAME value under a stable name, alongside the per-source field.
            #
            # Our role field is named differently per source -- firac_role,
            # court_reasoning_role, regulatory_role -- which is fine to read but
            # impossible to INDEX: a field whose name varies by row cannot be
            # filtered as one thing, so "show me only the reasoning parts" would
            # need three filterable attributes and three query branches forever.
            # `role_vocabulary` keeps the distinction that made three names
            # tempting, without paying for it at every consumer.
            #
            # Purely additive: the per-source field above is untouched, so every
            # existing reader keeps working and no regeneration is required for
            # correctness -- the next run simply carries both.
            chunk["role"] = role
            chunk["role_vocabulary"] = ROLE_VOCABULARY[role_field]
            chunk["schema_version"] = CHUNK_SCHEMA_VERSION
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

        # kind + authors -> the stored "dissent:<author>" string. A separate
        # opinion (dissent or concurring) must rest on dissent-role chunks and a
        # majority/board capsule must not -- the role marks the chunk, the
        # opinion type marks the capsule, and code checks that they agree.
        opinion = fold_opinion(cap.opinion_type, cap.dissent_authors)
        is_separate = is_separate_opinion(opinion)
        has_dissent_support = any(role_of.get(cid) in DISSENT_ROLES for cid in supporting)
        if is_separate != has_dissent_support:
            flags.append(f"opinion_type_support_mismatch:{opinion}")
        if is_separate and ":" not in opinion:
            flags.append(f"separate_opinion_unsigned:{opinion}")

        # Whether the outcome matches the operative ruling is a READING question
        # and is decided by the audit call (audit_document), not by a phrase table.
        if cap.outcome == "other":
            flags.append(f"outcome_other:{opinion}")

        # subject_id is a RETRIEVAL field in Turkish: normalised to one slug
        # shape so 'bilirki$ilik_listesinden_cikarma' and a dotted-i artifact
        # cannot split one subject into two facets. For an AYM individual
        # application the response model already constrains it to the
        # candidate rights, which are slugs already.
        subject_id = slug(cap.subject_id) or "unspecified"
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
            "outcome": cap.outcome,
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


# =============================================================================
# verify_document -- MECHANICS ONLY. Docs 15.2: code owns what is arithmetic, a
# fixed lookup, or must be byte-identical; Gemini owns anything that requires
# reading the text. So this checks: every paragraph in exactly one chunk, no
# orphan fragments, capsule support consistent with roles, numbers the summary
# states exist in the decision, values inside the vocabulary. Whether a ROLE or
# an OUTCOME is right is a reading question and goes to audit_document, a second
# model call. The Turkish phrase detectors that used to live here were never
# complete (three rounds of KVKK fixes, every gap a false rejection) and are gone.
# =============================================================================

JUNK_LINE_RE = re.compile(r"^(\d+|\d{4}/1999999|ee3|KARAR ICERIGI|[.:])$")


def _is_heading_line(text):
    """A short line with no lowercase letter: a heading, a docket label, a
    letter-spaced title, a signature name. Mechanical -- no vocabulary."""
    t = (text or "").strip()
    return len(t) <= 80 and not re.search(r"[a-zçğıöşüâîû]", t)


# Numbers a capsule may only use if the decision states them. Compared as
# NUMBERS, not strings: "33. maddesi", "33 uncu madde", "madde 33" and
# "33'inci maddesinin" are one article.
ARTICLE_NUM_RE = re.compile(
    r"\b(\d+)\s*(?:\.|['’ʼ]?\s*[IU]NC[IU]|['’ʼ]?\s*NC[IU])?\s*(?:/\s*[A-Z]\s*)?MADDE|\bMADDE\s+(\d+)")
LAW_NUM_RE = re.compile(r"\b(\d{2,5})\s+SAYILI\b")
CASE_NUM_RE = re.compile(r"\b(\d{4}/\d+)\b")


_ORDINAL_WORDS = {"BIRINCI": "1", "IKINCI": "2", "UCUNCU": "3", "DORDUNCU": "4",
                  "BESINCI": "5", "ALTINCI": "6", "YEDINCI": "7", "SEKIZINCI": "8",
                  "DOKUZUNCU": "9", "ONUNCU": "10"}
_ORDINAL_WORD_RE = re.compile(r"\b(" + "|".join(_ORDINAL_WORDS) + r")\s+(MADDE|FIKRA)")


def _stated_numbers(up):
    """{'m33', 'k6200', 'e1963/18', ...} from norm_upper() text. Ordinal words
    ("birinci maddesinin") count as their digit."""
    up = _ORDINAL_WORD_RE.sub(lambda m: _ORDINAL_WORDS[m.group(1)] + ". " + m.group(2), up)
    out = set()
    for a, b in ARTICLE_NUM_RE.findall(up):
        out.add("m" + (a or b))
    out |= {"k" + n for n in LAW_NUM_RE.findall(up)}
    out |= {"e" + n for n in CASE_NUM_RE.findall(up)}
    return out


def _segments_of(chunks):
    """Group size-cap pieces back into the model's segments and order them by
    first paragraph. Detectors judge the SEGMENT text: a piece holding only the
    tail of a quoted statute, or a heading without its verb, misfires."""
    groups = {}
    for ch in chunks:
        base = re.sub(r"_p\d+$", "", ch["chunk_label"])
        g = groups.setdefault(base, {"labels": [], "texts": [], "refs": None,
                                     "role": ch.get("role") or ch.get("_role"),
                                     "chunk_ids": []})
        g["labels"].append(ch["chunk_label"])
        g["texts"].append(ch["text"])
        g["chunk_ids"].append(ch["chunk_id"])
        g["refs"] = g["refs"] or sorted(int(r[1:]) for r in ch["source_paragraph_ids"]
                                        if re.fullmatch(r"p\d+", str(r)))
    segs = []
    for base, g in groups.items():
        g["text"] = "\n".join(g["texts"])
        g["lines"] = [l for l in g["text"].split("\n") if l.strip()]
        g["up"] = norm_upper(g["text"])
        g["first"] = g["refs"][0] if g["refs"] else 10 ** 9
        g["last"] = g["refs"][-1] if g["refs"] else -1
        segs.append(g)
    segs.sort(key=lambda s: s["first"])
    return segs




def verify_document(chunks, capsules, paragraphs, source, archetype, record=None):
    """Returns {"hard": [items], "soft": [items]}; an item is
    {"kind", "detail", "labels"}. Hard = the document may not be stored."""
    hard, soft = [], []

    def add(sev, kind, detail, labels=()):
        (hard if sev == "hard" else soft).append(
            {"kind": kind, "detail": detail, "labels": list(labels)})

    n = len(paragraphs)
    segs = _segments_of(chunks)
    doc_chars = sum(len(p) for p in paragraphs) or 1

    # ---- coverage: every paragraph in exactly one segment ---------------------
    count = [0] * (n + 1)
    for s in segs:
        for i in s["refs"] or []:
            if 1 <= i <= n:
                count[i] += 1
    overlap = [i for i in range(1, n + 1) if count[i] > 1]
    if overlap:
        add("hard", "paragraph_overlap",
            f"{len(overlap)} paragraph(s) in more than one segment: "
            f"{', '.join('p%d' % i for i in overlap[:8])}")
    unref = [i for i in range(1, n + 1) if count[i] == 0]
    content_missing, minor_missing = [], []
    for i in unref:
        p = paragraphs[i - 1]
        if len(p) < FRAGMENT_CHARS or _is_heading_line(p) or JUNK_LINE_RE.match(norm_upper(p)):
            minor_missing.append(i)
        else:
            content_missing.append(i)
    if content_missing:
        chars = sum(len(paragraphs[i - 1]) for i in content_missing)
        sev = "hard" if (chars > 200 or chars / doc_chars > 0.05
                         or any(len(paragraphs[i - 1]) >= 100 for i in content_missing)) else "soft"
        add(sev, "uncovered_content",
            f"{len(content_missing)} paragraph(s), {chars} chars, in NO segment: "
            f"{_ranges(content_missing)}")
    if minor_missing:
        add("soft", "uncovered_minor",
            f"{len(minor_missing)} header/short line(s) in no segment: {_ranges(minor_missing)}")

    # ---- orphan fragments (mechanical: length only) ---------------------------
    for s in segs:
        if len(s["text"]) < FRAGMENT_CHARS and not _is_heading_line(s["text"]):
            add("soft", "fragment_segment", f"{len(s['text'])}-char segment "
                f"{s['text'][:40]!r}; should merge with a neighbour", s["labels"])

    # ---- capsules ---------------------------------------------------------------
    role_of = {cid: s["role"] for s in segs for cid in s["chunk_ids"]}
    text_of = {cid: t for s in segs for cid, t in zip(s["chunk_ids"], s["texts"])}
    doc_up = norm_upper(" ".join(paragraphs))
    majority = [c for c in capsules if not is_separate_opinion(c.get("opinion_type"))]
    if segs and capsules and not majority:
        add("hard", "no_majority_capsule", "every capsule is a separate opinion")
    for c in capsules:
        ids = c.get("supporting_chunk_ids") or []
        roles = {role_of.get(cid) for cid in ids}
        sep = is_separate_opinion(c.get("opinion_type"))
        if sep and not (roles & DISSENT_ROLES):
            add("hard", "separate_opinion_unsupported",
                f"{c.get('opinion_type')} capsule rests on {sorted(r for r in roles if r)}, "
                f"no dissent segment")
        if not sep and (roles & DISSENT_ROLES):
            add("hard", "majority_on_dissent",
                f"majority capsule ({c.get('outcome')}) cites dissent segments")
        summary_up = norm_upper((c.get("reasoning_summary") or "") + " " +
                                (c.get("conclusion_sentence") or ""))
        support_up = norm_upper(" ".join(text_of.get(cid, "") for cid in ids))
        doc_nums, sup_nums = _stated_numbers(doc_up), _stated_numbers(support_up)
        for tok in sorted(_stated_numbers(summary_up)):
            if tok not in doc_nums:
                # law (k) and case (e) numbers are hard: an invented citation.
                # article numbers (m) are soft: Turkish article phrasing varies.
                add("hard" if tok[0] in "ke" else "soft", "capsule_number_not_in_document",
                    f"{c.get('opinion_type')} capsule states {tok!r} (m=article, k=law, "
                    f"e=case no), absent from the decision")
            elif tok not in sup_nums:
                add("soft", "capsule_number_not_in_support",
                    f"{tok!r} in summary is not in the supporting chunks")
        outcome = legacy_outcome(c.get("outcome"))
        if outcome is None:
            add("hard", "outcome_not_in_vocabulary",
                f"{c.get('opinion_type')} capsule outcome {c.get('outcome')!r} is not a disposition")
        elif outcome == "other":
            add("soft", "outcome_other", f"{c.get('opinion_type')} capsule outcome is 'other'")
        if outcome and not sep and record is not None and source == "aym":
            item = _aym_metadata_check(dict(c, outcome=outcome), record, archetype)
            if item:
                # The court's own metadata disagrees. Informative: the audit
                # call sees the same metadata and decides.
                add("soft", "outcome_vs_metadata", item[2])
    return {"hard": hard, "soft": soft}


def _ranges(idxs):
    out, start, prev = [], None, None
    for i in idxs:
        if start is None:
            start = prev = i
        elif i == prev + 1:
            prev = i
        else:
            out.append(f"p{start}" if start == prev else f"p{start}-p{prev}")
            start = prev = i
    if start is not None:
        out.append(f"p{start}" if start == prev else f"p{start}-p{prev}")
    return ", ".join(out[:10]) + (" ..." if len(out) > 10 else "")


def _aym_metadata_check(cap, record, archetype):
    """AYM metadata states the outcome per right / per reviewed article. The
    strongest independent signal there is, and previously unused."""
    data = parse_metadata(record).get("data") or {}
    outcome = cap.get("outcome")
    if archetype == "aym_individual_application":
        want = set()
        for e in data.get("examination_results") or []:
            if slug_right(e.get("right") or "") == cap.get("subject_id"):
                mapped = AYM_INDIVIDUAL_OUTCOME.get((e.get("outcome") or "").strip())
                if mapped:
                    want.add(mapped)
        if want and outcome not in want and outcome != "other":
            return ("hard", "outcome_contradicts_metadata",
                    f"{cap.get('subject_id')}: outcome {outcome!r}, metadata says {sorted(want)}")
        return None
    if archetype == "aym_norm_review":
        arts = set(re.findall(r"(?:^|_)m(\d+)", cap.get("subject_id") or ""))
        want = set()
        for norm in data.get("examined_norms") or []:
            for a in norm.get("articles") or []:
                if arts and (a.get("article_no") or "").strip() in arts:
                    want |= AYM_NORM_RESULT_ACCEPT.get((a.get("review_type_result") or "").strip(), set())
        if want and outcome not in want and outcome != "other":
            return ("hard", "outcome_contradicts_metadata",
                    f"{cap.get('subject_id')}: outcome {outcome!r}, metadata says {sorted(want)}")
    return None


def as_flag(item):
    return f"gate:{item['kind']}:{item['detail']}"


# =============================================================================
# audit_document -- the SECOND READ. Given the paragraphs, the first pass's
# segments (range + role) and capsules, the model returns only what it would
# change: role fixes, outcome fixes, dispositions with no capsule. Code applies
# the fixes mechanically (roles and outcomes still come from the Literals) and
# records every change as a flag. This is where "is this header really facts?"
# and "does this outcome match the ruling?" are decided -- by reading, not by
# a phrase table. One extra call per document, input-heavy, output tiny.
# =============================================================================


def _aym_metadata_hint(record, archetype):
    """The court's own structured verdicts, as text for the audit prompt."""
    data = parse_metadata(record).get("data") or {}
    lines = []
    if archetype == "aym_individual_application":
        for e in data.get("examination_results") or []:
            if e.get("right"):
                lines.append(f"  - {slug_right(e['right'])}: {e.get('outcome')} "
                             f"(-> {AYM_INDIVIDUAL_OUTCOME.get((e.get('outcome') or '').strip(), '?')})")
    elif archetype == "aym_norm_review":
        for norm in data.get("examined_norms") or []:
            for a in norm.get("articles") or []:
                res = (a.get("review_type_result") or "").strip()
                lines.append(f"  - {norm.get('norm_code_name')} m.{a.get('article_no')}: {res} "
                             f"(-> {'/'.join(sorted(AYM_NORM_RESULT_ACCEPT.get(res, set()))) or '?'})")
    return "\n".join(lines) if lines else None


def audit_document(client, model, paragraphs, segments, capsules, source, archetype,
                   record=None, stats=None):
    roles = tuple(prompts.ROLE_VOCAB[source][1])
    outcomes = tuple(OUTCOME_BY_ARCHETYPE.get(archetype) or OUTCOME_BY_ARCHETYPE[source])
    RoleFix = create_model("RoleFix", local_id=(str, ...), role=(Literal[roles], ...),
                           reason=(str, Field(min_length=5)))
    OutcomeFix = create_model("OutcomeFix", capsule_index=(int, ...),
                              outcome=(Literal[outcomes], ...), reason=(str, Field(min_length=5)))
    Missing = create_model("MissingCapsule", outcome=(Literal[outcomes], ...),
                           subject=(str, ...), reason=(str, Field(min_length=5)))
    Audit = create_model(f"Audit_{archetype}",
                         role_fixes=(List[RoleFix], Field(default_factory=list)),
                         outcome_fixes=(List[OutcomeFix], Field(default_factory=list)),
                         missing_capsules=(List[Missing], Field(default_factory=list)))
    hint = _aym_metadata_hint(record, archetype) if (record is not None and source == "aym") else None
    system = prompts.build_audit_instruction(source, archetype, roles, outcomes)
    content = prompts.build_audit_content(paragraphs, segments, capsules, hint)
    resp = call_gemini(client, model, system, content, Audit, max_tokens=8192)
    if stats is not None:
        stats["input_tokens"] += getattr(resp.usage_metadata, "prompt_token_count", 0) or 0
        stats["output_tokens"] += getattr(resp.usage_metadata, "candidates_token_count", 0) or 0
    if finish_reason_of(resp) == "MAX_TOKENS":
        raise ValueError("audit response truncated")
    return Audit.model_validate_json(resp.text)


def apply_audit(segments, capsules, verdict):
    """Apply the audit's fixes to pydantic segment/capsule objects. Returns
    (segments, capsules, flags). A fix naming an unknown segment or index is
    recorded and ignored -- never invented."""
    flags = []
    by_id = {seg.local_id: k for k, seg in enumerate(segments)}
    segments = list(segments)
    for fix in verdict.role_fixes:
        k = by_id.get(fix.local_id)
        if k is None:
            flags.append(f"audit_unknown_segment:{fix.local_id}")
            continue
        old = segments[k].role
        if old != fix.role:
            segments[k] = segments[k].model_copy(update={"role": fix.role})
            flags.append(f"audited_role:{fix.local_id}:{old}->{fix.role}:{fix.reason[:70]}")
    capsules = list(capsules)
    for fix in verdict.outcome_fixes:
        if not 0 <= fix.capsule_index < len(capsules):
            flags.append(f"audit_unknown_capsule:{fix.capsule_index}")
            continue
        old = capsules[fix.capsule_index].outcome
        if old != fix.outcome:
            capsules[fix.capsule_index] = capsules[fix.capsule_index].model_copy(
                update={"outcome": fix.outcome})
            flags.append(f"audited_outcome:[{fix.capsule_index}]:{old}->{fix.outcome}:{fix.reason[:70]}")
    for m in verdict.missing_capsules:
        flags.append(f"audit_missing_capsule:{m.outcome}:{slug(m.subject)}:{m.reason[:70]}")
    if not flags:
        flags.append("audit_agreed")
    return segments, capsules, flags


def extraction_coverage(record, source, paragraphs):
    """How much of the raw record's text the extracted paragraphs carry.

    Measured against WHICHEVER column holds more text, so it catches both a
    <p>-only walk that drops <li> blocks (kvkk: 0.11-0.20 before the fix) and a
    configured column that is nearly empty while the other holds the decision
    (the yargitay case). Below 0.6 the document is refused before any call.
    """
    from html import unescape
    from bs4 import BeautifulSoup

    def plain(s):
        s = s or ""
        if "<" in s and ">" in s:
            return norm_ws(BeautifulSoup(unescape(s), "html.parser").get_text(" "))
        return norm_ws(s)
    full = max(len(plain(record.get("html_content"))), len(plain(record.get("content_text"))))
    got = len(norm_ws(" ".join(paragraphs)))
    return round(got / full, 3) if full else 0.0


MIN_EXTRACTION_COVERAGE = 0.6


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
    # Input completeness BEFORE paying: if the paragraphs we extracted carry less
    # than 60% of the record's text, the model would chunk a fragment and every
    # capsule would be ungrounded. This is the check that would have caught the
    # kvkk <p>-only extractor (0.11-0.20 coverage) on day one.
    coverage = extraction_coverage(record, source, paragraphs)
    if coverage < MIN_EXTRACTION_COVERAGE:
        return None, None, {"doc_id": doc_id, "case_no": case_no,
                            "reason": "input_incomplete",
                            "error": f"extracted paragraphs carry {coverage:.0%} of the "
                                     f"record's text (minimum {MIN_EXTRACTION_COVERAGE:.0%}); "
                                     f"fix the extractor, not the prompt. No tokens spent.",
                            "failed_checks": [], "raw_response": None}, None

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

    archetype = archetype_of(record, source)
    schema = response_model_for(source, archetype, cand)
    subj_kind, subj_value = candidate_subject(record, source, paragraphs)
    subject_type = resolve_subject_type(record, source)
    decision_date, date_flag = compute_decision_date(record, source, paragraphs)
    # How many of the headings this source's note names are actually present. A
    # document with almost none gets function-based guidance instead, because the
    # note is then describing a layout it does not have -- true of 43 documents in
    # a year-spread export, across two different courts.
    prompt_kwargs = dict(
        subject_hint=(subj_kind, subj_value) if subj_kind else None,
        examined=examined_norms(record) if source == "aym" else None,
        markers_found=count_markers(paragraphs, source),
        archetype=archetype)
    user_content = prompts.build_user_content(paragraphs)

    def generate(system_instruction):
        """One API exchange with the existing perturbed-retry policy. Returns
        (parsed, raw_text, perturbed) or raises with the failure recorded."""
        raw_text, finish, last_err = None, None, None
        for attempt in (1, 2):
            # Attempt 2 perturbs on purpose: an identical temperature-0 retry cannot
            # produce a different result, so the old "retry once" was a no-op for
            # every deterministic failure.
            perturb = attempt == 2
            # A looping model fills whatever budget it is given, so the retry is
            # given a smaller one: enough headroom for a real answer (three times
            # the predicted need, with a floor), but not another full 65,535
            # tokens spent discovering the same loop.
            retry_cap = min(MAX_OUTPUT_TOKENS, max(8192, int(predicted * 3)))
            try:
                resp = call_gemini(client, model, system_instruction, user_content,
                                   schema, perturb=perturb,
                                   max_tokens=retry_cap if perturb else None)
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
                # An illegal enum value (outcome, opinion_type, legislation_type,
                # role, rights) raises HERE: pydantic re-validates what Gemini's
                # response_schema should already have enforced, so a bad value
                # takes the retry path instead of reaching storage.
                return schema.model_validate_json(raw_text), raw_text, perturb
            except Exception as e:               # noqa: BLE001 - any failure retries once
                last_err = f"{type(e).__name__}: {e}"
                if attempt == 2:
                    reason = ("max_output_tokens_truncated" if finish == "MAX_TOKENS"
                              else "api_or_parse_failed")
                    raise GenerationFailed({"doc_id": doc_id, "case_no": case_no,
                                            "reason": reason, "error": last_err,
                                            "finish_reason": finish, "failed_checks": [],
                                            "raw_response": raw_text})

    def audited(parsed):
        """Second read. Repairs first (dedupe, fragment merge), then the model
        reviews roles and outcomes and code applies what it returns."""
        segments, repair_flags = normalise_segments(parsed.segments, paragraphs)
        capsules = list(parsed.capsules)
        flags = []
        if client is not None:
            try:
                verdict = audit_document(client, model, paragraphs, segments, capsules,
                                         source, archetype, record=record, stats=stats)
                segments, capsules, flags = apply_audit(segments, capsules, verdict)
                stats["audits"] += 1
                stats["audit_role_fixes"] += sum(1 for f in flags if f.startswith("audited_role:"))
                stats["audit_outcome_fixes"] += sum(1 for f in flags if f.startswith("audited_outcome:"))
            except Exception as exc:                               # noqa: BLE001
                flags = [f"audit_failed:{type(exc).__name__}:{str(exc)[:80]}"]
        return parsed.model_copy(update={"segments": segments, "capsules": capsules}), flags

    def post_process(parsed):
        chunks, id_map, flags, minor_diffs = build_chunks(
            parsed, record, source, doc_id, case_no, paragraphs, decision_date)
        if date_flag:
            flags.append(date_flag)
        stats["minor_text_diffs"] += len(minor_diffs)
        capsules, cap_flags = build_capsules(parsed, chunks, id_map, source, case_no, cand,
                                             decision_date, subject_type)
        flags += cap_flags
        # MECHANICS: coverage, overlap, fragments, support consistency, numbers.
        gate = verify_document(chunks, capsules, paragraphs, source, archetype,
                               record=record)
        rejects = [f for f in flags if f.split(":")[0] in REJECT_FLAGS]
        rejects += [as_flag(f) for f in gate["hard"]]
        return chunks, capsules, flags, gate, rejects

    try:
        system_instruction = prompts.build_system_instruction(source, case_no, cand,
                                                              **prompt_kwargs)
        parsed, raw_text, perturbed = generate(system_instruction)
        parsed, audit_flags = audited(parsed)
        chunks, capsules, flags, gate, rejects = post_process(parsed)
        flags += audit_flags
        corrected = False
        if rejects:
            # ONE corrections round. The retry is not a blind re-send: the exact
            # violations are appended to the instruction, so a temperature-0
            # call sees a different prompt and can give a different answer.
            # Measured need: header-as-facts on 50 of 67 aym documents, 40
            # documents with paragraphs in no segment -- both are things the
            # model fixes when told precisely which segment is wrong.
            stats["corrections"] += 1
            system_instruction = prompts.build_system_instruction(
                source, case_no, cand, corrections=rejects, **prompt_kwargs)
            parsed, raw_text, perturbed = generate(system_instruction)
            parsed, audit_flags = audited(parsed)
            chunks, capsules, flags, gate, rejects = post_process(parsed)
            flags += audit_flags
            corrected = True
    except GenerationFailed as gf:
        return None, None, gf.review, None

    if perturbed:
        stats["perturbed_retries"] += 1
        print(f"  [{source}] {doc_id}  NOTE: needed a perturbed retry "
              f"(temperature 0.4 + frequency penalty) -- this document's output is "
              f"NOT byte-reproducible")
    if corrected:
        print(f"  [{source}] {doc_id}  NOTE: needed a corrections round "
              f"({len(rejects)} item(s) remain)" if rejects else
              f"  [{source}] {doc_id}  corrections round resolved every reject item")

    disagreement = legislation_crosscheck(chunks)
    for ch in chunks:
        ch.pop("_role", None)

    review = None
    if flags or gate["hard"] or gate["soft"]:
        review = {"doc_id": doc_id, "case_no": case_no, "archetype": archetype,
                  "reason": "rejected" if rejects else "validation_flags",
                  "failed_checks": flags, "reject_items": rejects,
                  "gate": {"hard": gate["hard"], "soft": gate["soft"]},
                  "legislation_disagreement": disagreement,
                  "raw_response": json.loads(raw_text) if raw_text else None}
    if rejects:
        # A bad chunk is IMPOSSIBLE to store: the document goes to the rejected
        # file with every remaining item named, never to {source}.json.
        review["rejected_chunks"] = chunks
        review["rejected_capsules"] = capsules
        return [], [], review, disagreement
    return chunks, capsules, review, disagreement


class GenerationFailed(Exception):
    def __init__(self, review):
        super().__init__(review.get("error"))
        self.review = review


# Flags that make a document UNSTORABLE. Everything else stays advisory. The
# content gate's hard items are added to this set at run time (they carry the
# prefix "gate:").
# Advisory flags that describe the model's citation copying or an internal
# confidence label, never the chunk text. Counted per source and kept in the
# review sidecar, but they do not make a document "flagged" on the console:
# 37 of 50 documents printed as flagged on one run, almost all for these.
NOISE_FLAGS = {
    "verbatim_mention_reconstructed", "verbatim_mention_not_in_text",
    "confidence_contradiction", "abbreviation_without_law_no",
    "citation_not_identifiable", "dropped_non_legislation", "dropped_treaty",
    "fragment_merged", "duplicate_ref_removed", "segment_emptied_by_dedupe",
    "audit_agreed",
}

REJECT_FLAGS = {
    "segment_has_no_valid_refs", "invalid_paragraph_ref",
    "capsule_has_no_resolvable_support", "unresolved_local_id",
    "reasoning_summary_not_turkish", "conclusion_sentence_not_turkish",
    "reasoning_summary_empty", "conclusion_sentence_empty",
    "opinion_type_support_mismatch", "subject_id_unspecified",
    "source_type_mismatch",
}


def run(sources, limit, only_doc_id=None, out_dir=None, force_red=False):
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

    grand = {"ok": 0, "flagged": 0, "failed": 0, "rejected": 0}
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

        # Preflight is ENFORCED here, not merely reported: a RED document (no
        # text, one-paragraph split, coverage below 0.6, predicted truncation)
        # would be paid for and produce nothing usable. Imported lazily because
        # preflight imports this module.
        picked_red = []
        if not force_red:
            import preflight
            kept = []
            for rec in picked:
                verdict = preflight.check_document(rec, source)
                if verdict["verdict"] == "RED":
                    reviews_red = {"doc_id": rec.get("doc_id"),
                                   "case_no": verdict["info"].get("case_no"),
                                   "reason": "preflight_red",
                                   "failed_checks": verdict["problems"]}
                    print(f"  [{source}] {rec.get('doc_id')}  SKIPPED (preflight RED): "
                          f"{verdict['problems'][0][:90]}")
                    picked_red.append(reviews_red)
                else:
                    kept.append(rec)
            picked = kept

        all_chunks, all_caps, reviews, rejected = [], [], list(picked_red), []
        stats = {"input_tokens": 0, "output_tokens": 0, "minor_text_diffs": 0,
                 "perturbed_retries": 0, "corrections": 0, "audits": 0,
                 "audit_role_fixes": 0, "audit_outcome_fixes": 0}
        noise = Counter()
        only_g, only_r, ok, flagged, failed = 0, 0, 0, 0, 0

        out_path = out_root / f"{source}.json"
        review_path = out_root / f"{source}_review.json"
        rejected_path = out_root / f"{source}_rejected.json"
        processed = []

        def flush():
            """Write the source's three files NOW. Called after every document so
            a run stopped at minute 10 keeps the documents it already paid for.
            Idempotent: a targeted run merges by doc_id, so re-writing after each
            document replaces that document's own earlier entries only."""
            chunks_out, caps_out, reviews_out, rejected_out = (
                all_chunks, all_caps, reviews, rejected)
            if only_doc_id:
                # TARGETED run: merge into what is already on disk. The documents
                # just run replace their own earlier entries (stored, review or
                # rejected); every other document is kept. Without this, re-running
                # one rejected document would wipe the rest of its source file.
                ran_ids = set(processed) | {str(r["doc_id"]) for r in picked_red}
                chunks_out, caps_out = _merge_stored(out_path, all_chunks, all_caps, ran_ids)
                reviews_out = _merge_sidecar(review_path, reviews, ran_ids)
                rejected_out = _merge_sidecar(rejected_path, rejected, ran_ids)
            if not chunks_out and out_path.is_file():
                # Never replace a good output file with an empty one.
                return
            out_path.write_text(
                json.dumps({"chunks": chunks_out, "reasoning_capsules": caps_out},
                           ensure_ascii=False, indent=2), encoding="utf-8")
            if reviews_out:
                review_path.write_text(
                    json.dumps(reviews_out, ensure_ascii=False, indent=2), encoding="utf-8")
            elif review_path.is_file():
                review_path.unlink()
            if rejected_out:
                rejected_path.write_text(
                    json.dumps(rejected_out, ensure_ascii=False, indent=2), encoding="utf-8")
            elif rejected_path.is_file():
                rejected_path.unlink()

        for rec in picked:
            chunks, capsules, review, disagreement = process_document(
                client, model, rec, source, stats)
            if chunks is None:                   # hard failure before post-processing
                reviews.append(review)
                failed += 1
                print(f"  [{source}] {rec.get('doc_id')}  FAILED: {review.get('reason')}")
                processed.append(str(rec.get("doc_id")))
                flush()
                continue
            if review and review.get("reason") == "rejected":
                rejected.append(review)
                print(f"  [{source}] {rec.get('doc_id')}  REJECTED: "
                      f"{', '.join(review['reject_items'][:4])}")
                processed.append(str(rec.get("doc_id")))
                flush()
                continue
            all_chunks.extend(chunks)
            all_caps.extend(capsules)
            only_g += len(disagreement["only_gemini"])
            only_r += len(disagreement["only_regex"])
            if review:
                reviews.append(review)
                for f in review["failed_checks"]:
                    noise[f.split(":")[0]] += 1
                loud = ([f for f in review["failed_checks"] if f.split(":")[0] not in NOISE_FLAGS]
                        + [as_flag(g) for g in review["gate"]["soft"]])
                if loud:
                    flagged += 1
                    print(f"  [{source}] {rec.get('doc_id')}  flagged: {', '.join(loud[:4])}")
                else:
                    ok += 1
            else:
                ok += 1
            print(f"  [{source}] {rec.get('doc_id')}  {len(chunks)} chunks, "
                  f"{len(capsules)} capsules")
            processed.append(str(rec.get("doc_id")))
            flush()

        # Never replace a good output file with an empty one. When every
        # document in a source fails, the old run is the better artifact and
        # silently clobbering it loses real work -- which is exactly what
        # happened to aym.json once.
        flush()

        grand["ok"] += ok
        grand["flagged"] += flagged
        grand["failed"] += failed
        grand["rejected"] += len(rejected)
        print(f"=== {source} ===  {len(picked)} docs | ok {ok} | flagged {flagged} | "
              f"rejected {len(rejected)} | failed {failed} | skipped_empty {skipped_empty} | "
              + (f"reserved_as_example {skipped_fewshot} | " if skipped_fewshot else "")
              + (f"preflight_red {len(picked_red)} | " if picked_red else "")
              + f"{len(all_chunks)} chunks | {len(all_caps)} capsules")
        if stats["corrections"]:
            print(f"  corrections rounds: {stats['corrections']}")
        if stats["audits"]:
            print(f"  audit (second read): {stats['audits']} documents, "
                  f"{stats['audit_role_fixes']} role fixes, "
                  f"{stats['audit_outcome_fixes']} outcome fixes")
        if noise:
            print("  advisory (counted, in the review file, not chunk errors): "
                  + ", ".join(f"{k} {v}" for k, v in noise.most_common()))
        print(f"  legislation: gemini-only {only_g}, regex-only {only_r}")
        print(f"  minor text diffs (model copy vs source join; stored text unaffected): "
              f"{stats['minor_text_diffs']}")
        if stats["perturbed_retries"]:
            print(f"  perturbed retries: {stats['perturbed_retries']} "
                  f"(those documents are NOT byte-reproducible)")
        print(f"  tokens: in {stats['input_tokens']}, out {stats['output_tokens']}\n")

    print(f"TOTAL: ok {grand['ok']} | flagged {grand['flagged']} | "
          f"rejected {grand['rejected']} | failed {grand['failed']}")
    print(f"Output: {out_root}")


def _doc_of_chunk(chunk):
    return chunk["chunk_label"].split("-p", 1)[0]


def _merge_stored(out_path, new_chunks, new_caps, ran_ids):
    """Existing {source}.json minus the documents just run, plus the new output."""
    if not out_path.is_file():
        return new_chunks, new_caps
    old = json.loads(out_path.read_text(encoding="utf-8"))
    keep_chunks = [c for c in old.get("chunks", []) if _doc_of_chunk(c) not in ran_ids]
    keep_ids = {c["chunk_id"] for c in keep_chunks}
    keep_caps = [c for c in old.get("reasoning_capsules", [])
                 if any(cid in keep_ids for cid in c.get("supporting_chunk_ids") or [])]
    return keep_chunks + new_chunks, keep_caps + new_caps


def _merge_sidecar(path, new_entries, ran_ids):
    if not path.is_file():
        return new_entries
    old = json.loads(path.read_text(encoding="utf-8"))
    return [e for e in old if str(e.get("doc_id")) not in ran_ids] + new_entries


def _load_raw(sources):
    raw = {}
    for source in sources:
        path = DATA_DIR / f"{source}.json"
        if path.is_file():
            for r in json.loads(path.read_text(encoding="utf-8")):
                raw[(source, str(r.get("doc_id")))] = r
    return raw


def verify_output(out_dir):
    """`--verify DIR`: judge every stored document offline. No API call, no
    write except DIR/verify_report.json. Exit 1 on any hard item -- this is the
    proof that the check can fail a run."""
    base = Path(out_dir)
    if not base.is_absolute():
        base = ROOT / base
    raw = _load_raw(SOURCES)
    report, n_docs, n_fail, kinds = [], 0, 0, {}
    for path in sorted(base.glob("*.json")):
        if path.stem.endswith(("_review", "_rejected", "_report")) or path.stem == "manifest":
            continue
        source = path.stem
        if source not in SOURCES:
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        by_doc = {}
        for ch in data.get("chunks", []):
            by_doc.setdefault(ch["chunk_label"].split("-p", 1)[0], {"chunks": [], "capsules": []})
            by_doc[ch["chunk_label"].split("-p", 1)[0]]["chunks"].append(ch)
        chunk_doc = {ch["chunk_id"]: ch["chunk_label"].split("-p", 1)[0]
                     for ch in data.get("chunks", [])}
        for cap in data.get("reasoning_capsules", []):
            docs = {chunk_doc.get(cid) for cid in cap.get("supporting_chunk_ids") or []} - {None}
            for d in docs:
                by_doc[d]["capsules"].append(cap)
        for doc_id, d in by_doc.items():
            record = raw.get((source, doc_id))
            if record is None:
                report.append({"source": source, "doc_id": doc_id, "verdict": "NO_RAW"})
                continue
            paragraphs = extract_paragraphs(record, source)
            archetype = archetype_of(record, source)
            gate = verify_document(d["chunks"], d["capsules"], paragraphs, source,
                                   archetype, record=record)
            n_docs += 1
            verdict = "FAIL" if gate["hard"] else ("WARN" if gate["soft"] else "PASS")
            n_fail += verdict == "FAIL"
            for it in gate["hard"]:
                kinds.setdefault(("hard", it["kind"]), set()).add(doc_id)
            for it in gate["soft"]:
                kinds.setdefault(("soft", it["kind"]), set()).add(doc_id)
            report.append({"source": source, "doc_id": doc_id,
                           "case_no": (d["chunks"][0].get("case_no") if d["chunks"] else None),
                           "archetype": archetype, "verdict": verdict,
                           "paragraphs": len(paragraphs), "chunks": len(d["chunks"]),
                           "hard": gate["hard"], "soft": gate["soft"]})
    print("=" * 76)
    print(f" VERIFY -- {base}")
    print("=" * 76)
    print(f" documents {n_docs} | FAIL {n_fail} | "
          f"WARN {sum(1 for r in report if r['verdict'] == 'WARN')} | "
          f"PASS {sum(1 for r in report if r['verdict'] == 'PASS')}\n")
    for (sev, kind), docs in sorted(kinds.items(), key=lambda kv: (kv[0][0], -len(kv[1]))):
        print(f"   {len(docs):>4} docs  [{sev:4}] {kind}")
    print()
    for r in report:
        if r["verdict"] == "FAIL":
            print(f" FAIL {r['source']:13} {r['doc_id'][:14]:14} {str(r['case_no']):12} {r['archetype']}")
            for it in r["hard"][:6]:
                print(f"        {it['kind']}: {it['detail'][:110]}")
    out = base / "verify_report.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n wrote {out}")
    return 1 if n_fail else 0


def audit_output(out_dir):
    """`--audit DIR`: the second read over STORED output. Paid (one call per
    document, input-heavy). Applies role/outcome fixes in place, re-derives
    content_type/reasoning_stage, writes DIR/audit_report.json."""
    from types import SimpleNamespace
    load_dotenv(ROOT / ".env")
    model = os.getenv("MODEL", "gemini-2.5-flash-lite")
    client = build_client()
    base = Path(out_dir)
    if not base.is_absolute():
        base = ROOT / base
    raw = _load_raw(SOURCES)
    report, stats = [], {"input_tokens": 0, "output_tokens": 0}
    for path in sorted(base.glob("*.json")):
        source = path.stem
        if source not in SOURCES:
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        chunks, caps = data.get("chunks", []), data.get("reasoning_capsules", [])
        by_doc = {}
        for ch in chunks:
            by_doc.setdefault(_doc_of_chunk(ch), []).append(ch)
        chunk_doc = {ch["chunk_id"]: _doc_of_chunk(ch) for ch in chunks}
        changed = 0
        for doc_id, dchunks in by_doc.items():
            record = raw.get((source, doc_id))
            if record is None:
                continue
            paragraphs = extract_paragraphs(record, source)
            archetype = archetype_of(record, source)
            segs = _segments_of(dchunks)
            seg_objs = [SimpleNamespace(local_id=re.sub(r"_p\d+$", "", s_["labels"][0]),
                                        paragraph_refs=[f"p{i}" for i in s_["refs"]],
                                        role=s_["role"]) for s_ in segs]
            dcaps = [c for c in caps if any(chunk_doc.get(cid) == doc_id
                                            for cid in c.get("supporting_chunk_ids") or [])]
            cap_objs = [SimpleNamespace(opinion_type=c["opinion_type"], outcome=c["outcome"],
                                        subject_id=c["subject_id"],
                                        conclusion_sentence=c["conclusion_sentence"]) for c in dcaps]
            try:
                verdict = audit_document(client, model, paragraphs, seg_objs, cap_objs,
                                         source, archetype, record=record, stats=stats)
            except Exception as exc:                               # noqa: BLE001
                report.append({"source": source, "doc_id": doc_id, "error": str(exc)[:200]})
                print(f"  [{source}] {doc_id}  audit FAILED: {str(exc)[:100]}")
                continue
            fixes = []
            role_field = SOURCES[source]["role_field"]
            for fix in verdict.role_fixes:
                for ch in dchunks:
                    if re.sub(r"_p\d+$", "", ch["chunk_label"]) == fix.local_id and ch["role"] != fix.role:
                        fixes.append(f"role {ch['chunk_label']}: {ch['role']}->{fix.role} ({fix.reason[:60]})")
                        ch["role"] = fix.role
                        if role_field == "firac_role":
                            ch["firac_role"] = fix.role
                        else:
                            ch[role_field] = fix.role
                        ch["content_type"], ch["reasoning_stage"] = derive(fix.role)
            for fix in verdict.outcome_fixes:
                if 0 <= fix.capsule_index < len(dcaps) and dcaps[fix.capsule_index]["outcome"] != fix.outcome:
                    fixes.append(f"outcome [{fix.capsule_index}]: {dcaps[fix.capsule_index]['outcome']}"
                                 f"->{fix.outcome} ({fix.reason[:60]})")
                    dcaps[fix.capsule_index]["outcome"] = fix.outcome
            for m in verdict.missing_capsules:
                fixes.append(f"missing capsule: {m.outcome} {slug(m.subject)} ({m.reason[:60]})")
            changed += bool(fixes)
            report.append({"source": source, "doc_id": doc_id, "archetype": archetype, "fixes": fixes})
            print(f"  [{source}] {doc_id}  " + ("; ".join(fixes[:3]) if fixes else "audit agreed"))
        path.write_text(json.dumps({"chunks": chunks, "reasoning_capsules": caps},
                                   ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"=== {source} === {len(by_doc)} docs audited, {changed} changed")
    print(f"tokens: in {stats['input_tokens']}, out {stats['output_tokens']}")
    out = base / "audit_report.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {out}")
    return 0


def check_extraction(sources):
    """Step 1 of the plan: for every usable record in data/, how much of the raw
    text do our paragraphs carry? No LLM call. Writes output/extraction_report.json."""
    rows = []
    for source in sources:
        path = DATA_DIR / f"{source}.json"
        if not path.is_file() or SOURCES[source]["text_field"] is None:
            continue
        for r in json.loads(path.read_text(encoding="utf-8")):
            if r.get("status") not in USABLE_STATUSES:
                continue
            if not ((r.get("html_content") or "").strip() or (r.get("content_text") or "").strip()):
                continue
            paras = extract_paragraphs(r, source)
            cov = extraction_coverage(r, source, paras)
            rows.append({"source": source, "doc_id": str(r.get("doc_id")),
                         "paragraphs": len(paras), "coverage": cov,
                         "chars": sum(len(p) for p in paras)})
    print("=" * 76)
    print(" EXTRACTION -- do our paragraphs carry the whole document?")
    print("=" * 76)
    print(f"{'source':14} {'docs':>5} {'median ¶':>9} {'median cov':>11} {'<0.85':>6} {'<0.60':>6}")
    for source in sources:
        rs = [x for x in rows if x["source"] == source]
        if not rs:
            continue
        covs = sorted(x["coverage"] for x in rs)
        paras = sorted(x["paragraphs"] for x in rs)
        print(f"{source:14} {len(rs):>5} {paras[len(paras) // 2]:>9} "
              f"{covs[len(covs) // 2]:>11.3f} "
              f"{sum(1 for c in covs if c < 0.85):>6} {sum(1 for c in covs if c < 0.6):>6}")
    low = sorted((x for x in rows if x["coverage"] < 0.85), key=lambda x: x["coverage"])
    if low:
        print(f"\n below 0.85 ({len(low)}):")
        for x in low[:25]:
            print(f"   {x['source']:13} {x['doc_id'][:20]:20} cov {x['coverage']:.3f} "
                  f"paras {x['paragraphs']:>4} chars {x['chars']:>7,}")
    out = ROOT / "output" / "extraction_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n wrote {out}")
    return 1 if any(x["coverage"] < MIN_EXTRACTION_COVERAGE for x in rows) else 0


def main():
    ap = argparse.ArgumentParser(description="LLM chunking via Gemini 2.5 Flash-Lite")
    ap.add_argument("--source", choices=sorted(SOURCES), help="only this source")
    ap.add_argument("--limit", type=int, help="documents per source (overrides .env)")
    ap.add_argument("--out-dir", help="write output here instead of output/chunk/. Use for "
                                      "probe runs -- writing a different document set into "
                                      "output/chunk/ invalidates the retrieval queries")
    ap.add_argument("--doc-id-file", help="JSON file mapping source -> [doc_id, ...]. "
                                          "Runs exactly those documents, per source. "
                                          "Written by preflight so a vetted run set "
                                          "does not have to be retyped as a "
                                          "thousand-character command line.")
    ap.add_argument("--doc-id", help="generate exactly these documents (comma-separated), "
                                     "even if reserved in FEWSHOT_DOC_IDS. Used to build a "
                                     "worked example, or to choose a test set deliberately "
                                     "rather than taking whichever documents come first")
    ap.add_argument("--force-red", action="store_true",
                    help="send documents preflight marks RED anyway (default: skip them)")
    ap.add_argument("--verify", metavar="DIR",
                    help="no API call: judge every stored document in DIR (coverage, "
                         "roles, capsules) and exit 1 on any hard item")
    ap.add_argument("--audit", metavar="DIR",
                    help="PAID: second-read audit over every stored document in DIR; "
                         "applies role/outcome fixes in place")
    ap.add_argument("--check-extraction", action="store_true",
                    help="no API call: measure how much of every raw record's text the "
                         "extracted paragraphs carry, for all of data/")
    args = ap.parse_args()

    if args.verify:
        sys.exit(verify_output(args.verify))
    if args.audit:
        sys.exit(audit_output(args.audit))
    if args.check_extraction:
        sys.exit(check_extraction([args.source] if args.source else
                                  [s for s in SOURCES if s not in TEXT_PENDING_SOURCES]))

    if args.doc_id_file:
        # A vetted run set, produced by preflight. Each source runs with exactly
        # the documents listed for it, so the selection that was reviewed is the
        # selection that runs -- no retyping, no --limit picking whichever
        # documents happen to come first in the file.
        plan = json.loads(Path(args.doc_id_file).read_text(encoding="utf-8"))
        sources = [args.source] if args.source else [s for s in SOURCES if plan.get(s)]
        total = sum(len(plan.get(s) or []) for s in sources)
        print(f"run set  : {args.doc_id_file}  ({total} documents across "
              f"{len(sources)} sources)\n")
        for source in sources:
            ids = plan.get(source) or []
            if ids:
                run([source], args.limit, ",".join(str(i) for i in ids), args.out_dir,
                    force_red=args.force_red)
        return

    run([args.source] if args.source else list(SOURCES), args.limit, args.doc_id,
        args.out_dir, force_red=args.force_red)


if __name__ == "__main__":
    main()
