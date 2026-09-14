"""
corpus.py -- loads the generated chunks/capsules and the raw source documents,
and builds the join maps the verifier and the retrieval test both need.

JOIN KEYS (verified against the real output, do not guess these)
  - chunk["chunk_id"] is an opaque bare UUID5. It carries NO document
    information and cannot be parsed.
  - chunk["chunk_label"] is the join key: "{doc_id}-p{spec}". Splitting on the
    first "-p" is safe because doc_ids are either all-numeric or lowercase-hex
    UUIDs, and "p" is not a hex digit.
  - capsule -> document goes supporting_chunk_ids -> chunk_id -> chunk_label
    -> doc_id. Capsules carry no doc_id of their own.
  - Key on doc_id, NEVER case_no: case_no genuinely collides across different
    documents in the wider corpus (3 real first_degree cases), so a case_no key
    fans out silently as the corpus grows.

INDEPENDENCE NOTE
  This module imports `chunk_lib` -- the verbatim-copied, shared text helper --
  so that paragraph extraction matches what the pipeline actually fed the model.
  Using a different extraction here would test extraction, not storage. It does
  NOT import chunk_generate: none of the generator's validation logic is reused,
  and the *_review.json sidecars are never read.
"""

import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]          # llm_chunk/
DATA_DIR = ROOT / "data"
CHUNK_DIR = ROOT / "output" / "chunk"
REPORT_DIR = ROOT / "output" / "retrieval"
WORKSHEET_DIR = Path(__file__).resolve().parent / "worksheet"

sys.path.insert(0, str(ROOT / "scripts" / "llm_chunk"))
import chunk_lib as lib  # noqa: E402  (path set above; stays inside llm_chunk/)

SOURCES = ["aym", "bam", "danistay", "first_degree", "kvkk"]

# Which raw field each source's text comes from, and the extra role field its
# chunks carry. Mirrors the pipeline's own per-source table.
TEXT_FIELD = {"aym": "html_content", "kvkk": "html_content",
              "bam": "content_text", "danistay": "content_text",
              "first_degree": "content_text"}
ROLE_FIELD = {"aym": "firac_role", "bam": "court_reasoning_role",
              "danistay": "court_reasoning_role",
              "first_degree": "court_reasoning_role",
              "kvkk": "regulatory_role"}

CHUNK_BASE_FIELDS = [
    "chunk_id", "chunk_label", "source_type", "case_no", "decision_date",
    "citation_granularity", "source_paragraph_ids", "text", "char_length",
    "content_type", "firac_role", "reasoning_stage", "rights", "confidence",
    "cited_legislations",
]
CAPSULE_FIELDS = [
    "case_no", "source_type", "decision_date", "subject_type", "subject_id",
    "opinion_type", "outcome", "conclusion_sentence", "reasoning_summary",
    "reasoning_summary_method", "supporting_chunk_ids",
]
# Ten keys, always all present, null when absent (docs 14.2 null policy).
# law_short was added so a law cited only by abbreviation stays identifiable.
CITATION_FIELDS = [
    "canonical_id", "legislation_type", "law_no", "law_short", "law_name",
    "article_no", "paragraph_no", "verbatim_mention", "law_date", "confidence",
]
ISO_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

PARAGRAPH_RE = re.compile(r"^p\d+$")


def norm_ws(s):
    return " ".join((s or "").split())


def doc_id_of(chunk):
    """'676966100-p18_p1' -> '676966100'.  UUID doc_ids survive intact because
    no hex digit is 'p'."""
    label = chunk.get("chunk_label") or ""
    return label.split("-p", 1)[0] if "-p" in label else label


def extract_paragraphs(record, source):
    """The same view the pipeline fed the model: numbered paragraphs.

    For aym/kvkk this is html_content through the shared bs4 helper. For the
    other three it is content_text split on newlines -- those files contain no
    blank lines at all, so splitting on blank lines would return the whole
    document as one paragraph.
    """
    raw = record.get(TEXT_FIELD[source]) or ""
    if not raw.strip():
        return []
    if TEXT_FIELD[source] == "html_content":
        return lib.aym_extract_paragraph_texts(raw)
    return [p for p in (norm_ws(x) for x in raw.splitlines()) if p]


def load_raw_documents(wanted_doc_ids=None):
    """{(source, doc_id): {"record":..., "paragraphs":[...]}} for the documents
    we actually generated, read straight from data/*.json."""
    out = {}
    for source in SOURCES:
        path = DATA_DIR / f"{source}.json"
        if not path.is_file():
            continue
        for rec in json.loads(path.read_text(encoding="utf-8")):
            key = (source, rec.get("doc_id"))
            if wanted_doc_ids is not None and key not in wanted_doc_ids:
                continue
            out[key] = {"record": rec, "paragraphs": extract_paragraphs(rec, source)}
    return out


class Corpus:
    """The generated output plus every map the tests need."""

    def __init__(self):
        self.by_source = {}
        self.chunks, self.capsules = [], []
        for source in SOURCES:
            path = CHUNK_DIR / f"{source}.json"
            if not path.is_file():
                continue
            data = json.loads(path.read_text(encoding="utf-8"))
            self.by_source[source] = data
            self.chunks.extend(data.get("chunks", []))
            self.capsules.extend(data.get("reasoning_capsules", []))

        self.chunk_by_id = {c["chunk_id"]: c for c in self.chunks}
        self.doc_of_chunk = {c["chunk_id"]: doc_id_of(c) for c in self.chunks}

        # (source, doc_id) for every document represented in the output
        self.documents = []
        seen = set()
        for c in self.chunks:
            key = (c["source_type"], doc_id_of(c))
            if key not in seen:
                seen.add(key)
                self.documents.append(key)

        # (source, doc_id, "p18") -> [chunk_id, ...]; a paragraph can land in
        # several chunks when the 2000-char cap split its segment.
        self.paragraph_index = {}
        for c in self.chunks:
            key_doc = (c["source_type"], doc_id_of(c))
            for para in c.get("source_paragraph_ids") or []:
                self.paragraph_index.setdefault(key_doc + (para,), []).append(c["chunk_id"])

        self.case_no_of_doc = {}
        for c in self.chunks:
            self.case_no_of_doc.setdefault((c["source_type"], doc_id_of(c)), c.get("case_no"))

    def capsule_doc_ids(self, capsule):
        """Capsules carry no doc_id: resolve through their supporting chunks.
        Returns a set because nothing guarantees they all point at one document
        -- if they don't, that is itself a finding."""
        return {self.doc_of_chunk[cid]
                for cid in capsule.get("supporting_chunk_ids") or []
                if cid in self.doc_of_chunk}

    def capsule_text(self, capsule):
        """What Stage 1 searches: the capsule's own free text."""
        parts = [capsule.get("conclusion_sentence"), capsule.get("reasoning_summary"),
                 (capsule.get("subject_id") or "").replace("_", " ")]
        return " ".join(p for p in parts if p)

    def chunks_of_doc(self, source, doc_id):
        return [c for c in self.chunks
                if c["source_type"] == source and doc_id_of(c) == doc_id]

    def summary(self):
        return {
            "documents": len(self.documents),
            "chunks": len(self.chunks),
            "capsules": len(self.capsules),
            "per_source": {s: {"chunks": len(d.get("chunks", [])),
                               "capsules": len(d.get("reasoning_capsules", []))}
                           for s, d in self.by_source.items()},
        }
