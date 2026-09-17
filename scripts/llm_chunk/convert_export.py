"""Turn a database SQL export into the bare-list shape data/ uses, and merge it
into the existing file for that source.

Two JSON shapes exist in this project and both are real:

  * a bare ``list`` of rows              -- what ``data/<source>.json`` holds
  * ``{"<sql string>": [...rows]}``      -- what the export tool produces

``chunk_generate.run()`` does a plain ``json.loads()`` and indexes the result as
a list, so the wrapped shape silently fails there. Unwrapping happens here, once,
rather than being guessed at every call site.

MERGE, not replace. The exports are slices, not re-exports: the 50 rekabet rows
exported in September share **zero** doc_ids with the 200 already in data/, and
the same holds for uyusmazlik. Replacing would throw away 450 distinct decisions.

Rows are copied VERBATIM. In particular ``content_text`` keeps its ``""`` rather
than being rewritten to null: data/ is a faithful copy of the database, and the
""-vs-NULL distinction is handled in the pipeline where it belongs. (That
distinction is not academic -- filtering the export on ``content_text is not
null`` returns every empty row, because the column holds "" and not NULL.)

Usage:
    python convert_export.py --source rekabet --in ../../rekabet.json
    python convert_export.py --source rekabet --in ../../rekabet.json --dry-run
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]          # llm_chunk/
DATA_DIR = ROOT / "data"

# The column set every source in this corpus shares. Asserted on every row so a
# malformed or partial export is refused here rather than producing a KeyError
# deep inside the pipeline.
EXPECTED_KEYS = {
    "id", "doc_id", "court_type", "esas_year", "esas_no", "karar_year", "karar_no",
    "title", "content_text", "html_content", "subjects", "metadata", "status",
    "finalization_status", "failure_reason", "retry_count", "next_retry_at",
    "last_finalization_check_at", "next_finalization_check_at", "created_at",
    "updated_at", "chamber_id", "meili_status", "qdrant_status",
    "sync_failure_reason", "extraction_status",
}


def unwrap(obj, label):
    """Accept either shape; refuse anything ambiguous.

    A dict with more than one key is refused rather than resolved by picking the
    longest value or the first -- an export we do not understand should stop the
    run, not be guessed at.
    """
    if isinstance(obj, list):
        return obj, None
    if not isinstance(obj, dict):
        raise SystemExit(f"{label}: expected a list or a dict, got {type(obj).__name__}")
    if len(obj) != 1:
        raise SystemExit(
            f"{label}: SQL-wrapped export must have exactly one key, found {len(obj)}: "
            f"{sorted(obj)[:5]}")
    sql, rows = next(iter(obj.items()))
    if not isinstance(rows, list):
        raise SystemExit(f"{label}: key {sql!r} does not hold a list of rows")
    return rows, sql


def check_schema(rows, label):
    """Every row carries the full column set. Reports the first offender with the
    actual difference, because 'schema mismatch' alone is not actionable."""
    for i, r in enumerate(rows):
        if not isinstance(r, dict):
            raise SystemExit(f"{label}: row {i} is {type(r).__name__}, not an object")
        keys = set(r)
        if keys != EXPECTED_KEYS:
            raise SystemExit(
                f"{label}: row {i} (doc_id={r.get('doc_id')}) has an unexpected schema\n"
                f"  missing: {sorted(EXPECTED_KEYS - keys)}\n"
                f"  extra  : {sorted(keys - EXPECTED_KEYS)}")


def text_stats(rows):
    n_text = sum(1 for r in rows if (r.get("content_text") or "").strip())
    n_html = sum(1 for r in rows if (r.get("html_content") or "").strip())
    return n_text, n_html


def merge(existing, incoming):
    """Existing rows first, then incoming rows whose doc_id is new.

    Order matters: pick_documents() takes the first N usable records, so appending
    keeps the documents any earlier run selected at the front and stops a new
    export from silently changing which documents get chunked.
    """
    seen = {str(r.get("doc_id")) for r in existing}
    added = [r for r in incoming if str(r.get("doc_id")) not in seen]
    return existing + added, len(incoming) - len(added)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True, help="source name, e.g. rekabet")
    ap.add_argument("--in", dest="infile", required=True, help="the export JSON to convert")
    ap.add_argument("--dry-run", action="store_true",
                    help="report what would change and write nothing")
    ap.add_argument("--out-dir", help="write here instead of data/. Use a staging "
                                      "directory to inspect an export before it "
                                      "touches the corpus the pipeline reads.")
    args = ap.parse_args()

    src = Path(args.infile)
    if not src.is_absolute():
        src = (Path.cwd() / src).resolve()
    if not src.is_file():
        raise SystemExit(f"no such file: {src}")

    dest = (Path(args.out_dir) if args.out_dir else DATA_DIR) / f"{args.source}.json"
    dest.parent.mkdir(parents=True, exist_ok=True)

    incoming, sql = unwrap(json.loads(src.read_text(encoding="utf-8")), src.name)
    check_schema(incoming, src.name)
    print(f"source   : {args.source}")
    print(f"export   : {src}")
    if sql:
        print(f"  sql    : {' '.join(sql.split())}")
    n_text, n_html = text_stats(incoming)
    print(f"  rows   : {len(incoming)}  (content_text {n_text}, html_content {n_html})")

    wrong = {r.get("court_type") for r in incoming} - {args.source}
    if wrong:
        raise SystemExit(f"{src.name}: rows carry court_type {sorted(wrong)}, "
                         f"not {args.source!r} -- wrong --source?")

    if dest.is_file():
        existing, _ = unwrap(json.loads(dest.read_text(encoding="utf-8")), dest.name)
        check_schema(existing, dest.name)
        et, eh = text_stats(existing)
        print(f"existing : {dest}")
        print(f"  rows   : {len(existing)}  (content_text {et}, html_content {eh})")
    else:
        existing = []
        print(f"existing : {dest.name} does not exist, creating")

    merged, dupes = merge(existing, incoming)
    mt, mh = text_stats(merged)
    print(f"merged   : {len(merged)} rows  "
          f"(+{len(merged) - len(existing)} new, {dupes} already present)")
    print(f"  with content_text: {mt}   with html_content: {mh}")
    if mt == 0 and mh == 0:
        print(f"  NOTE: no row in {args.source} has extracted text. The decision "
              f"bodies are PDFs (metadata.data.pdf_url); this source cannot be "
              f"chunked until extraction runs upstream.")

    if args.dry_run:
        print("\ndry run -- nothing written")
        return 0

    if dest.is_file():
        backup = dest.with_suffix(".json.bak-before-merge")
        shutil.copy2(dest, backup)
        print(f"backup   : {backup.name}")

    dest.write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"written  : {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
