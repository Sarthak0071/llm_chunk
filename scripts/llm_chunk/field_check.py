"""Test the rekabet and uyusmazlik wiring against every real row -- without text.

Neither source can be chunked: all 10,367 rekabet rows in the database are
status=pending_extraction and a `content_text <> ''` filter returns none, and the
uyusmazlik rows on disk look the same. The bodies are PDFs.

But the METADATA is complete, so everything derived from it can be verified now
rather than the day extraction lands. This runs the real pipeline functions over
all 250 rekabet and 350 uyusmazlik records and asserts the results, then proves
by injection that the only missing piece is the paragraph strategy.

No Gemini call, no cost, no write.

    python field_check.py
"""

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import chunk_generate as gen        # noqa: E402

DATA_DIR = gen.ROOT / "data"
ISO = re.compile(r"^\d{4}-\d{2}-\d{2}$")

# Sentinels the uyusmazlik scraper writes when it cannot read a field. A row like
# this must yield a clean null WITH a reason -- the silent-null failure that the
# _iso_date separator bug produced last round, where a date simply vanished and
# nothing recorded that it had.
SENTINELS = {"Bulunamadı", "Belirtilmemiş"}


def records(source):
    path = DATA_DIR / f"{source}.json"
    if not path.is_file():
        raise SystemExit(f"missing {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def check_source(source):
    rows = records(source)
    fails, notes = [], []

    # --- case_no -------------------------------------------------------------
    case_nos, missing_case = [], []
    for r in rows:
        cn = gen.compute_case_no(r, source)
        (case_nos if cn else missing_case).append(cn or r.get("doc_id"))
    dupes = len(case_nos) - len(set(case_nos))
    if missing_case:
        fails.append(f"case_no null on {len(missing_case)}/{len(rows)} rows "
                     f"(first: {missing_case[0]})")
    if dupes:
        fails.append(f"case_no repeats on {dupes} rows -- not unique per decision")
    bad_shape = [c for c in case_nos if not gen.is_wellformed_case_no(c, source)]
    if bad_shape:
        fails.append(f"case_no fails this source's shape on {len(bad_shape)} rows "
                     f"(first: {bad_shape[0]!r})")
    notes.append(f"case_no      : {len(case_nos)}/{len(rows)} non-null, "
                 f"{len(set(case_nos))} distinct, shape ok "
                 f"(e.g. {case_nos[0]!r})" if case_nos else "case_no: none")

    # --- decision_date -------------------------------------------------------
    # Paragraphs are [] for every row here, so this exercises the metadata path
    # only -- which is the path these sources will always use.
    dated, nulled, malformed = 0, [], []
    for r in rows:
        iso, reason = gen.compute_decision_date(r, source, [])
        if iso is None:
            nulled.append((r.get("doc_id"), reason))
        elif not ISO.match(iso):
            malformed.append(iso)
        else:
            dated += 1
    if malformed:
        fails.append(f"decision_date not ISO on {len(malformed)} rows "
                     f"(first: {malformed[0]!r})")
    for doc_id, reason in nulled:
        if not reason:
            fails.append(f"decision_date null with NO reason on {doc_id} -- "
                         f"a silent null is the bug, not the null")
    notes.append(f"decision_date: {dated}/{len(rows)} valid ISO, "
                 f"{len(nulled)} null (all flagged)")

    # A sentinel row must land in `nulled`, never be parsed into a date.
    sentinel_rows = [r for r in rows
                     if (gen.parse_metadata(r).get("data") or {}).get("karar_no_raw")
                     in SENTINELS]
    if sentinel_rows:
        nulled_ids = {d for d, _ in nulled}
        leaked = [r.get("doc_id") for r in sentinel_rows
                  if r.get("doc_id") not in nulled_ids]
        if leaked:
            fails.append(f"{len(leaked)} sentinel row(s) produced a date anyway: {leaked}")
        notes.append(f"sentinel rows: {len(sentinel_rows)} found, all correctly null")

    # --- subject_type --------------------------------------------------------
    subj = {}
    for r in rows:
        subj[gen.resolve_subject_type(r, source)] = subj.get(
            gen.resolve_subject_type(r, source), 0) + 1
    if source == "rekabet":
        known = set(gen.REKABET_SUBJECT_TYPE.values())
        unknown = set(subj) - known
        if unknown:
            fails.append(f"subject_type outside the mapped vocabulary: {sorted(unknown)}")
        raw_types = {(gen.parse_metadata(r).get("data") or {}).get("decision_type")
                     for r in rows}
        unmapped = {t for t in raw_types if t and t not in gen.REKABET_SUBJECT_TYPE}
        if unmapped:
            fails.append(f"decision_type values with no mapping: {sorted(unmapped)} "
                         f"-- new Kurul vocabulary, map them deliberately")
    notes.append("subject_type : " + ", ".join(f"{k} {v}" for k, v in sorted(subj.items())))

    # --- text, and the guard -------------------------------------------------
    with_text = sum(1 for r in rows
                    if (r.get("content_text") or "").strip()
                    or (r.get("html_content") or "").strip())
    notes.append(f"extracted text: {with_text}/{len(rows)} rows")
    if with_text:
        fails.append(f"{with_text} rows now HAVE text -- this source is no longer "
                     f"text-pending. Read 2-3 decisions, choose a paragraph "
                     f"strategy, and remove {source!r} from TEXT_PENDING_SOURCES.")

    picked, skipped_empty, _ = gen.pick_documents(rows, source, 2)
    if picked:
        fails.append(f"pick_documents returned {len(picked)} documents for a "
                     f"text-pending source")
    if skipped_empty != len(rows):
        fails.append(f"pick_documents counted {skipped_empty} empty, expected {len(rows)}")

    return fails, notes, rows


def injection_test(rows, source):
    """The 'will it work when the data arrives?' test.

    Give one real record a body and confirm the pipeline gets all the way to the
    paragraph split before stopping -- with the NAMED error, not a KeyError or a
    silent []. That is the difference between one known missing piece and an
    integration nobody has tried.
    """
    rec = dict(rows[0])
    rec["content_text"] = "BIRINCI BOLUM\nIkinci satir.\nUCUNCU SATIR."
    rec["html_content"] = "<p>BIRINCI BOLUM</p><p>Ikinci satir.</p>"
    rec["status"] = "completed"

    out = []
    case_no = gen.compute_case_no(rec, source)
    date, _ = gen.compute_decision_date(rec, source, [])
    subject = gen.resolve_subject_type(rec, source)
    out.append(f"  case_no={case_no!r}  decision_date={date!r}  subject_type={subject!r}")
    if not (case_no and date and subject):
        return out + ["  FAIL: a metadata field came back empty with text present"], False

    try:
        gen.extract_paragraphs(rec, source)
    except NotImplementedError as exc:
        first = str(exc).split(".")[0]
        out.append(f"  extract_paragraphs -> NotImplementedError: {first}.")
        out.append("  correct: metadata wired, exactly one piece missing")
        return out, True
    except Exception as exc:                                  # noqa: BLE001
        out.append(f"  FAIL: {type(exc).__name__}: {exc}")
        return out, False
    out.append("  FAIL: paragraphs were extracted, but no strategy was ever chosen")
    return out, False


def main():
    ok = True
    for source in sorted(gen.TEXT_PENDING_SOURCES):
        print(f"=== {source} ===")
        fails, notes, rows = check_source(source)
        for n in notes:
            print(f"  {n}")
        print("  -- injection test (real record + synthetic body):")
        lines, passed = injection_test(rows, source)
        for line in lines:
            print(line)
        if fails:
            ok = False
            print(f"  FAIL ({len(fails)}):")
            for f in fails:
                print(f"    {f}")
        elif not passed:
            ok = False
        else:
            print("  PASS")
        print()

    print("ALL PASS" if ok else "FAILURES ABOVE")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
