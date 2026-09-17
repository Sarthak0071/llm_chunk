"""What KINDS of document does each source actually contain?

The pre-flight answers "can this document be read". This answers the harder
question behind it: "is this the same KIND of document our prompt was written
for?"

The difference is not academic. All 65 AYM decisions in a year-spread export turned
out to be norm review -- constitutional review of a statute -- while our AYM prompt
describes individual applications, which is what 94% of the older sample was. Both
are read perfectly. One of them is described by a prompt about applicants and
violated rights that it does not have.

A document type is inferred from three signals that do not depend on knowing the
source in advance:

  TITLE SHAPE      digits replaced by #, so "E.1962/19, K.1962/18 Sayılı Karar"
                   and "E.2024/7, K.2024/9 Sayılı Karar" collapse to one shape
  METADATA KEYS    which fields the record carries; norm review has no
                   examination_results, individual applications do
  STRUCTURE        which of the section markers our prompt names are present

Documents sharing all three are the same archetype. An archetype our prompt does
not describe is where role tagging becomes guesswork.

    python profile_documents.py                      # data/
    python profile_documents.py --data-dir data/incoming
    python profile_documents.py --source aym --examples 2
"""

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import chunk_generate as gen        # noqa: E402
from preflight import EXPECTED_MARKERS, deaccent  # noqa: E402

RE_DIGITS = re.compile(r"\d+")


def title_shape(title):
    """"E.1962/19, K.1962/18 Sayılı Karar" -> "E.#/#, K.#/# Sayılı Karar".

    Numbers are what differ between two decisions of the same kind; everything
    else is the house style of that kind.
    """
    t = RE_DIGITS.sub("#", (title or "").strip())
    t = re.sub(r"\s+", " ", t)
    return t[:70] or "(no title)"


def metadata_shape(record):
    """The inner metadata keys, which differ by document type more reliably than
    the text does -- an individual application carries examination_results, a norm
    review carries nothing of the kind."""
    data = (gen.parse_metadata(record) or {}).get("data") or {}
    return tuple(sorted(data)) if isinstance(data, dict) else ()


def marker_set(record, source):
    expected = EXPECTED_MARKERS.get(source, [])
    if not expected:
        return ()
    try:
        paras = gen.extract_paragraphs(record, source)
    except Exception:                                          # noqa: BLE001
        return ()
    body = deaccent(" ".join(paras)).upper()
    return tuple(m for m in expected if deaccent(m).upper() in body)


def profile(records, source):
    groups = defaultdict(list)
    for r in records:
        key = (title_shape(r.get("title")), marker_set(r, source))
        groups[key].append(r)
    return groups


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source")
    ap.add_argument("--data-dir")
    ap.add_argument("--examples", type=int, default=1)
    args = ap.parse_args()

    base = Path(args.data_dir) if args.data_dir else gen.DATA_DIR
    sources = [args.source] if args.source else sorted(gen.SOURCES)

    print("=" * 78)
    print(f" DOCUMENT ARCHETYPES in {base}")
    print("=" * 78)

    report = {}
    for source in sources:
        path = base / f"{source}.json"
        if not path.is_file():
            continue
        records = json.loads(path.read_text(encoding="utf-8"))
        field = gen.SOURCES[source]["text_field"]
        usable = [r for r in records
                  if r.get("status") in gen.USABLE_STATUSES
                  and field and (r.get(field) or r.get("html_content") or "").strip()]
        if not usable:
            continue

        groups = profile(usable, source)
        expected = EXPECTED_MARKERS.get(source, [])
        print(f"\n{'-' * 78}\n {source}  —  {len(usable)} usable documents, "
              f"{len(groups)} archetype(s)\n{'-' * 78}")

        rows = []
        for (shape, markers), docs in sorted(groups.items(), key=lambda x: -len(x[1])):
            share = len(docs) * 100 // len(usable)
            covered = (not expected) or (len(markers) >= max(2, len(expected) // 3))
            metas = Counter(metadata_shape(d) for d in docs)
            meta_keys = metas.most_common(1)[0][0] if metas else ()
            print(f"\n  {len(docs):>4} docs ({share:>3}%)   "
                  f"{'COVERED' if covered else 'NOT DESCRIBED BY OUR PROMPT'}")
            print(f"       title   : {shape}")
            print(f"       markers : {', '.join(markers) if markers else 'none'}"
                  f"  ({len(markers)} of {len(expected)})" if expected
                  else "       markers : n/a (prompt is written around function)")
            print(f"       metadata: {', '.join(meta_keys) if meta_keys else 'none'}")
            yrs = sorted({d.get("karar_year") or d.get("esas_year") for d in docs}
                         - {None})
            if yrs:
                print(f"       years   : {yrs[0]}–{yrs[-1]}"
                      + (f"  ({len(yrs)} distinct)" if len(yrs) > 1 else ""))
            for d in docs[:args.examples]:
                print(f"       example : {d.get('doc_id')}  "
                      f"{(d.get('title') or '')[:52]}")
            rows.append({"docs": len(docs), "share": share, "covered": covered,
                         "title_shape": shape, "markers": list(markers),
                         "metadata_keys": list(meta_keys), "years": yrs})
        report[source] = rows

    print(f"\n{'=' * 78}\n SUMMARY — where the prompt does not describe the document\n"
          f"{'=' * 78}")
    total_bad = 0
    for source, rows in report.items():
        bad = [r for r in rows if not r["covered"]]
        n = sum(r["docs"] for r in rows)
        nb = sum(r["docs"] for r in bad)
        total_bad += nb
        flag = "  <-- needs a prompt variant" if nb else ""
        print(f"  {source:14} {nb:>4} of {n:>4} documents "
              f"({nb * 100 // max(n, 1):>3}%) not described{flag}")
    print(f"\n  {total_bad} documents in total would be tagged by a prompt written "
          f"for a different\n  kind of decision. They will still chunk -- the roles "
          f"are what is at risk.")

    out = gen.ROOT / "output" / "archetypes.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
