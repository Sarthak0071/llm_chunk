"""Turn our generated chunks into index records, and prove they satisfy the contract.

Reads output/chunk/*.json plus the raw court records in data/, emits one record per
chunk in the shape Hammurabi can actually index, and reports every record that
would fail.

No embedding, no Gemini, no network. Free to run as often as you like.

    python to_index_records.py                 # validate, report, write nothing
    python to_index_records.py --write         # also write report/index_records.json
    python to_index_records.py --source kvkk   # one source
    python to_index_records.py --show 2        # print 2 full records
"""

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent                                   # llm_chunk/
sys.path.insert(0, str(ROOT / "scripts" / "llm_chunk"))
sys.path.insert(0, str(ROOT / "scripts" / "retrieval_test"))
sys.path.insert(0, str(HERE))

import chunk_generate as gen        # noqa: E402  (SOURCES: role_field per source)
import corpus as cor                # noqa: E402  (doc_id_of)
import contract                     # noqa: E402

CHUNK_DIR = ROOT / "output" / "chunk"
DATA_DIR = ROOT / "data"


def load_raw():
    """{(source, doc_id): raw court record}"""
    out = {}
    for path in sorted(DATA_DIR.glob("*.json")):
        if ".bak" in path.name or path.stem.startswith("documents_document"):
            continue
        if path.stem not in gen.SOURCES:
            continue
        for rec in json.loads(path.read_text(encoding="utf-8")):
            out[(path.stem, str(rec.get("doc_id")))] = rec
    return out


def build_all(only_source=None):
    raw = load_raw()
    records, skipped = [], []

    for path in sorted(CHUNK_DIR.glob("*.json")):
        if path.stem.endswith("_review"):
            continue
        source = path.stem
        if only_source and source != only_source:
            continue
        role_field = gen.SOURCES[source]["role_field"]
        payload = json.loads(path.read_text(encoding="utf-8"))

        # chunk_index is position WITHIN its document, not within the file -- it is
        # used to show passages in reading order, so it must restart per decision.
        per_doc = {}
        for chunk in payload.get("chunks", []):
            doc_id = cor.doc_id_of(chunk)
            record = raw.get((source, str(doc_id)))
            if record is None:
                skipped.append(f"{source}/{doc_id}: no raw record in data/")
                continue
            idx = per_doc.get(doc_id, 0)
            per_doc[doc_id] = idx + 1
            records.append(contract.build_record(chunk, record, source, idx, role_field))

    return records, skipped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source")
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--show", type=int, default=0)
    args = ap.parse_args()

    records, skipped = build_all(args.source)
    if not records:
        print("no chunks found in output/chunk/")
        return 1

    print("=" * 72)
    print(" INDEX RECORD CONTRACT")
    print("=" * 72)
    print(f" chunks converted : {len(records)}")
    print(f" schema version   : {contract.SCHEMA_VERSION}")
    if skipped:
        print(f" skipped          : {len(skipped)}")
        for s in skipped[:5]:
            print(f"   {s}")

    # --- failures -----------------------------------------------------------
    failed = [(r, p) for r in records if (p := contract.validate(r))]
    print(f"\n contract failures: {len(failed)} of {len(records)}")
    if failed:
        seen = {}
        for r, probs in failed:
            for p in probs:
                key = p.split(":")[0]
                seen.setdefault(key, []).append(r)
        for key, rs in sorted(seen.items(), key=lambda x: -len(x[1])):
            srcs = sorted({r["high_court"] for r in rs})
            print(f"   {len(rs):>4}  {key:<34} {','.join(srcs)}")

    # --- per source ---------------------------------------------------------
    print(f"\n{'source':14} {'chunks':>7} {'ok':>5} {'chamber':>8} {'E_no':>6} "
          f"{'K_no':>6} {'paras':>6}")
    by = {}
    for r in records:
        by.setdefault(r["high_court"], []).append(r)
    for src, rs in sorted(by.items()):
        ok = sum(1 for r in rs if not contract.validate(r))
        print(f"{src:14} {len(rs):>7} {ok:>5} "
              f"{sum(1 for r in rs if r['chamber'] is not None):>8} "
              f"{sum(1 for r in rs if r['E_no']):>6} "
              f"{sum(1 for r in rs if r['K_no']):>6} "
              f"{sum(1 for r in rs if r['paragraph_ids']):>6}")

    # --- what we had to substitute -----------------------------------------
    notes = {}
    for r in records:
        for n in r["derivation_notes"]:
            notes[n] = notes.get(n, 0) + 1
    print("\n derivations applied (where the raw data was incomplete):")
    for n, c in sorted(notes.items(), key=lambda x: -x[1]):
        print(f"   {c:>4}  {n}")
    if not notes:
        print("   none")

    # --- production's own gaps ---------------------------------------------
    gaps = {}
    for r in records:
        for f in contract.production_gap(r):
            gaps.setdefault(f, set()).add(r["high_court"])
    print("\n production fields still null (not failures -- genuinely absent):")
    for f, srcs in sorted(gaps.items()):
        print(f"   {f:<14} {','.join(sorted(srcs))}")
    if not gaps:
        print("   none")

    # --- identity sanity ----------------------------------------------------
    uuids = [r["uuid"] for r in records]
    parents = {r["parent_uuid"] for r in records}
    print(f"\n uuid unique      : {len(set(uuids)) == len(uuids)}  "
          f"({len(set(uuids))}/{len(uuids)})")
    print(f" distinct parents : {len(parents)}  (expected = documents chunked)")

    for r in records[:args.show]:
        print("\n" + json.dumps({k: v for k, v in r.items()
                                 if k not in ("text", "cited_legislations")},
                                ensure_ascii=False, indent=2))
        print(f'  text: {(r["text"] or "")[:120]}...')

    if args.write:
        out = HERE / "report" / "index_records.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(records, ensure_ascii=False, indent=2),
                       encoding="utf-8")
        print(f"\n wrote {out}")

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
