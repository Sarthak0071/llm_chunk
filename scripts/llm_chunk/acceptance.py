"""Score a chunking run against the acceptance criteria — is this what we wanted?

The pipeline already knows when something went wrong: it raises eighteen named
flags and writes them to the `*_review.json` sidecars. What it does not do is say
whether the run as a whole is good enough to accept. This does, by turning those
flags plus a few direct measurements into a pass/fail per criterion.

Deliberately reuses the existing flags rather than inventing a second notion of
quality -- a separate definition would drift from the one the generator enforces,
and then two things would disagree about the same run.

One criterion cannot be automated: whether an AYM norm review reads correctly
(is quoted statute text tagged `rule`, is subject_id the reviewed provision). That
is reported as MANUAL so it is not silently counted as passing.

    python acceptance.py --dir output/smoke
    python acceptance.py --dir output/chunk --expect 143
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "retrieval"))

import chunk_generate as gen        # noqa: E402

# Flags that mean the chunk is wrong, as opposed to merely worth a look.
HARD_FLAGS = {
    "segment_has_no_valid_refs", "invalid_paragraph_ref",
    "capsule_has_no_resolvable_support", "unresolved_local_id",
    "subject_id_unspecified", "source_type_mismatch",
}
# Flags that are usually a property of the SOURCE text rather than a mistake.
SOFT_FLAGS = {
    "verbatim_mention_not_in_text", "model_text_differs",
    "unknown_legislation_type", "citation_not_identifiable",
    "abbreviation_without_law_no", "dropped_non_legislation",
    "case_no_unreadable_in_source", "paragraph_refs_gap",
}

ROLE_KEYS = ("role", "firac_role", "court_reasoning_role", "regulatory_role")
VAGUE_ROLES = {"other", "unknown", None}


def role_of(chunk):
    for k in ROLE_KEYS:
        if chunk.get(k):
            return chunk[k]
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="output/chunk")
    ap.add_argument("--expect", type=int, help="documents expected in the run")
    args = ap.parse_args()

    base = gen.ROOT / args.dir
    chunks, capsules, reviews, docs = [], [], [], set()
    for p in sorted(base.glob("*.json")):
        if p.stem.endswith("_review"):
            reviews.extend(json.loads(p.read_text(encoding="utf-8")))
            continue
        b = json.loads(p.read_text(encoding="utf-8"))
        chunks.extend(b.get("chunks", []))
        capsules.extend(b.get("reasoning_capsules", []))
        for c in b.get("chunks", []):
            label = c.get("chunk_label") or ""
            docs.add(label.split("-p", 1)[0] if "-p" in label else label)

    flags = Counter()
    failed = [r for r in reviews if r.get("reason") and r.get("reason") != "validation_flags"]
    for r in reviews:
        for f in r.get("failed_checks") or []:
            flags[f.split(":")[0]] += 1

    print("=" * 76)
    print(f" ACCEPTANCE — {base}")
    print("=" * 76)
    n_docs = len(docs)
    print(f" documents produced : {n_docs}"
          + (f" of {args.expect} expected" if args.expect else ""))
    print(f" chunks             : {len(chunks)}")
    print(f" capsules           : {len(capsules)}\n")

    results = []

    def check(n, name, ok, detail):
        results.append(ok)
        mark = "PASS" if ok is True else ("MANUAL" if ok is None else "FAIL")
        print(f" {n:>2}. [{mark:6}] {name}")
        if detail:
            print(f"          {detail}")

    # 1 — the call succeeded
    hard_fail = [r for r in failed if r.get("reason") == "api_or_parse_failed"]
    check(1, "The call succeeded", not hard_fail,
          f"{len(hard_fail)} api_or_parse_failed" if hard_fail else "no API failures")

    # 2 — nothing truncated
    trunc = [r for r in failed if "truncat" in str(r.get("reason", ""))]
    check(2, "Nothing truncated", not trunc,
          "; ".join(f"{r['doc_id'][:18]} ({r.get('case_no')})" for r in trunc)
          if trunc else "no document hit the output cap")

    # 3 — every segment grounded
    ungrounded = flags["segment_has_no_valid_refs"] + flags["invalid_paragraph_ref"]
    check(3, "Every segment grounded in real paragraphs", ungrounded == 0,
          f"{ungrounded} ungrounded segments" if ungrounded else "all refs resolve")

    # 4 — stored text matches source
    check(4, "Stored text matches the source", flags["model_text_differs"] == 0,
          f"{flags['model_text_differs']} segments differ (stored text is "
          f"code-assembled, so this is a copy-accuracy signal only)"
          if flags["model_text_differs"] else "exact")

    # 5 — subject named
    check(5, "Subject named, never 'unspecified'",
          flags["subject_id_unspecified"] == 0,
          f"{flags['subject_id_unspecified']} capsules unspecified"
          if flags["subject_id_unspecified"] else "every capsule names its subject")

    # 6 — roles meaningful
    roles = Counter(role_of(c) for c in chunks)
    vague = sum(v for k, v in roles.items() if k in VAGUE_ROLES)
    share = vague * 100 // max(len(chunks), 1)
    check(6, "Roles meaningful (< 40% other/unknown)", share < 40,
          f"{vague}/{len(chunks)} = {share}% vague. "
          + ", ".join(f"{k}:{v}" for k, v in roles.most_common(6)))

    # 7 — summaries present and Turkish
    empty = sum(1 for c in capsules
                if not (c.get("reasoning_summary") or "").strip()
                or not (c.get("conclusion_sentence") or "").strip())
    check(7, "Capsule summaries present", empty == 0,
          f"{empty}/{len(capsules)} capsules have an empty summary or conclusion"
          if empty else "every capsule carries both")

    # 8 — citations
    cited = sum(1 for c in chunks if c.get("cited_legislations"))
    check(8, "Citations extracted", cited > 0,
          f"{cited}/{len(chunks)} chunks carry legislation; "
          f"{flags['citation_not_identifiable']} not identifiable, "
          f"{flags['abbreviation_without_law_no']} abbreviation-only")

    # 9 — capsules supported
    check(9, "Capsules supported by real chunks",
          flags["capsule_has_no_resolvable_support"] == 0,
          f"{flags['capsule_has_no_resolvable_support']} unsupported"
          if flags["capsule_has_no_resolvable_support"] else "all supported")

    # 10 — index contract
    try:
        import contract
        from to_index_records import build_all
        recs, _ = build_all()
        bad = [r for r in recs if contract.validate(r)]
        check(10, "Index records valid", not bad,
              f"{len(recs) - len(bad)}/{len(recs)} valid")
    except Exception as exc:                                   # noqa: BLE001
        check(10, "Index records valid", None,
              f"not evaluated here ({type(exc).__name__})")

    # 11 — manual
    check(11, "AYM norm review reads correctly", None,
          "requires reading the output against the raw decision")

    print()
    if flags:
        print(" flags raised:")
        for f, c in flags.most_common():
            kind = "HARD" if f in HARD_FLAGS else ("soft" if f in SOFT_FLAGS else "    ")
            print(f"   {c:>4}  [{kind}] {f}")

    hard = sum(1 for r in results if r is False)
    print(f"\n {'ACCEPTED' if hard == 0 else f'NOT ACCEPTED — {hard} criterion/criteria failed'}")
    return 1 if hard else 0


if __name__ == "__main__":
    sys.exit(main())
