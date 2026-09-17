"""E7 — does production's procedural penalty punish chunks unfairly?

Production multiplies a result's score by 0.4 if its text matches a procedural
phrase. The test is a plain substring search with no sense of proportion, so the
question is whether storing a decision as chunks makes it MORE likely to be hit.

The comparison is exact: for each chunked decision, is the whole document flagged,
and how many of its chunks are flagged? If a document is clean but some of its
chunks are flagged, those chunks lose 60% of their score for text the baseline
carries too — a penalty caused by the storage unit, not by the content.

Needs no Qdrant, no Meilisearch, no embedding.

    python measure_procedural.py
"""

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "scripts" / "llm_chunk"))
sys.path.insert(0, str(ROOT / "scripts" / "retrieval_test"))
sys.path.insert(0, str(HERE))

import chunk_generate as gen        # noqa: E402
import corpus as cor                # noqa: E402
import procedural as proc           # noqa: E402
import config as cfg                # noqa: E402


def main():
    c = cor.Corpus()
    raw = {}
    for path in sorted((ROOT / "data").glob("*.json")):
        if ".bak" in path.name or path.stem.startswith("documents_document"):
            continue
        if path.stem not in gen.SOURCES:
            continue
        for rec in json.loads(path.read_text(encoding="utf-8")):
            raw[(path.stem, str(rec.get("doc_id")))] = rec

    print("=" * 76)
    print(" E7 — PROCEDURAL PENALTY: does chunking change who gets punished?")
    print("=" * 76)
    print(f" penalty   : x{cfg.PROCEDURAL_PENALTY} (a 60% score cut)")
    print(f" patterns  : {len(cfg.PROCEDURAL_PATTERNS)} procedural + "
          f"{len(proc.JURISDICTIONAL_PATTERNS)} jurisdictional\n")

    print(f"{'source / doc':34} {'doc?':>5} {'chunks':>7} {'flagged':>8} {'new?':>6}")
    print("-" * 76)

    totals = {"docs_flagged": 0, "chunks_flagged": 0, "chunks": 0,
              "new_penalty": 0, "docs": 0}
    detail = []

    for source, doc_id in sorted(c.documents):
        rec = raw.get((source, str(doc_id)))
        if rec is None:
            continue
        paragraphs = gen.extract_paragraphs(rec, source)
        doc_text = "\n".join(paragraphs)
        doc_flag, doc_reason = proc.is_procedural(doc_text)

        chunks = c.chunks_of_doc(source, doc_id)
        flagged = [ch for ch in chunks if proc.is_procedural(ch["text"])[0]]
        # Chunks penalised where the document is NOT -- pure storage-unit effect.
        new_penalty = len(flagged) if not doc_flag else 0

        totals["docs"] += 1
        totals["docs_flagged"] += int(doc_flag)
        totals["chunks"] += len(chunks)
        totals["chunks_flagged"] += len(flagged)
        totals["new_penalty"] += new_penalty

        print(f"{source + '/' + str(doc_id)[:18]:34} {'YES' if doc_flag else '-':>5} "
              f"{len(chunks):>7} {len(flagged):>8} {new_penalty if new_penalty else '-':>6}")

        for ch in flagged:
            detail.append({
                "source": source, "doc_id": str(doc_id),
                "chunk_label": ch.get("chunk_label"),
                "role": ch.get("role") or ch.get("court_reasoning_role")
                        or ch.get("firac_role") or ch.get("regulatory_role"),
                "content_type": ch.get("content_type"),
                "chars": len(ch["text"]),
                "reason": proc.is_procedural(ch["text"])[1],
                "phrase_density": round(proc.density(ch["text"]), 4),
                "doc_also_flagged": doc_flag,
            })

    print("-" * 76)
    print(f" documents flagged : {totals['docs_flagged']}/{totals['docs']}")
    print(f" chunks flagged    : {totals['chunks_flagged']}/{totals['chunks']}")
    print(f" chunks penalised where the DOCUMENT is not: {totals['new_penalty']}")

    if detail:
        print(f"\n the flagged chunks, and how much of each is the matched phrase:")
        print(f"   {'role':18} {'type':10} {'chars':>6} {'density':>8}  reason")
        for d in sorted(detail, key=lambda x: -x["phrase_density"]):
            print(f"   {str(d['role']):18} {str(d['content_type']):10} "
                  f"{d['chars']:>6} {d['phrase_density'] * 100:>7.2f}%  {d['reason']}"
                  + ("" if d["doc_also_flagged"] else "   <- document NOT flagged"))

        dens = [d["phrase_density"] for d in detail]
        print(f"\n A phrase is up to {max(dens) * 100:.1f}% of a flagged chunk.")
        print(" The same phrase inside a whole decision is a rounding error, yet both")
        print(" are penalised identically — the penalty measures presence, not weight.")

    verdict = ("NO measurable unfairness in this sample"
               if totals["new_penalty"] == 0 else
               f"{totals['new_penalty']} chunk(s) penalised purely for being chunks")
    print(f"\n VERDICT: {verdict}")
    if totals["new_penalty"] == 0 and totals["chunks_flagged"] == 0:
        print(" Note: zero flagged chunks means this is UNTESTED, not proven safe.")
        print(" Re-run when the corpus contains procedural decisions.")

    out = HERE / "report" / "procedural.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"totals": totals, "flagged_chunks": detail},
                              ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
