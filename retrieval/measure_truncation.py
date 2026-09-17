"""E1 — how much of the corpus is invisible to a whole-document vector index?

Production embeds one whole decision as one vector using BAAI/bge-m3, which accepts
at most 8192 tokens. Anything past that is silently dropped: no error, no warning,
no record that the tail of the decision was never indexed.

This measures the real figure with bge-m3's OWN tokenizer (XLMRobertaTokenizer) on
our actual Turkish legal text, rather than assuming a chars-per-token ratio. Turkish
is agglutinative and legal Turkish is full of long suffixed forms, so the ratio is
worth measuring rather than guessing.

Chunks have no such ceiling — that is the point of the experiment.

Tokenizer only. No model weights, no embedding, no network beyond the one-off
tokenizer download. Free to run.

    python measure_truncation.py
    python measure_truncation.py --sample 200      # quicker pass
"""

import argparse
import json
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "scripts" / "llm_chunk"))
sys.path.insert(0, str(ROOT / "scripts" / "retrieval_test"))

import chunk_generate as gen        # noqa: E402
import corpus as cor                # noqa: E402

MODEL = "BAAI/bge-m3"
MAX_TOKENS = 8192


def load_documents(limit=None):
    """(source, doc_id, text) for every document that actually has text."""
    out = []
    for path in sorted(ROOT.glob("data/*.json")):
        if ".bak" in path.name or path.stem.startswith("documents_document"):
            continue
        source = path.stem
        if source not in gen.SOURCES:
            continue
        for rec in json.loads(path.read_text(encoding="utf-8")):
            paragraphs = []
            try:
                paragraphs = gen.extract_paragraphs(rec, source)
            except NotImplementedError:
                continue          # rekabet / uyusmazlik: no text upstream
            if paragraphs:
                out.append((source, str(rec.get("doc_id")), "\n".join(paragraphs)))
            if limit and len(out) >= limit:
                return out
    return out


def load_chunks():
    """Our chunks, for the comparison half."""
    out = []
    for path in sorted((ROOT / "output" / "chunk").glob("*.json")):
        if path.stem.endswith("_review"):
            continue
        for ch in json.loads(path.read_text(encoding="utf-8")).get("chunks", []):
            out.append((path.stem, ch.get("text") or ""))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int)
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)

    docs = load_documents(args.sample)
    print("=" * 74)
    print(" E1 — TRUNCATION LOSS IN A WHOLE-DOCUMENT VECTOR INDEX")
    print("=" * 74)
    print(f" tokenizer : {tok.__class__.__name__} from {MODEL}")
    print(f" ceiling   : {MAX_TOKENS} tokens (model_max_length = {tok.model_max_length})")
    print(f" documents : {len(docs)}\n")
    print(" tokenising… (no model weights, tokenizer only)")

    rows = []
    for i, (source, doc_id, text) in enumerate(docs):
        n_tok = len(tok.encode(text, add_special_tokens=False, truncation=False))
        rows.append({"source": source, "doc_id": doc_id, "chars": len(text),
                     "tokens": n_tok})
        if (i + 1) % 200 == 0:
            print(f"   {i + 1}/{len(docs)}")

    ratios = [r["chars"] / r["tokens"] for r in rows if r["tokens"]]
    ratio = statistics.median(ratios)
    over = [r for r in rows if r["tokens"] > MAX_TOKENS]
    total_tok = sum(r["tokens"] for r in rows)
    seen_tok = sum(min(r["tokens"], MAX_TOKENS) for r in rows)

    print(f"\n MEASURED chars per token (median) : {ratio:.2f}")
    print(f"   -> the ceiling is about {MAX_TOKENS * ratio:,.0f} characters of "
          f"Turkish legal text")

    print(f"\n documents over the ceiling        : {len(over)} of {len(rows)} "
          f"({len(over) * 100 / len(rows):.1f}%)")
    print(f" corpus tokens a vector index sees : {seen_tok * 100 / total_tok:.1f}%")
    print(f" corpus tokens INVISIBLE           : "
          f"{100 - seen_tok * 100 / total_tok:.1f}%")

    print(f"\n{'source':14} {'docs':>6} {'over':>6} {'%over':>7} {'median tok':>11} "
          f"{'max tok':>9} {'% seen':>7}")
    by = {}
    for r in rows:
        by.setdefault(r["source"], []).append(r)
    for src, rs in sorted(by.items()):
        o = [r for r in rs if r["tokens"] > MAX_TOKENS]
        t = sum(r["tokens"] for r in rs)
        s = sum(min(r["tokens"], MAX_TOKENS) for r in rs)
        print(f"{src:14} {len(rs):>6} {len(o):>6} {len(o) * 100 / len(rs):>6.1f}% "
              f"{statistics.median(r['tokens'] for r in rs):>11,.0f} "
              f"{max(r['tokens'] for r in rs):>9,} {s * 100 / t:>6.1f}%")

    worst = sorted(rows, key=lambda r: -r["tokens"])[:5]
    print(f"\n worst affected documents:")
    for r in worst:
        pct = MAX_TOKENS * 100 / r["tokens"]
        print(f"   {r['source']:13} {r['doc_id'][:16]:18} {r['tokens']:>8,} tokens"
              f"  ->  only {pct:.0f}% embedded")

    # --- the comparison half -------------------------------------------------
    chunks = load_chunks()
    if chunks:
        ctok = [len(tok.encode(t, add_special_tokens=False, truncation=False))
                for _, t in chunks]
        over_c = [n for n in ctok if n > MAX_TOKENS]
        print(f"\n OUR CHUNKS ({len(chunks)}):")
        print(f"   median {statistics.median(ctok):,.0f} tokens, "
              f"max {max(ctok):,}, over ceiling {len(over_c)}")
        print(f"   -> {100.0 if not over_c else 0:.0f}% of chunk text is embedded"
              if not over_c else
              f"   -> {len(over_c)} chunks exceed the ceiling")
        headroom = MAX_TOKENS / statistics.median(ctok)
        print(f"   chunks use {statistics.median(ctok) * 100 / MAX_TOKENS:.1f}% of "
              f"the ceiling — {headroom:.0f}x headroom")
        print(f"   (feeds llm_chunk change #4: is the 2,000-char cap too small?)")

    out = HERE / "report" / "truncation.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(
        {"model": MODEL, "max_tokens": MAX_TOKENS,
         "median_chars_per_token": round(ratio, 3),
         "documents": len(rows), "over_ceiling": len(over),
         "pct_tokens_visible": round(seen_tok * 100 / total_tok, 2),
         "rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
