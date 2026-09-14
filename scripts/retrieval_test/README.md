# retrieval_test — storage verification + honest retrieval smoke test

Two separate questions, deliberately never conflated:

| | Question | Nature | Valid at 10 documents? |
|---|---|---|---|
| **Storage** | Is what got saved what should have been saved? | Mechanical | **Yes — real pass/fail** |
| **Retrieval** | Given a real question, is the right document found? | Statistical | **No — smoke test only** |

A chunk can be perfectly stored yet fail retrieval (weak `reasoning_summary`), or
pass retrieval by luck while being badly stored (a keyword hit on a bad field).
Testing them together hides which one broke.

## Order of operations

```powershell
cd C:\Users\NITRO\Desktop\Hammurabi\llm_chunk\scripts\retrieval_test

python verify_storage.py              # Stage 0 — real verdicts, run this first
python make_worksheet.py              # builds worksheet/ from the RAW court text
#   ... you write queries.json (see queries.example.json) ...
python test_retrieval.py --audit-only # leakage audit, no scoring
python test_retrieval.py              # full report -> output/retrieval/report.json
python test_retrieval.py --self-test  # proves the harness can actually fail
```

`verify_storage.py` exits **1** if any check FAILs, **0** if only warnings.

## Why queries must come from the worksheets

If a query is written by reading the generated `reasoning_summary` and rewording
it, BM25 is matching text against a paraphrase of itself: the test passes by
construction and measures nothing. So:

- **Queries come only from `worksheet/` files**, which contain raw court text and
  nothing else. Never open `output/chunk/` while writing a query.
- **Stage 2 ground truth is a paragraph number, not a chunk_id.** You write
  `expect_paragraph: "p18"`; code resolves which chunk contains it via
  `source_paragraph_ids`. You never read generated chunk text to pick an answer.
  This also means the answer key survives re-segmentation — chunk counts already
  moved 41 → 33 → 35 across runs, so a chunk_id key would rot every run.
- `--audit-only` reports, per query, the unigram overlap and the longest n-gram
  shared with its target. A shared 5-gram is flagged as a copied phrase. Nothing
  is auto-rejected; it is printed so a pass earned by overlap cannot hide.
- `queries.json` is SHA-256 hashed into every report, so edits made after seeing
  results are visible.

### Who wrote the current queries — disclosed, not hidden

The 30 queries in `queries.json` were written by **Claude** (the assistant), on
2026-09-14, from the `worksheet/` files only. The generated output was not opened
while writing. That is a weaker independence guarantee than a human author with no
stake in the result, and it should be read as such. Two things make it more than a
promise:

1. The audit is the enforcement. The first draft was flagged: one `case` query
   shared a 6-gram (`sigorta şirketi taşıma sırasında hasar gören`) with the
   generated capsule — not because either copied the other, but because the
   query and Gemini's summary converged on the same natural phrase for the same
   facts. It was reworded **before** the first scored run; the hash changed from
   `8e0e42…` to `238cf9…`, and the final audit shows 0 of 30 sharing a 5-gram.
2. Unigram overlap is printed per query and ranges 0.15–0.75. The high end is
   short questions about a single legal concept (`yargılamanın yenilenmesi`),
   where the terms *are* the concept. It is shown, not smoothed.

If you want the stronger guarantee, replace any query with your own — the
harness treats them identically, and the hash records that you did.

### On paragraph numbers

For **aym**, `[pN]` is the court's own numbered paragraph — a real citation unit.
For the other four, `[pN]` is *our* synthetic split (`splitlines()`, or `<p>` tags
for kvkk). Still sound as an answer key because it is deterministic code output
rather than generated content, but it is a location marker, not an official
paragraph reference. The worksheets say which you are looking at.

Note that **167 of 427 paragraph references resolve to more than one chunk**,
because every piece of a cap-split segment inherits the parent's full
`source_paragraph_ids`. So "any chunk containing that paragraph counts correct"
is the only sound rule, not merely a generous one.

## What the retrieval numbers can and cannot say

With 10 documents, chance alone scores Stage 1 top-1 **1 in 10**, and one query
flipping moves a per-source score by 50 points. So:

- **No aggregate percentage is printed anywhere.** Every number is `k/n` with its
  own chance baseline beside it, computed per query (hypergeometric, not assumed
  uniform).
- Stage 2 pools of ≤ 3 chunks print `top-3 meaningless` — at that size "in the
  top 3" is arithmetic, not retrieval.
- A **shuffled-label control** re-scores over 2,000 permutations of the
  query→document mapping. If the real score is not clearly above the shuffled
  mean, the real score is not evidence. (Verified: the control lands on 0.099
  against a theoretical 0.100.)
- **MRR** is reported alongside top-1/top-3, since rank position carries more
  information than a binary hit at this n.

**BM25 has no Turkish stemmer**, so `yargılanmayla` will not match `yargılanma`.
Every number here is therefore *pessimistic* relative to real Meilisearch, which
applies a Turkish analyser. A weak result is not by itself evidence of bad
chunking. Do not tune the tokenizer, `k1`, `b` or the stopword list against
results on a 10-document sample — that converts the harness into a rubber stamp.

## Independence

`verify_storage.py` shares no validation logic with `chunk_generate.py` and never
reads the `*_review.json` sidecars — a verifier that trusts the generator's own
account of itself cannot catch the generator's blind spots. It re-derives
`case_no`, the Turkish-language check and the enum vocabularies from scratch, and
adds the checks the generator structurally cannot do: corpus-wide `chunk_id`
uniqueness, cross-source schema comparison, and the document's own header versus
the database columns.

It does import `chunk_lib` for paragraph extraction, deliberately: using a
different extraction here would test extraction rather than storage.

Nothing in this folder reads anything outside `llm_chunk/`.

## Files

| File | Purpose |
|---|---|
| `bm25.py` | Turkish tokenizer + Okapi BM25 (`k1=1.5`, `b=0.75`). Limitations documented in the docstring. |
| `corpus.py` | Loads `output/chunk/*.json`; builds the `doc_id` and `(doc_id, paragraph) → chunk_id` maps. |
| `verify_storage.py` | Stage 0. Independent of the generator. |
| `make_worksheet.py` | Writes `worksheet/` — raw numbered paragraphs for query authoring. |
| `queries.example.json` | Format reference. Never loaded; contributes no scored query. |
| `test_retrieval.py` | Audit, retrieval, controls, report. |
