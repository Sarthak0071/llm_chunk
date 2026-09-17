# llm_chunk — raw court JSON → chunks, LLM first

Turns one record of `data/{source}.json` into `chunks[]` + `reasoning_capsules[]` in the
fixed shape of *Chunk Structure and Retrieval Architecture.docx* (§4, §6, §13.1).
The model reads; code does only what §15.2–15.4 give to code.

## Run

```powershell
cd C:\Users\NITRO\Desktop\Hammurabi\llm_chunk\scripts\llm_chunk
$env:PYTHONIOENCODING = "utf-8"
python chunker.py --source kvkk --limit 10                        # first N usable documents
python chunker.py --doc-id-file ..\..\output\kvkk_first10.json --out-dir output/chunk   # exact ids, merges by doc_id
python test_chunk.py --verify output/chunk                        # free: every paragraph stored once? values legal?
```

Output: `output/chunk/{source}.json` (chunks + capsules, written after every document) and
`{source}_review.json` (per document: coverage, audit log, notes, the model's raw answer).

## Files

| file | role |
|---|---|
| `scripts/llm_chunk/chunker.py` | the pipeline: read → derive → call → repair → audit → assemble → check → write |
| `scripts/llm_chunk/prompts.py` | generation prompt, per-kind notes, allowed values with Turkish glosses, worked examples from `fewshot/`, audit prompt |
| `scripts/llm_chunk/test_chunk.py` | `--schema`, `--extraction`, `--verify DIR`, `--replay DIR` (free); `--live --audit DIR`, `--live --smoke` (paid) |
| `scripts/llm_chunk/chunk_lib.py` | Turkish upper/lower, size-cap splitter (verbatim helpers, unchanged) |
| `scripts/llm_chunk/chunk_generate_v1.py` | the previous pipeline, kept for reference |
| `.env` | credentials path and `MODEL`; never the key itself |

## What the model owns (docs §15.2–15.4)

Segments (which paragraphs group together, `role`, `confidence`, `rights` from a
candidate list, `cited_legislations`), capsules (`outcome`, `opinion_type` + authors,
`subject_id`, `conclusion_sentence`, `reasoning_summary`, supporting segments). Every
closed field is a `Literal` in the response schema, so an illegal value cannot be
parsed. Then a **second read** (`audit`) receives the paragraphs plus the first pass's
segments and capsules and returns only fixes (wrong role, wrong outcome, missing
disposition); code applies them and logs `audited_role` / `audited_outcome`.

## What code owns

`chunk_id` (uuid5), `chunk_label`, `char_length`, `text` joined from `paragraph_refs`,
size cap, `content_type` and `reasoning_stage` from the role (§14.1), `canonical_id`,
`law_short` normalisation, `case_no`, `decision_date`, `subject_type`, `source_type`,
`citation_granularity`, `reasoning_summary_method`, and the document kind (AYM norm
review / individual application, Yargıtay ceza / hukuk).

Mechanical repairs, never rejections: a paragraph listed twice keeps its first
segment; a one-liner under 40 characters joins its neighbour; a paragraph the model
left out goes into a catch-all chunk (`uncovered_filled`), so **every paragraph is
in exactly one chunk by construction**; a capsule pointer to a segment the model
never wrote is dropped (`dangling_support_dropped`).

Mechanical checks, reported in `_review.json` and by `--verify`, never blocking: a
law or case number in a summary that the decision does not contain; a capsule with
no support; a majority capsule citing dissent chunks; non-Turkish or empty capsule
text. No Turkish phrase tables anywhere.

## Which text column is read

Measured on 60 records per source: bam / danistay / first_degree `content_text` has one
line per paragraph and is read directly. aym / kvkk / yargitay `content_text` is ONE
line with the breaks removed and words glued ("Karar ÖzetiKarar Tarihi:"), so their
`html_content` is walked (block tags and `<br>` end a paragraph, tags stripped). Same
words either way; coverage is checked against the longer column and is 1.000 on every
source (`test_chunk.py --extraction`). The model also receives the plain `content_text`
as a reference copy whenever it holds words the paragraphs do not.

## Vocabularies

English keys for machine values, Turkish for content. `outcome` per kind
(`OUTCOME_BY_KIND` in chunker.py, glosses in `OUTCOME_GLOSS`); `opinion_type` stored as
`majority` | `board_decision` | `dissent:<author>` | `concurring:<author>`; a dissent's
`outcome` is the disposition the dissenter argued for (a purely procedural dissent names the disposition it leads to, e.g. `dismissed_procedural`).
`legislation_type` stored values are the five of §13.1; `treaty` and `not_legislation`
are accepted from the model and dropped with a count.

## Status (September 2026)

Evaluation set of 80 documents, 10 per kind (AYM individual and norm review, bam,
Danistay, first_degree, Yargitay hukuk and ceza, KVKK): 80 stored, 0 failed,
4,740 of 4,740 paragraphs in chunks, 0 duplicated, 0 issues, every value inside the
vocabulary. Remaining notes are informational (named laws not cited on 27 documents,
7 short chunks, 4 dissents on `other`). Pipeline runs 4 documents in parallel, checks
its schemas against the API before sending anything, writes after every document,
and merges targeted reruns by doc_id. Next: the full corpus (~1,160 documents,
~$3, ~1.5 h) with `python chunker.py --source <court> --limit 1000 --out-dir output/full`
per court, then `test_chunk.py --verify output/full` and `--report output/full`.

Lessons that shaped the current code (kept so they are not re-learned): the KVKK
substance lived in `<li>`/`<td>` and a `<p>`-only walk dropped 80% of it; Turkish
phrase tables for roles and outcomes were never complete and every gap rejected a
correct answer, so reading is the model's job and code checks only mechanics; the
second read must be boxed by code (allowed role targets, never removing the last
ruling chunk, never fixing an outcome to `other`) because prompt limits alone do not
hold; a grouping instruction must exempt the disposition or it gets swallowed;
`procedural_objection` as a dissent outcome became a lazy default and was removed;
`maxItems` next to enums makes Gemini reject the whole schema.
