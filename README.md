# llm_chunk — LLM chunking pipeline (docs §15)

One Gemini 2.5 Flash-Lite call per document produces `chunks[]` +
`reasoning_capsules[]` in the same schema as the regex pipeline, but with real
Turkish `reasoning_summary` text and native legislation extraction.

## Run it

```powershell
cd C:\Users\NITRO\Desktop\Hammurabi\llm_chunk\scripts\llm_chunk
python chunk_generate.py
```

That processes `DOCS_PER_SOURCE` (default **2**) documents from each of the 5
sources. Other forms:

```powershell
python chunk_generate.py --source kvkk            # one source
python chunk_generate.py --source aym --limit 1   # one document
```

Output lands in `llm_chunk/output/chunk/{source}.json`, with flagged documents in
`{source}_review.json`.

## Self-contained

Nothing here reads anything outside `llm_chunk/`. Verified two ways: a static grep
for `Chunk_test_newdata|llm_test|../../..` across `scripts/` returns nothing, and a
runtime trace of every file opened shows 0 reads touching the sibling projects.

The proven splitting/extraction logic was **copied verbatim** into
`scripts/llm_chunk/chunk_lib.py` rather than imported or rewritten — 8 functions
plus 11 module objects, the complete dependency closure, computed mechanically.
Equivalence was checked against the original: every function source identical,
every regex identical, and 0 behavioural mismatches across 200 real documents
(595 citations, 1,257 split pieces).

The only path pointing outside is `GOOGLE_APPLICATION_CREDENTIALS` in `.env` — your
service account key, which should live outside the project folder.

## Layout

| Path | What |
|---|---|
| `.env` | Credentials + config. Key **path** only, never its contents. |
| `data/*.json` | Raw corpus, 200 records per source. |
| `fewshot/*.json` | The 5 hand-verified gold documents, used as worked examples. |
| `scripts/llm_chunk/chunk_generate.py` | The pipeline. |
| `scripts/llm_chunk/chunk_lib.py` | Verbatim-copied proven logic. Do not "clean up". |
| `scripts/llm_chunk/prompts.py` | System instructions + few-shot construction. |
| `output/chunk/` | Generated output and review sidecars. |

Runs on system Python 3.11 (`google-genai` 1.59.0, `pydantic`, `bs4`, `python-dotenv`
already installed). No virtualenv.

## Field ownership

Code owns what is arithmetic, a fixed lookup, or must be byte-identical across
separate API calls: `chunk_id` (`uuid.uuid5`, Qdrant-native from the start),
`canonical_id`, `char_length`, `citation_granularity`, `chunk_label`, `subject_type`,
`source_type`, and `reasoning_summary_method` (forced to `llm_generated`, discarded if
the model supplies it — a system does not self-certify its own audit trail).

Gemini owns the reading and judgement: roles, `content_type`, `reasoning_stage`,
`rights` (narrowed from a code-supplied candidate list), `confidence`, the
`cited_legislations` array, and every capsule field except the two above.

**One deviation from the original plan.** `text` is assembled by code from the
paragraphs Gemini referenced, not copied from Gemini's own `text` field. Asked to
reproduce a 5,411-char merge verbatim, the model silently dropped two characters
mid-word at a paragraph boundary, and elsewhere lowercased headings (`DAVA` → `Dava`).
Since `paragraph_refs` already tells us which source paragraphs belong together,
joining them in code makes grounding hold **by construction** instead of being
detected after the fact. Gemini still owns the judgement (which paragraphs group);
only the mechanical assembly moved. Divergence between the model's copy and the real
join is still recorded — large deltas as a `model_text_differs` flag, small ones as a
counted `minor text diffs` statistic — so it is surfaced, never silently resolved.

`canonical_id` uses the new `{law_no}/{article_no}` target format (docs §13.6/§15.5),
falling back to `constitution/{article}` and then a slugified law name for regulations
and directives, which have no law number at all.

## Reading the output

Console summary per source:

```
=== aym ===  2 docs | ok 1 | flagged 1 | failed 0 | skipped_empty 0 | 34 chunks | 3 capsules
  legislation: gemini-only 1, regex-only 5
  minor text diffs (model copy vs source join; stored text unaffected): 6
  tokens: in 35197, out 22395
```

`flagged` does not mean broken — it means a cross-check disagreed and a human should
look. Flags seen on the current 10-document run:

| Flag | Meaning |
|---|---|
| `case_no_mismatch` | Gemini read a **well-formed but different** case number than the metadata says. **Real data problem** on `first_degree/442551100`: DB columns say esas `2017/395`, karar `2018/921`; the document text says `2016/1359` / `2018/919`. |
| `case_no_unreadable_in_source` | Gemini returned something that is not a case number at all, because the document redacts its own. Kept separate from the above — see below. |
| `verbatim_mention_not_in_text` | A citation whose quoted text isn't in the segment. The citation is dropped rather than stored ungrounded. |
| `outcome_vs_ruling_text` | The capsule's outcome label disagrees with the Turkish ruling pattern in the ruling chunk. Flagged, never auto-resolved (docs §15.6). |
| `model_text_differs` | Model's copy diverged from the real paragraph join by enough to suggest a missed `paragraph_refs` entry. |
| `content_type_role_inconsistent` | `content_type: ruling` with a role that isn't a ruling role (including a null role). |
| `unknown_legislation_type` | A `legislation_type` outside the five documented values. **See the schema gap below.** |
| `citation_not_identifiable` | No `law_no`, not the constitution, and no `law_name` either — `canonical_id` is null because there is genuinely nothing to join on. |
| `confidence_contradiction` | `article_no` present but `confidence: low` (docs §15.6). |

| `reasoning_summary_not_turkish` / `conclusion_sentence_not_turkish` | Either capsule free-text field came back in English. |

Current distribution over the 10 documents (34 flag instances, 0 hard failures):
`verbatim_mention_not_in_text` 8, `case_no_mismatch` 8 (all from that one
`first_degree` record), `content_type_role_inconsistent` 7,
`confidence_contradiction` 4, `model_text_differs` 2, `outcome_vs_ruling_text` 2,
`citation_not_identifiable` 2, `capsule_case_no_mismatch` 1.

**Counts move between runs even with `temperature=0`, because the prompt changes.**
Determinism holds for an *unchanged* prompt against the same document — two
back-to-back runs were byte-identical — but any prompt edit re-segments the documents,
and chunk counts follow. Chunk count is `sum of size-cap pieces per model segment`;
paragraph extraction, the 2000 cap and the splitter are pure code and provably
unchanged (aym 120/90 paragraphs, first_degree 25/20, identical across every run), so
when the total moves, segmentation moved — not processing. On the latest run aym is 16
model segments → 35 chunks (9 segments split by the cap) and first_degree 11 → 15.
Compare flag *types* across runs, not counts.

### Both capsule text fields are Turkish

`reasoning_summary` **and** `conclusion_sentence` are required to be Turkish, and both
are checked. Stage 1 retrieval searches the capsule's text against Turkish queries, so
English in either field is dead weight — the exact failure docs §7.3 caught only via a
retrieval test.

This **deliberately diverges from docs §6**, which specifies a "plain-English answer"
for `conclusion_sentence`, and from all five `fewshot/` gold files, whose
`conclusion_sentence` is in English. The prompt therefore tells the model explicitly
not to copy the worked example on that one field. If the docs are updated, §6 is the
line to change.

The language check does not rely on Turkish-specific characters alone: diacritic-free
Turkish is common in scraped legal text ("Bankanin kusuru ispatlanamadi ve dava
reddedildi"), and a one-sentence field is too short for their absence to mean anything.
An English-stopword ratio decides against; diacritics *or* common Turkish words decide
for. Verified both directions — all 12 previously-English sentences flagged, and
diacritic-free Turkish accepted.

### Redacted case numbers are not a misread

Turkish court decisions are published with identifying details replaced by `...` —
names, notary numbers, plate numbers, and **the case-number lines themselves**.
`bam/726546200` begins:

```
DOSYA NO : ...
KARAR NO : ...
```

with 43 such redactions in total, the most of any bam document. On one run Gemini
reported `case_no: "..."` on all 7 segments — reading the document truthfully rather
than echoing the value supplied to it. That is a different failure mode from
`first_degree/442551100`, where the model read a genuine alternate case number, and
lumping them under one flag hid the distinction.

Two changes: the prompt now states that a redaction is not a value and the supplied
number must be used when the document's own header is redacted (while still reporting a
genuinely different *real* number); and the flag splits on whether the returned value
matches `\d{4}/\d+`, so an unreadable source never masquerades as a data conflict.
After the fix that document carries `2019/2035` on all 11 chunks and no literal `...`
appears as a `case_no` anywhere in the output.

### A documented schema gap worth a decision

`legislation_type` has five documented values (docs §13.1) and none of them covers
**international treaties**, which AYM cites constantly. On these two aym documents the
model reached for `treaty` (European Convention on Human Rights, Protocol 7) and
`other` (a European Court of Human Rights judgment). Both are flagged as
`unknown_legislation_type` rather than silently accepted.

The ECHR judgment is correctly *not* legislation — case law does not belong in
`cited_legislations`, and the prompt now says so. The Convention itself is a different
matter: it is a real legal instrument the court applies directly, and the schema has
nowhere to put it. Adding `treaty` to `legislation_type` would close that; leaving it
out means those citations keep getting dropped. That is your call, not something this
script should decide quietly.

## Determinism

`temperature=0`, `seed=42`, `thinking_budget=0`. Two consecutive `--source aym
--limit 1` runs produced **byte-identical** output (19 chunks, same `chunk_id`s, same
702-char `reasoning_summary`). That is an empirical result on this model, not a
provider guarantee — no LLM vendor promises reproducible sampling. `chunk_id` depends
only on `doc_id` + paragraph range, so it stays stable even if wording drifts.

## Out of scope here

The full validation harness (corpus-wide `chunk_id` uniqueness, schema conformance
across every field, contradiction checks) is separate infrastructure, per the spec.
`scripts/retrieval_test/` is intentionally empty. No embeddings, no Qdrant, no
Meilisearch — JSON output only.

## Before scaling up

- aym documents run up to 184K characters. The current 10-document run costs roughly
  **108K input / 56K output tokens ≈ $0.033**. The full 1,000-document corpus
  extrapolates to roughly **$3–4**, not a blocker, but worth measuring rather than
  assuming.
- The regex cross-check found **7 citations in aym that Gemini missed**, while Gemini
  found **16 in bam** that regex missed (mostly the Yönetmelik/Yönerge and
  cross-sentence cases regex structurally cannot reach). Neither side dominates, which
  is exactly why docs §15.7 keeps the regex pipeline running permanently.
- Docs §13.6 called regulations/directives "the single largest known gap" because they
  have no law number. Those now come through with joinable keys, e.g.
  `rekabeti-sınırlayıcı-anlaşma-...-yönetmelik/5` and `/6`.
