# `retrieval/` — the contract with Hammurabi, and the harness that proves it

Two deliverables, not one:

1. **A contract** — the smallest additive change set Hammurabi needs, plus the exact
   record shape we must produce to satisfy it.
2. **A harness** — a faithful local replica of Hammurabi's retrieval, upgraded only
   where our chunks require it, proving chunks retrieve better than whole documents.

Nothing here writes to Hammurabi, contacts production, or regenerates chunks.

## Design rule: additive only

Every change we ask Hammurabi for must be **additive** (new collection, new index,
new code path), **flagged** (default off), **reversible** (flag off ⇒ byte-identical
to today) and **provable** (a test shows the whole-document path is unchanged).
A change that cannot be additive is not a change — it is a negotiation, and it goes
to the open-questions list instead.

## Status

| Phase | What | State |
|---|---|---|
| 1 | Record contract + validator | **done** — 142/142 chunks pass |
| 2 | Embedding + both index modules | not started |
| 3 | Index both arms | not started |
| 4 | Agent path at real constants | not started |
| 5 | Hybrid path incl. RRF + procedural penalty | not started |
| 6 | A/B harness, experiments E1–E9 | not started |

## Files

| File | Purpose |
|---|---|
| `contract.py` | The record shape, the derivations, and the validator. |
| `to_index_records.py` | Our chunks → index records, with a full report. Free to run. |
| `report/index_records.json` | The 142 converted records. |

```bash
python to_index_records.py            # validate and report
python to_index_records.py --write    # also write report/index_records.json
python to_index_records.py --show 1   # print one full record
```

## What Phase 1 established

**Our chunks and Hammurabi's payload shared zero field names.** The contract closes
that gap by deriving production's twelve fields from the raw court record we already
load, and adds the six that make a chunk a chunk (`parent_uuid`, `paragraph_ids`,
`chunk_index`, `unit`, `schema_version`, `role`).

Three problems in the real data were found and handled. Each would have failed
silently in production:

| Problem | Effect if unhandled | Handling |
|---|---|---|
| **kvkk has no `esas_year`** | Production splits every search into `esas_year >= 2023` and `< 2023`. A null matches **neither**, so kvkk would be invisible to the whole semantic path. | Fall back to `karar_year`, recorded in `derivation_notes`. |
| **aym has no `karar_year`** | Mirror of the above. | Fall back to `esas_year`, recorded. |
| **`chamber_id` is not the chamber** | It holds the real chamber only for yargitay (5). bam stores 588, danistay 125, first_degree 25783 — internal ids that would never match a chamber filter. | Parse the chamber from the **title**; ignore `chamber_id`. Verified: bam 11/6, danistay 13, first_degree 4/7, yargitay 5. |

Also fixed: for institutions with no chamber structure, `court` fell back to the
title — which for aym is an **applicant's name** and for kvkk a case summary. Since
`court` is a *filterable* field, that is both useless and a privacy smell. Those
sources now use the institution's own name.

## Two values are guesses

Isolated in one function each, so confirming them is a one-line change:

- **`filename`** — production uses `__280908200.txt`; our `doc_id` is `1203538100`.
  Plausible, unverified.
- **`parent_uuid`** — must equal production's uuid for the same decision; their
  generation scheme is unknown. Minted deterministically so re-indexing overwrites
  rather than duplicates.

## The one field rename we propose

Our role field is named differently per source — `firac_role`,
`court_reasoning_role`, `regulatory_role`. A field whose **name** varies by row
cannot be indexed or filtered as one thing; you would need three filterable
attributes and three query branches forever to ask "show me only the reasoning".

The record therefore carries a single **`role`** plus **`role_vocabulary`** naming
where it came from. The per-source fields stay in our own output untouched, so
nothing downstream breaks and no information is lost.

## Open questions for the team that owns indexing

1. How is a decision's `uuid` generated? Decides overwrite vs duplicate.
2. What is the `filename` convention exactly?
3. `decisions_bge`'s real vector size and distance metric — not declared anywhere
   in the code we can read.
4. Can a second collection and index be provisioned, and by whom?
5. Is the four-court enum (`bam`, `yargitay`, `danistay`, `first_degree`) deliberate?
   The agent cannot filter to the Constitutional Court or KVKK today, and the tool
   description tells the model to default to `yargitay`.

## Known limits

Only 12 documents exist as chunks, so the harness can measure chunk advantage only
on those; the remaining ~1,025 documents are honest distractors, not test subjects.
Our own documentation calls for 20–30 per source before this counts as validated.
