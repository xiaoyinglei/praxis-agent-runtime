# Non-code data ACI model run

Overall verdict: **CONCLUSIVE FUNCTIONAL PASS — 3/3 artifacts independently accepted**

This is one real, combined model run across an Excel workbook, a PDF, and a
2,000-row CSV. It establishes that the data-task ACI can complete this exact
multi-artifact workflow. It is not a repeated-trial reliability benchmark.

## Provenance

- source_commit: `8c272c1f7c27b952a001000c0ca69acc8bf008ec`
- source_tree: `5c07f85bda3ea895abc3aadedc54926f0a25cb2a`
- source_unchanged_during_run: `true`
- model_alias: `deepseek_v4_flash`
- provider_model: `deepseek-v4-flash`
- turn_id: `3d460a63-21f7-4b0c-a855-9dc146e5e26a`
- runtime: macOS Darwin 25.5.0 arm64, CPython 3.12.12
- execution boundary: managed Python under macOS Seatbelt, network disabled
- workspace condition: isolated data directory with no workspace `.venv`

## Task in plain language

The model had to finish all three deliverables in one Turn:

1. Aggregate regional net revenue from `sales.xlsx`, join targets, preserve
   full numeric precision, and write a sorted `excel_summary.xlsx` workbook.
2. Extract operating metrics and source-page attribution from
   `operations.pdf`, compute two derived percentages, and write
   `pdf_findings.json` with an exact schema.
3. Read `ab_experiment.csv`, calculate a two-sided two-proportion z-test and
   95% confidence interval using the specified standard errors and
   `variant - control` direction, then write `ab_result.json`.

The task required one structured read-back of every generated artifact and
immediate completion after verification.

Input SHA-256 values were:

| Input | SHA-256 |
| --- | --- |
| `sales.xlsx` | `3793e915fa700f7ad8b4eb8a0d11aa676da65a571aaa3c0e687dc35440f7efd2` |
| `operations.pdf` | `d84aefc9039f1b394cfe99578e36fa12d4ebc67b7b1e70a3c10f1b564ebe185a` |
| `ab_experiment.csv` | `8d6f14eb38358fd59b7d35ebe9bfbc9b08565ce248b6987f2f404f5ea5fc7b32` |

## Observed runtime trace

| Step | Tool | Result |
| ---: | --- | --- |
| 1 | `list_files` | Succeeded; unnecessary because all paths were already explicit |
| 2–4 | `inspect_data_file` | All three inputs parsed successfully |
| 5 | `execute_python` | Rejected before execution because the model emitted malformed arguments |
| 6 | `execute_python` | Paused for write approval, resumed once with `allow_once`, then generated all three declared outputs successfully |
| 7–9 | `inspect_data_file` | All three exact output paths returned `valid=true` with runtime SHA-256 receipts |

- final Turn status: `completed`
- loop stop reason: `accepted`
- wall-clock time reported by the CLI: `40,746 ms`
- tool attempts: `9` total, `8` successful, `1` invalid argument attempt
- approval resumes: `1`
- undeclared shell execution: `0`
- repeated output inspections: `0`

The malformed first Python call matters: this run proves recovery and eventual
task completion, not perfect first-attempt argument validity.

## Independent acceptance

A separate deterministic verifier reopened the files after the Agent stopped.
It did not rely on the model's final answer or on `valid=true` alone.

| Artifact | Checks | Result |
| --- | --- | --- |
| `excel_summary.xlsx` | Exact sheet/header contract, row order, revenue totals, targets, and unrounded attainment values | **PASS** |
| `pdf_findings.json` | Exact keys, page mapping, extracted facts, and both derived percentages | **PASS** |
| `ab_result.json` | Counts, rates, lift, pooled-SE z statistic, p-value, unpooled-SE interval, and decision | **PASS** |

Acceptance result: **3/3 passed**.

Key checked values:

- Excel: East `3135 / 3000 / 1.045`, North `2630 / 2500 / 1.052`,
  West `2220 / 2300 / 0.9652173913043478`.
- PDF: revenue `8.4`, on-time delivery `96.2`, refund rate `1.8`,
  non-refund rate `98.2`, gap `2.0`, risk corridor `East Harbor`, delayed
  shipments `37`.
- Statistics: rates `0.12` and `0.16`, z statistic
  `2.577696311132336`, two-sided p-value `0.009946136752047764`, and
  difference interval `[0.009636368514840166, 0.07036363148515985]`.

Output receipts:

| Output | Bytes | SHA-256 |
| --- | ---: | --- |
| `excel_summary.xlsx` | 4,987 | `fe275b56379da661f0815b4beb6db343238e41df7a2426b80b9817fb664bd57d` |
| `pdf_findings.json` | 342 | `2f6c92fe1c7016bce28cc447314277e5df99e5ce55a453afaf68b5df806a6614` |
| `ab_result.json` | 447 | `c6748df058a1898c990811de3c5aaccb3b1dacad71ba701b3bdcfe7ba3673af0` |

## Multi-turn recovery finding

An earlier live attempt exposed a separate checkpoint bug: a reasoning model
that exhausted its output limit before emitting visible content left an empty
assistant message, and an OpenAI-compatible provider rejected that message on
resume. Source commit `8c272c1f` keeps that pause observable while excluding
the non-replayable empty message from canonical history. The regression test
`test_max_tokens_reasoning_only_is_not_persisted_as_empty_assistant_message`
passes. The conclusive run above did not itself hit the token-limit path, so the
post-fix claim for that edge case is automated regression evidence, not a
second live-model reliability result.

## What this does not establish

- It is one functional trial, not a repeated quality gate or a population-level
  model score.
- The exact formulas, schema, precision, and sign convention were explicit.
  The run does not prove reliable behavior under ambiguous business requests.
- `inspect_data_file` proves parseability, bounded content visibility, and file
  identity. Semantic correctness comes from the independent acceptance checks.
- The Seatbelt evidence is for trusted-local macOS execution, not a multi-tenant
  remote sandbox.
