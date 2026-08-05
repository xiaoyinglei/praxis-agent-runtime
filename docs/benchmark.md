# Praxis model-quality benchmark

> **PENDING — NOT YET MEASURED**

This page is the pre-live evidence template for the current Praxis candidate.
It contains the committed method and reporting contract, but no model-quality
result. The live runner and renderer replace this page only after a clean source
commit passes the deterministic local gates. Until then, there is no PASS score.

## Scope

- Model alias: `groq_gpt_oss_120b`
- Provider: `groq`
- Provider model: `openai/gpt-oss-120b`
- Repetitions: **5 cases × 3 trials**
- Evaluator: the committed model-quality gate and its versioned suite identity

The five capabilities are:

1. exact file read;
2. search then read;
3. missing-file recovery;
4. approval continuation and workspace mutation;
5. repeated-failure control.

Each trial exercises the public Praxis Agent path in an isolated workspace. A
case passes only when its task-specific evaluator accepts the runtime result.
Infrastructure failure is **INCONCLUSIVE** and is not converted into PASS or a
model-quality score.

## Evidence contract

The generated report records these exact fields or projections:

| Evidence field | Required meaning |
| --- | --- |
| source commit | Clean Git commit containing the runtime and evaluator used |
| source tree | Fingerprint bound to the committed source before and after provider calls |
| UTC timestamp | When the measured run started |
| model identity | Alias, provider, and provider model |
| environment | Redacted platform/runtime facts; no credentials or local absolute paths |
| task success | Per-trial evaluator outcome plus overall rate |
| capability rates | Aggregation by the five named capabilities |
| tool calls | Counts and a bounded, redacted trace |
| model calls | Per-trial and aggregate request counts |
| latency | Per-trial and aggregate elapsed time |
| token usage | Available prompt, completion, and total usage without secret metadata |
| failures | Evaluator failures and bounded diagnostic categories |
| infrastructure status | Provider, runtime, or environment failure recorded as INCONCLUSIVE |
| workspace mutation evidence | A validated fixture before/after assertion contract; not a captured filesystem diff |

The checked-in machine-readable evidence will live under
`evals/model_quality/runs/` and is rendered into this page plus the expanded
[approval-continuation record](runs/groq-gpt-oss-120b.md). The renderer supplies
the concrete filename only after the actual run; this pending page does not link
to a nonexistent or invented JSON artifact.

## Wider coding protocol

The repository also contains a 30-task coding-agent protocol at
`evals/code_agent/benchmark_v1.json`. Its manifest is **manifest validated**,
but that protocol was **not a release gate** for this change and was
not measured as part of this five-case run. Historical partial or failed results
must not be reused as a current score.

## Publication rule

The live result may be PASS, FAIL, or INCONCLUSIVE. The rendered page must retain
the measured source commit, suite revision, evaluator version, thresholds,
observations, and raw-report link so the verdict can be audited. Secrets,
provider headers, absolute temporary paths, and unnecessary model content are
redacted before any evidence is committed.
