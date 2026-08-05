# Groq GPT-OSS 120B: approval continuation

> **PENDING — NOT YET MEASURED**

This is the pre-live template for the expanded `approval_continuation` record.
It does not contain a trial, verdict, source commit, metric, or raw-report link.
The committed renderer replaces it after the real run on the final clean
candidate.

## Model identity

| Field | Value |
| --- | --- |
| Alias | `groq_gpt_oss_120b` |
| Provider | `groq` |
| Provider model | `openai/gpt-oss-120b` |
| Measurement status | **PENDING — NOT YET MEASURED** |

## Expanded evidence contract

The selected completed trial must render all of the following from the redacted
machine-readable report:

- **task** — the exact bounded workspace objective;
- **redacted tool trace** — ordered tool choices and bounded results;
- **approval event** — the destructive capability request and decision;
- **before/after diff** — the validated workspace mutation without local paths;
- **final answer** — the model's bounded completion response;
- **evaluator verdict** — task-specific acceptance plus any failure reasons;
- **raw JSON** — a relative link to the actual redacted report created by the
  run under `evals/model_quality/runs/`.

The record also identifies the trial index, runtime stop reason, verification
evidence, tool/model call counts, latency, token usage when the provider reports
it, and infrastructure status. No value is filled from memory or a historical
run.

If provider, environment, or runtime infrastructure prevents a conclusive
trial, the page must say **INCONCLUSIVE** and show the bounded failure stage. It
must not fabricate the expanded success trace or downgrade infrastructure
failure into a model-quality failure.
