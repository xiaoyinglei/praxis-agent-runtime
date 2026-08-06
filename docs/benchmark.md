# Model quality benchmark

Overall verdict: **FAILED**

## Evidence provenance

- source_commit: `1e4c16873f4d6983397d5965b6d527c09d680689`
- source_tree: `3bb88769e62b6afa7f47ef8c7db48cd29d9d32b5`
- source_unchanged: `true`
- dirty: `false`
- suite_id: `agent-model-tool-quality-v1`
- suite_revision: `suite_064a22c3535430e148e8`
- evaluator_version: `agent_model_quality_gate_v3`
- measured_at: `2026-08-06T01:12:04.688740+00:00`
- Redacted raw report: [2026-08-06-deepseek-v4-flash.json](../evals/model_quality/runs/2026-08-06-deepseek-v4-flash.json)

## Environment

| Field | Value |
| --- | --- |
| `os` | `Darwin` |
| `os_release` | `25.5.0` |
| `architecture` | `arm64` |
| `python_version` | `3.12.12` |
| `python_implementation` | `CPython` |

## Model results

### `deepseek_v4_flash`

- Provider: `deepseek`
- Provider model: `deepseek-v4-flash`
- Trials: `3`
- Infrastructure status: **CONCLUSIVE**
- Evaluator verdict: **FAILED**

| Metric | Observed | Threshold |
| --- | ---: | --- |
| `approval_continuation_rate` | `1.0` | `min 1.0` |
| `argument_validity_rate` | `1.0` | `min 1.0` |
| `failure_recovery_rate` | `1.0` | `min 1.0` |
| `file_tool_selection_rate` | `1.0` | `min 1.0` |
| `mean_model_calls_per_case` | `2.8` | `max 3.0` |
| `mean_tool_calls_per_case` | `2.0` | `max 2.0` |
| `redundant_tool_call_rate` | `0.0` | `max 0.0` |
| `repeated_failure_control_rate` | `1.0` | `min 1.0` |
| `task_success_rate` | `0.8` | `min 1.0` |

Reported failures:

- task_success_rate: observed 0.8 < baseline floor 1.0

## Case results

| Model | Trial | Case | Capability | Verdict |
| --- | ---: | --- | --- | --- |
| `deepseek_v4_flash` | `1` | `exact_file_read` | `file_tool_selection` | **PASSED** |
| `deepseek_v4_flash` | `1` | `symbol_search_then_read` | `file_tool_selection` | **PASSED** |
| `deepseek_v4_flash` | `1` | `missing_file_recovery` | `failure_recovery` | **PASSED** |
| `deepseek_v4_flash` | `1` | `approval_continue` | `approval_continuation` | **FAILED** |
| `deepseek_v4_flash` | `1` | `single_failure_no_retry` | `repeated_failure_control` | **PASSED** |
| `deepseek_v4_flash` | `2` | `exact_file_read` | `file_tool_selection` | **PASSED** |
| `deepseek_v4_flash` | `2` | `symbol_search_then_read` | `file_tool_selection` | **PASSED** |
| `deepseek_v4_flash` | `2` | `missing_file_recovery` | `failure_recovery` | **PASSED** |
| `deepseek_v4_flash` | `2` | `approval_continue` | `approval_continuation` | **PASSED** |
| `deepseek_v4_flash` | `2` | `single_failure_no_retry` | `repeated_failure_control` | **PASSED** |
| `deepseek_v4_flash` | `3` | `exact_file_read` | `file_tool_selection` | **PASSED** |
| `deepseek_v4_flash` | `3` | `symbol_search_then_read` | `file_tool_selection` | **PASSED** |
| `deepseek_v4_flash` | `3` | `missing_file_recovery` | `failure_recovery` | **PASSED** |
| `deepseek_v4_flash` | `3` | `approval_continue` | `approval_continuation` | **PASSED** |
| `deepseek_v4_flash` | `3` | `single_failure_no_retry` | `repeated_failure_control` | **PASSED** |

## Per-case usage

| Model | Trial | Case | Tool calls | Model calls | Latency ms | Input tokens | Output tokens | Total tokens |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `deepseek_v4_flash` | `1` | `exact_file_read` | `1` | `2` | `5981.747125042602` | `7677` | `141` | `7818` |
| `deepseek_v4_flash` | `1` | `symbol_search_then_read` | `2` | `3` | `5524.620373966172` | `12540` | `279` | `12819` |
| `deepseek_v4_flash` | `1` | `missing_file_recovery` | `2` | `3` | `6131.022706860676` | `11957` | `248` | `12205` |
| `deepseek_v4_flash` | `1` | `approval_continue` | `4` | `4` | `16269.783834228292` | `16626` | `1358` | `17984` |
| `deepseek_v4_flash` | `1` | `single_failure_no_retry` | `1` | `2` | `3542.033873964101` | `7457` | `217` | `7674` |
| `deepseek_v4_flash` | `2` | `exact_file_read` | `1` | `2` | `3063.550000777468` | `7700` | `119` | `7819` |
| `deepseek_v4_flash` | `2` | `symbol_search_then_read` | `2` | `3` | `5957.819832721725` | `12551` | `287` | `12838` |
| `deepseek_v4_flash` | `2` | `missing_file_recovery` | `2` | `3` | `5213.307332247496` | `11906` | `207` | `12113` |
| `deepseek_v4_flash` | `2` | `approval_continue` | `3` | `4` | `9354.696249822155` | `16523` | `477` | `17000` |
| `deepseek_v4_flash` | `2` | `single_failure_no_retry` | `1` | `2` | `4130.67916687578` | `7441` | `192` | `7633` |
| `deepseek_v4_flash` | `3` | `exact_file_read` | `1` | `2` | `3508.5549589712173` | `7779` | `234` | `8013` |
| `deepseek_v4_flash` | `3` | `symbol_search_then_read` | `2` | `3` | `5079.446041956544` | `12602` | `278` | `12880` |
| `deepseek_v4_flash` | `3` | `missing_file_recovery` | `2` | `3` | `5209.85666802153` | `12030` | `346` | `12376` |
| `deepseek_v4_flash` | `3` | `approval_continue` | `3` | `4` | `7353.910793084651` | `16469` | `556` | `17025` |
| `deepseek_v4_flash` | `3` | `single_failure_no_retry` | `1` | `2` | `4289.961458882317` | `7456` | `394` | `7850` |

## 30-task coding-agent protocol

- Manifest: `evals/code_agent/benchmark_v1.json`
- Manifest status: **validated only**
- The 30-task protocol was not run as this release gate and has no current score here.
