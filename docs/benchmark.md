# Model quality benchmark

Overall verdict: **PASSED**

## Evidence provenance

- source_commit: `38b971a17a25a2fbd80d6fbbe9e313b9e7cf7aab`
- source_tree: `986c46d013e8a9bf77ea7d852c06897cbb60a0bc`
- source_unchanged: `true`
- dirty: `false`
- suite_id: `agent-model-tool-quality-v1`
- suite_revision: `suite_064a22c3535430e148e8`
- evaluator_version: `agent_model_quality_gate_v3`
- measured_at: `2026-08-06T14:31:38.875390+00:00`
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
- Evaluator verdict: **PASSED**

| Metric | Observed | Threshold |
| --- | ---: | --- |
| `approval_continuation_rate` | `1.0` | `min 1.0` |
| `argument_validity_rate` | `1.0` | `min 1.0` |
| `failure_recovery_rate` | `1.0` | `min 1.0` |
| `file_tool_selection_rate` | `1.0` | `min 1.0` |
| `mean_model_calls_per_case` | `2.6` | `max 3.0` |
| `mean_tool_calls_per_case` | `1.6` | `max 2.0` |
| `redundant_tool_call_rate` | `0.0` | `max 0.0` |
| `repeated_failure_control_rate` | `1.0` | `min 1.0` |
| `task_success_rate` | `1.0` | `min 1.0` |

## Case results

| Model | Trial | Case | Capability | Verdict |
| --- | ---: | --- | --- | --- |
| `deepseek_v4_flash` | `1` | `exact_file_read` | `file_tool_selection` | **PASSED** |
| `deepseek_v4_flash` | `1` | `symbol_search_then_read` | `file_tool_selection` | **PASSED** |
| `deepseek_v4_flash` | `1` | `missing_file_recovery` | `failure_recovery` | **PASSED** |
| `deepseek_v4_flash` | `1` | `approval_continue` | `approval_continuation` | **PASSED** |
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
| `deepseek_v4_flash` | `1` | `exact_file_read` | `1` | `2` | `5823.884666198865` | `8571` | `153` | `8724` |
| `deepseek_v4_flash` | `1` | `symbol_search_then_read` | `2` | `3` | `3700.647875899449` | `13753` | `251` | `14004` |
| `deepseek_v4_flash` | `1` | `missing_file_recovery` | `2` | `3` | `4325.591332977638` | `13341` | `297` | `13638` |
| `deepseek_v4_flash` | `1` | `approval_continue` | `2` | `3` | `8086.3622080069035` | `13567` | `829` | `14396` |
| `deepseek_v4_flash` | `1` | `single_failure_no_retry` | `1` | `2` | `1676.8677500076592` | `8258` | `57` | `8315` |
| `deepseek_v4_flash` | `2` | `exact_file_read` | `1` | `2` | `2871.851959032938` | `8626` | `185` | `8811` |
| `deepseek_v4_flash` | `2` | `symbol_search_then_read` | `2` | `3` | `5362.613917095587` | `13859` | `329` | `14188` |
| `deepseek_v4_flash` | `2` | `missing_file_recovery` | `2` | `3` | `4391.472792020068` | `13232` | `247` | `13479` |
| `deepseek_v4_flash` | `2` | `approval_continue` | `2` | `3` | `10807.177040958777` | `13394` | `1025` | `14419` |
| `deepseek_v4_flash` | `2` | `single_failure_no_retry` | `1` | `2` | `4718.672832241282` | `8291` | `442` | `8733` |
| `deepseek_v4_flash` | `3` | `exact_file_read` | `1` | `2` | `2283.1925419159234` | `8558` | `146` | `8704` |
| `deepseek_v4_flash` | `3` | `symbol_search_then_read` | `2` | `3` | `3925.064583076164` | `13671` | `231` | `13902` |
| `deepseek_v4_flash` | `3` | `missing_file_recovery` | `2` | `3` | `5164.285126142204` | `13434` | `354` | `13788` |
| `deepseek_v4_flash` | `3` | `approval_continue` | `2` | `3` | `7651.055208174512` | `13153` | `702` | `13855` |
| `deepseek_v4_flash` | `3` | `single_failure_no_retry` | `1` | `2` | `3370.724498992786` | `8300` | `200` | `8500` |

## 30-task coding-agent protocol

- Manifest: `evals/code_agent/benchmark_v1.json`
- Manifest status: **validated only**
- The 30-task protocol was not run as this release gate and has no current score here.
