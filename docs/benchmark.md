# Model quality benchmark

Overall verdict: **FAILED**

## Evidence provenance

- source_commit: `48e5eb0191ff2279e40cb294ffcf845a8a016e13`
- source_tree: `94c353fe62ef7f09157eeca883a351f1db42df52`
- source_unchanged: `true`
- dirty: `false`
- suite_id: `agent-model-tool-quality-v1`
- suite_revision: `suite_064a22c3535430e148e8`
- evaluator_version: `agent_model_quality_gate_v3`
- measured_at: `2026-08-06T09:43:16.967520+00:00`
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
| `mean_model_calls_per_case` | `3.2` | `max 3.0` |
| `mean_tool_calls_per_case` | `2.4` | `max 2.0` |
| `redundant_tool_call_rate` | `0.0` | `max 0.0` |
| `repeated_failure_control_rate` | `1.0` | `min 1.0` |
| `task_success_rate` | `0.8` | `min 1.0` |

Reported failures:

- task_success_rate: observed 0.8 < baseline floor 1.0
- mean_tool_calls_per_case: observed 2.4 > baseline ceiling 2.0
- mean_model_calls_per_case: observed 3.2 > baseline ceiling 3.0

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
| `deepseek_v4_flash` | `2` | `approval_continue` | `approval_continuation` | **FAILED** |
| `deepseek_v4_flash` | `2` | `single_failure_no_retry` | `repeated_failure_control` | **PASSED** |
| `deepseek_v4_flash` | `3` | `exact_file_read` | `file_tool_selection` | **PASSED** |
| `deepseek_v4_flash` | `3` | `symbol_search_then_read` | `file_tool_selection` | **PASSED** |
| `deepseek_v4_flash` | `3` | `missing_file_recovery` | `failure_recovery` | **PASSED** |
| `deepseek_v4_flash` | `3` | `approval_continue` | `approval_continuation` | **FAILED** |
| `deepseek_v4_flash` | `3` | `single_failure_no_retry` | `repeated_failure_control` | **PASSED** |

## Per-case usage

| Model | Trial | Case | Tool calls | Model calls | Latency ms | Input tokens | Output tokens | Total tokens |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `deepseek_v4_flash` | `1` | `exact_file_read` | `1` | `2` | `5438.894835300744` | `7724` | `190` | `7914` |
| `deepseek_v4_flash` | `1` | `symbol_search_then_read` | `2` | `3` | `6793.128416873515` | `12548` | `321` | `12869` |
| `deepseek_v4_flash` | `1` | `missing_file_recovery` | `4` | `4` | `9200.436001177877` | `17140` | `576` | `17716` |
| `deepseek_v4_flash` | `1` | `approval_continue` | `4` | `5` | `20008.09920905158` | `22256` | `2111` | `24367` |
| `deepseek_v4_flash` | `1` | `single_failure_no_retry` | `1` | `2` | `2900.5583750549704` | `7450` | `220` | `7670` |
| `deepseek_v4_flash` | `2` | `exact_file_read` | `1` | `2` | `3012.360041961074` | `7770` | `171` | `7941` |
| `deepseek_v4_flash` | `2` | `symbol_search_then_read` | `2` | `3` | `7404.86199897714` | `12352` | `215` | `12567` |
| `deepseek_v4_flash` | `2` | `missing_file_recovery` | `3` | `3` | `5989.8909570183605` | `12380` | `384` | `12764` |
| `deepseek_v4_flash` | `2` | `approval_continue` | `4` | `5` | `22608.967332169414` | `22140` | `1797` | `23937` |
| `deepseek_v4_flash` | `2` | `single_failure_no_retry` | `1` | `2` | `5518.3193339034915` | `7468` | `210` | `7678` |
| `deepseek_v4_flash` | `3` | `exact_file_read` | `1` | `2` | `3058.431833051145` | `7700` | `148` | `7848` |
| `deepseek_v4_flash` | `3` | `symbol_search_then_read` | `2` | `3` | `4610.109667060897` | `12508` | `290` | `12798` |
| `deepseek_v4_flash` | `3` | `missing_file_recovery` | `3` | `4` | `6674.206709023565` | `16768` | `514` | `17282` |
| `deepseek_v4_flash` | `3` | `approval_continue` | `3` | `4` | `16849.822042277083` | `16543` | `1378` | `17921` |
| `deepseek_v4_flash` | `3` | `single_failure_no_retry` | `1` | `2` | `2859.4645420089364` | `7431` | `125` | `7556` |

## 30-task coding-agent protocol

- Manifest: `evals/code_agent/benchmark_v1.json`
- Manifest status: **validated only**
- The 30-task protocol was not run as this release gate and has no current score here.
