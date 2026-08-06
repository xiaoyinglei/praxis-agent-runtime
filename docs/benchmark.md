# Model quality benchmark

Overall verdict: **PASSED**

## Evidence provenance

- source_commit: `b9e7ca30f8f469ef29d608661c9f084ae182093a`
- source_tree: `2a2329993aa51cf10ff33bc2f46fba333e721341`
- source_unchanged: `true`
- dirty: `false`
- suite_id: `agent-model-tool-quality-v1`
- suite_revision: `suite_064a22c3535430e148e8`
- evaluator_version: `agent_model_quality_gate_v3`
- measured_at: `2026-08-06T14:52:43.690064+00:00`
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
| `deepseek_v4_flash` | `1` | `exact_file_read` | `1` | `2` | `8963.368250057101` | `8633` | `168` | `8801` |
| `deepseek_v4_flash` | `1` | `symbol_search_then_read` | `2` | `3` | `3700.890251202509` | `13615` | `168` | `13783` |
| `deepseek_v4_flash` | `1` | `missing_file_recovery` | `2` | `3` | `5224.570082267746` | `13369` | `290` | `13659` |
| `deepseek_v4_flash` | `1` | `approval_continue` | `2` | `3` | `9468.98008394055` | `13497` | `895` | `14392` |
| `deepseek_v4_flash` | `1` | `single_failure_no_retry` | `1` | `2` | `3153.574125142768` | `8342` | `105` | `8447` |
| `deepseek_v4_flash` | `2` | `exact_file_read` | `1` | `2` | `2988.350415835157` | `8592` | `141` | `8733` |
| `deepseek_v4_flash` | `2` | `symbol_search_then_read` | `2` | `3` | `4635.299290996045` | `13829` | `255` | `14084` |
| `deepseek_v4_flash` | `2` | `missing_file_recovery` | `2` | `3` | `4952.284709084779` | `13406` | `284` | `13690` |
| `deepseek_v4_flash` | `2` | `approval_continue` | `2` | `3` | `17483.117500087246` | `13430` | `1705` | `15135` |
| `deepseek_v4_flash` | `2` | `single_failure_no_retry` | `1` | `2` | `3133.7859593331814` | `8344` | `212` | `8556` |
| `deepseek_v4_flash` | `3` | `exact_file_read` | `1` | `2` | `2791.6957908309996` | `8604` | `121` | `8725` |
| `deepseek_v4_flash` | `3` | `symbol_search_then_read` | `2` | `3` | `5343.548043165356` | `13817` | `261` | `14078` |
| `deepseek_v4_flash` | `3` | `missing_file_recovery` | `2` | `3` | `5391.309915808961` | `13482` | `319` | `13801` |
| `deepseek_v4_flash` | `3` | `approval_continue` | `2` | `3` | `18053.717749891803` | `13356` | `1713` | `15069` |
| `deepseek_v4_flash` | `3` | `single_failure_no_retry` | `1` | `2` | `3883.8066251482815` | `8294` | `223` | `8517` |

## 30-task coding-agent protocol

- Manifest: `evals/code_agent/benchmark_v1.json`
- Manifest status: **validated only**
- The 30-task protocol was not run as this release gate and has no current score here.
