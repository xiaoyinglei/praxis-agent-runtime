# Agent tool-use reliability benchmark

Overall verdict: **PASSED**

This is a narrow Agent-Computer Interface (ACI) reliability gate.
It is not a general coding or reasoning benchmark. It checks whether a model
can follow a fixed set of controlled, machine-checkable file-tool scenarios.

Synthetic marker values make each result machine-checkable. They are test
fixtures, not examples of realistic user content. The exact instructions and
machine-readable identifiers remain available under Technical evidence.

## Result at a glance

| Model | Scenarios | Trials per scenario | Passed executions | Gate result |
| --- | ---: | ---: | ---: | --- |
| `deepseek-v4-flash` | `5` | `3` | **15/15 passed** | **PASSED** |

A passing execution means both the requested outcome and the scenario's
tool-use rule were satisfied. Counts are taken from the linked report, not
entered by hand.

## What the agent was asked to do

| Scenario | User request | What counts as a pass | Observed result |
| --- | --- | --- | ---: |
| **Read a known file directly** | Open a file whose path is already known and return the value inside it. | Read the specified file with a valid call, return the correct value, and make no invalid tool calls. | **3/3 passed** |
| **Find an unknown file, then read it** | Find which file contains a named symbol, open that file, and return its value. | Use valid search-then-read calls in that order and return the correct value. | **3/3 passed** |
| **Recover when a file was renamed** | Try an old report path and, when it is missing, locate and read the renamed report. | Observe the expected failed read, later read the renamed file successfully, and return its contents. | **3/3 passed** |
| **Resume an edit after approval** | Replace one value in a text file, then confirm that the requested value is present. | Pause for approval, resume, apply the requested edit exactly once, and leave the expected file content. | **3/3 passed** |
| **Stop after a confirmed missing file** | Try a known-missing file once and report that it is unavailable. | Make the required failed read once, issue no other tool calls, and report that the file is unavailable. | **3/3 passed** |

## What this result means

A pass shows that the evaluated model followed the reported small, controlled
tool workflows consistently under this runtime and evaluator.
It does **not** establish broad programming ability, repository-scale planning
quality, or safety outside the tested boundaries. The separate 30-task
coding-agent manifest has only been validated and was not scored in this run.

## Technical evidence

### Evidence provenance

- source_commit: `b9e7ca30f8f469ef29d608661c9f084ae182093a`
- source_tree: `2a2329993aa51cf10ff33bc2f46fba333e721341`
- source_unchanged: `true`
- dirty: `false`
- suite_id: `agent-model-tool-quality-v1`
- suite_revision: `suite_064a22c3535430e148e8`
- evaluator_version: `agent_model_quality_gate_v3`
- measured_at: `2026-08-06T14:52:43.690064+00:00`
- Redacted raw report: [2026-08-06-deepseek-v4-flash.json](../evals/model_quality/runs/2026-08-06-deepseek-v4-flash.json)

### Environment

| Field | Value |
| --- | --- |
| `os` | `Darwin` |
| `os_release` | `25.5.0` |
| `architecture` | `arm64` |
| `python_version` | `3.12.12` |
| `python_implementation` | `CPython` |

### Raw model metrics

#### `deepseek_v4_flash`

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

### Exact evaluated instructions

| Case ID | Capability | Exact instruction sent to the model |
| --- | --- | --- |
| `exact_file_read` | `file_tool_selection` | Read the attached file exact.txt with the most direct file tool. Return exactly QUALITY_GATE_EXACT and no other text. |
| `symbol_search_then_read` | `file_tool_selection` | Find the file containing MODEL_QUALITY_TARGET with search_text, then read that matching file with read_file. Return exactly TARGET_VALUE_731 and no other text. |
| `missing_file_recovery` | `failure_recovery` | First call read_file on input_files/report.txt. That old path may fail. If it fails, recover with the available file tools, locate the renamed report, read it, and return exactly RECOVERED_842 and no other text. |
| `approval_continue` | `approval_continuation` | Change the unique text before_gate to after_gate in the attached approval.txt using apply_patch, then confirm completion. |
| `single_failure_no_retry` | `repeated_failure_control` | Try read_file on input_files/definitely-missing.txt exactly once. If it reports that the file is missing, do not retry the same call and do not use another tool. Return exactly FILE_UNAVAILABLE and no other text. |

### Raw case results

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

### Per-case usage

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

### 30-task coding-agent protocol

- Manifest: `evals/code_agent/benchmark_v1.json`
- Manifest status: **validated only**
- The 30-task protocol was not run as this release gate and has no current score here.
