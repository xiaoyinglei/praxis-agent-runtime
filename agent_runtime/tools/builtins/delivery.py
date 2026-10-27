"""Model-requested delivery actions executed host-side under normal approvals.

These tools expose ``GitDelivery`` to the model the way Claude Code exposes
``git commit``/``gh pr create`` through the shell: the model requests the
action, the runtime executes it with the user's own identity after the normal
approval mode decision. The sandbox and the model still never touch repository
metadata; delivery follows the same ask/auto/full semantics as every other
tool, and existing ``tools`` permission rules apply unchanged.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import cast

from pydantic import BaseModel, ConfigDict, Field

from agent_runtime.git_delivery import (
    DeliveryError,
    DeliveryInterruptedError,
    DeliveryResult,
    GitDelivery,
    PullRequestResult,
)
from agent_runtime.tools.tool import (
    CancellationMode,
    InterruptBehavior,
    JsonValue,
    NormalizedToolOutput,
    ResolvedToolUse,
    Tool,
    ToolDefinition,
    ToolEffect,
    ToolTarget,
    json_schema_output,
    pydantic_input,
)
from agent_runtime.workspace import WorkspaceRuntime


class CommitChangesInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    message: str = Field(
        min_length=1,
        max_length=1000,
        description=(
            "One concise conventional-commit subject line, for example "
            "'fix: handle empty timeout in config loader'. Not a report; the "
            "diff speaks for itself."
        ),
    )
    branch_name: str | None = Field(
        default=None,
        max_length=200,
        description=(
            "Optional explicit praxis/* branch name. Omit to reuse the "
            "current delivery branch or create a timestamped one."
        ),
    )


class CommitChangesOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    commit_id: str = ""
    branch: str = ""
    changed_file_count: int = 0
    created_branch: bool = False
    interruption_requested: bool = False
    outcome: str = "completed"
    error: str | None = None


class OpenPullRequestInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str | None = Field(
        default=None,
        max_length=1000,
        description=(
            "Pull request title; defaults to the newest commit subject on "
            "the delivery branch."
        ),
    )
    body: str | None = Field(
        default=None,
        max_length=20000,
        description=(
            "Optional pull request body; defaults to the list of commit "
            "subjects since the base branch."
        ),
    )
    base_branch: str | None = Field(
        default=None,
        max_length=200,
        description=(
            "Optional base branch for the pull request; defaults to the "
            "branch the delivery line forked from."
        ),
    )


class OpenPullRequestOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    url: str = ""
    branch: str = ""
    created: bool = False
    interruption_requested: bool = False
    outcome: str = "completed"
    error: str | None = None


def create_commit_changes_tool(workspace: WorkspaceRuntime) -> Tool:
    """Commit workspace changes on a praxis/* branch after normal approvals."""
    input_schema, validate_input = pydantic_input(CommitChangesInput)
    output_schema, _validate_output = pydantic_input(CommitChangesOutput)
    root = Path(workspace.root)

    async def run(arguments: Mapping[str, JsonValue]) -> CommitChangesOutput:
        request = CommitChangesInput.model_validate(dict(arguments))
        interrupted = False
        try:
            result = await GitDelivery(root).commit(
                request.message, branch_name=request.branch_name,
            )
        except DeliveryInterruptedError as error:
            result = cast(DeliveryResult, error.result)
            interrupted = True
        except DeliveryError as error:
            return CommitChangesOutput(error=str(error), outcome=error.outcome)
        return CommitChangesOutput(
            commit_id=result.commit_id,
            branch=result.branch,
            changed_file_count=result.changed_file_count,
            created_branch=result.created_branch,
            interruption_requested=interrupted,
        )

    def normalize_output(raw: object) -> NormalizedToolOutput:
        output = CommitChangesOutput.model_validate(raw)
        return NormalizedToolOutput(
            structured_content=json_schema_output(
                output_schema, output.model_dump(mode="json"),
            ),
            is_error=output.error is not None,
            error_code="delivery_failed" if output.error is not None else None,
            error_message=output.error,
            retryable=False,
        )

    return Tool(
        definition=ToolDefinition(
            name="commit_changes",
            description=(
                "Commit every current workspace change on the repository's own "
                "praxis/* delivery branch, using the user's Git identity, ignore "
                "rules and hooks. This host-side tool is the only way you can "
                "create commits: repository metadata stays closed to you, and "
                "stacked calls reuse the same praxis branch. Inspect the pending "
                "set first with run_command 'git status --porcelain'. Refuses "
                "when nothing changed, the workspace is not a repository root, "
                "or a merge/rebase is unfinished."
            ),
            input_schema=input_schema,
        ),
        validate_input=validate_input,
        run=run,
        normalize_output=normalize_output,
        output_schema=output_schema,
        static_effects=frozenset({ToolEffect.EXECUTE_PROCESS}),
        resolve_use=lambda _arguments: ResolvedToolUse(
            effects=frozenset({ToolEffect.EXECUTE_PROCESS}),
            targets=(
                ToolTarget(kind="cwd_path", value=str(root)),
                ToolTarget(kind="execution_mode", value="host_delivery"),
            ),
        ),
        execution_revision="delivery-commit-v2",
        idempotent=False,
        concurrency_safe=False,
        cancellation_mode=CancellationMode.COOPERATIVE,
        interrupt_behavior=InterruptBehavior.FINISH_CURRENT,
        timeout_seconds=60.0,
        max_model_output_bytes=8_192,
    )


def create_open_pull_request_tool(workspace: WorkspaceRuntime) -> Tool:
    """Push the delivery branch and open its pull request after approvals."""
    input_schema, validate_input = pydantic_input(OpenPullRequestInput)
    output_schema, _validate_output = pydantic_input(OpenPullRequestOutput)
    root = Path(workspace.root)

    async def run(arguments: Mapping[str, JsonValue]) -> OpenPullRequestOutput:
        request = OpenPullRequestInput.model_validate(dict(arguments))
        interrupted = False
        try:
            result = await GitDelivery(root).open_pull_request(
                title=request.title, body=request.body, base_branch=request.base_branch,
            )
        except DeliveryInterruptedError as error:
            result = cast(PullRequestResult, error.result)
            interrupted = True
        except DeliveryError as error:
            return OpenPullRequestOutput(error=str(error), outcome=error.outcome)
        return OpenPullRequestOutput(
            url=result.url, branch=result.branch, created=result.created,
            interruption_requested=interrupted,
        )

    def normalize_output(raw: object) -> NormalizedToolOutput:
        output = OpenPullRequestOutput.model_validate(raw)
        return NormalizedToolOutput(
            structured_content=json_schema_output(
                output_schema, output.model_dump(mode="json"),
            ),
            is_error=output.error is not None,
            error_code="delivery_failed" if output.error is not None else None,
            error_message=output.error,
            retryable=False,
        )

    return Tool(
        definition=ToolDefinition(
            name="open_pull_request",
            description=(
                "Push the current praxis/* delivery branch to origin and open (or "
                "return) its GitHub pull request through the GitHub CLI. Commits "
                "beyond the base branch must already exist, so call commit_changes "
                "first. The base defaults to the branch the delivery line forked "
                "from; the title defaults to the newest commit subject. Pushing "
                "uses the user's own Git credentials: when they are missing the "
                "call fails with the gh auth setup-git hint instead of hanging. "
                "Returns the pull request URL."
            ),
            input_schema=input_schema,
        ),
        validate_input=validate_input,
        run=run,
        normalize_output=normalize_output,
        output_schema=output_schema,
        static_effects=frozenset({ToolEffect.EXECUTE_PROCESS, ToolEffect.NETWORK}),
        resolve_use=lambda _arguments: ResolvedToolUse(
            effects=frozenset({ToolEffect.EXECUTE_PROCESS, ToolEffect.NETWORK}),
            targets=(
                ToolTarget(kind="cwd_path", value=str(root)),
                ToolTarget(kind="execution_mode", value="host_delivery"),
            ),
        ),
        execution_revision="delivery-pull-request-v2",
        idempotent=False,
        concurrency_safe=False,
        cancellation_mode=CancellationMode.COOPERATIVE,
        interrupt_behavior=InterruptBehavior.FINISH_CURRENT,
        timeout_seconds=180.0,
        max_model_output_bytes=8_192,
    )


__all__ = [
    "create_commit_changes_tool",
    "create_open_pull_request_tool",
]
