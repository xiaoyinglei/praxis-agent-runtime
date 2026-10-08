"""Manual real-model paired trials; never part of the frozen code benchmark or fast CI.

Uses the public SDK with a real configured model and controlled public-web fixtures.
Task output/files are checked independently; tool order and completion claims are not scores.
Reuses the code benchmark's bounded subprocess execution, cleanup, diffs and redacted logs.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import shutil
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.agent_code_benchmark import (  # noqa: E402
    _changed_paths,
    _initialize_snapshot_git,
    _provider_error_code,
    _run_command,
    _workspace_diff,
    _write_log,
)


def worker(runtime: Path, case: dict[str, Any], directory: Path) -> None:
    sys.path.insert(0, str(runtime))
    import httpx

    from agent_runtime import Agent
    from agent_runtime.tools import web_http

    calls = []
    client_type = web_http.PublicWebClient

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append({'method': request.method, 'url': str(request.url)})
        body = case.get('pages', {}).get(request.url.path)
        if request.url.host != 'example.com' or body is None:
            return httpx.Response(404)
        media = 'application/json' if request.url.path.endswith('.json') else 'text/html'
        return httpx.Response(200, text=body, headers={'content-type': media})

    # Replace only the HTTP transport. Model resolution, provider requests,
    # tool execution, permissions, context and checkpointing remain public SDK paths.
    web_http.PublicWebClient = lambda **options: client_type(  # type: ignore[misc]
        **options, transport=httpx.MockTransport(handler))
    agent = Agent(model='deepseek-flash', workspace_path=directory / 'workspace',
                  checkpoint_db=directory / 'rollout.sqlite', model_session_path=None, enable_workspace_mcp=False)
    result = asyncio.run(agent.run(case['instruction'], max_turns=8, max_tokens_total=80_000,
                                  allow_write_tools=bool(case.get('write')),
                                  allow_execute_tools=bool(case.get('write')),
                                  allow_web_tools=True, require_workspace_change=bool(case.get('write'))))
    payload = {'status': result.status, 'answer': result.answer, 'turn_id': result.turn_id,
               'stop_reason': result.stop_reason, 'usage': asdict(result.usage),
               'diagnostics': [asdict(d) for d in result.diagnostics], 'network_requests': calls,
               'tool_errors': [{'tool': t.tool_name, 'code': t.error_code} for t in result.tool_calls if t.is_error]}
    (directory / 'sdk_result.json').write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    print(json.dumps({'status': result.status, 'turn_id': result.turn_id, 'model_calls': result.usage.model_calls}))


def check(case: dict[str, Any], directory: Path, sdk: dict[str, Any]) -> dict[str, Any]:
    if case.get('write'):
        command = [sys.executable, '-c', '''from pairs import parse_pair
assert parse_pair('  a : b:c  ') == ('a', 'b:c')
assert parse_pair('x:y') == ('x', 'y')
for text in ['abc', ':x', 'x:', ' : ']:
    try: parse_pair(text)
    except ValueError: pass
    else: raise AssertionError(text)
''']
        result = _run_command(command, cwd=directory / 'workspace', timeout_seconds=10)
        _write_log(directory / 'acceptance.stdout', result.stdout)
        _write_log(directory / 'acceptance.stderr', result.stderr)
        return {'passed': result.returncode == 0, 'kind': 'independent_code_behavior'}
    answer = (sdk.get('answer') or '').strip()
    if answer.startswith('```') and answer.endswith('```'):
        answer = '\n'.join(answer.splitlines()[1:-1])
    try:
        actual = json.loads(answer)
    except ValueError:
        actual = None
    return {'passed': actual == case['expected'], 'kind': 'independent_exact_result', 'actual': actual}


def classify_outcome(
    sdk: dict[str, Any], *, provider: str | None, timed_out: bool, browser_unavailable: bool, passed: bool,
) -> str:
    reason = sdk.get('stop_reason')
    if provider or reason == 'model_provider_failed':
        return 'provider_or_network_failure'
    if timed_out:
        return 'timeout'
    if reason == 'model_budget_limit_exceeded':
        return 'task_budget_exhausted'
    if any(d.get('component') == 'model' for d in sdk.get('diagnostics', [])):
        return 'model_response_failure'
    if browser_unavailable:
        return 'capability_unavailable'
    return 'passed' if passed else 'task_failed'


def run_trials(args: argparse.Namespace) -> None:
    from agent_runtime.text import load_env_file

    load_env_file(ROOT / '.env')
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    cases_path = ROOT / 'tests/agent/fixtures/web/task_cases.json'
    cases = json.loads(cases_path.read_text())
    snapshots = {'baseline': args.baseline.resolve(), 'candidate': output / 'candidate-runtime'}
    previous = json.loads((output / 'summary.json').read_text()) if (output / 'summary.json').exists() else None
    fingerprints = {}
    source_hashes = {}
    for variant, runtime in {'baseline': snapshots['baseline'], 'candidate': ROOT}.items():
        hashes = {str(p.relative_to(runtime)): hashlib.sha256(p.read_bytes()).hexdigest()
                  for name in ['agent_runtime', 'rag', 'configs'] for p in (runtime / name).rglob('*')
                  if p.is_file() and '__pycache__' not in p.parts}
        fingerprints[variant] = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
        source_hashes[variant] = hashes
    fixtures_hash = hashlib.sha256(cases_path.read_bytes()).hexdigest()
    if previous and (previous['source_fingerprints'] != fingerprints
                     or previous.get('fixtures_sha256') != fixtures_hash or previous['trials'] != args.trials):
        raise ValueError('Runtime, fixtures or trial count changed; use a fresh output directory.')
    if not snapshots['candidate'].exists():
        for name in ['agent_runtime', 'rag', 'configs']:
            shutil.copytree(ROOT / name, snapshots['candidate'] / name,
                            ignore=shutil.ignore_patterns('__pycache__'))
    for variant, hashes in source_hashes.items():
        (output / f'{variant}-source-hashes.json').write_text(json.dumps(hashes, indent=2))
    (output / 'fixtures.json').write_bytes(cases_path.read_bytes())
    records = previous['records'] if previous else []
    for trial in range(1, args.trials + 1):
        for case in cases:
            for variant in (['baseline', 'candidate'] if trial % 2 else ['candidate', 'baseline']):
                directory = output / f'{case["id"]}-{variant}-{trial}'
                if (directory / 'result.json').exists():
                    continue
                workspace = directory / 'workspace'
                workspace.mkdir(parents=True, exist_ok=True)
                (workspace / 'task_input.txt').write_text(case['instruction'])
                for name, body in case.get('files', {}).items():
                    (workspace / name).write_text(body)
                _initialize_snapshot_git(workspace)
                case_path = directory / 'case.json'
                case_path.write_text(json.dumps(case, ensure_ascii=False))
                result = _run_command([sys.executable, str(Path(__file__).resolve()), '--worker',
                                       str(snapshots[variant]), str(case_path), str(directory)],
                                      cwd=workspace, timeout_seconds=150,
                                      live_log_paths=(directory / 'agent.stdout', directory / 'agent.stderr'),
                                      progress_label=f'{case["id"]}/{variant}/{trial}')
                _write_log(directory / 'agent.diff', _workspace_diff(workspace, _changed_paths(workspace)))
                sdk_path = directory / 'sdk_result.json'
                sdk = json.loads(sdk_path.read_text()) if sdk_path.exists() else {}
                acceptance = check(case, directory, sdk)
                model_diagnostics = [d for d in sdk.get('diagnostics', []) if d.get('component') == 'model']
                provider = _provider_error_code('', result.stderr + json.dumps(model_diagnostics))
                outcome = classify_outcome(sdk, provider=provider, timed_out=result.timed_out,
                                           browser_unavailable=bool(case.get('requires_browser'))
                                           and sys.platform == 'darwin', passed=acceptance['passed'])
                record = {'task': case['id'], 'variant': variant, 'trial': trial, 'outcome': outcome,
                          'provider_error': provider, 'returncode': result.returncode,
                          'acceptance': acceptance, 'turn_id': sdk.get('turn_id'),
                          'sdk_status': sdk.get('status'), 'runtime_fingerprint': fingerprints[variant],
                          'permissions': {'write': bool(case.get('write')), 'execute': bool(case.get('write')),
                                          'web': True}}
                records.append(record)
                (directory / 'result.json').write_text(json.dumps(record, ensure_ascii=False, indent=2))
                (output / 'summary.json').write_text(json.dumps({
                    'model': 'deepseek-flash', 'fixture_network': True, 'trials': args.trials,
                    'budget': {'max_turns': 8, 'max_tokens_total': 80_000, 'timeout_seconds': 150},
                    'source_fingerprints': fingerprints, 'fixtures_sha256': fixtures_hash, 'records': records,
                }, ensure_ascii=False, indent=2))
                print(json.dumps(record, ensure_ascii=False), flush=True)
                # An infrastructure failure cannot produce a paired quality result.
                if outcome == 'provider_or_network_failure':
                    return


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == '--worker':
        worker(Path(sys.argv[2]), json.loads(Path(sys.argv[3]).read_text()), Path(sys.argv[4]))
        return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, required=True, help='Local runtime snapshot; no worktree mutation.')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--trials', type=int, choices=range(1, 4), default=2)
    args = parser.parse_args()
    run_trials(args)


if __name__ == '__main__':
    main()
