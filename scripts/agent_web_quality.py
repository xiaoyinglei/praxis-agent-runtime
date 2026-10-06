"""Reproducible fixture audit of the real web Tool -> Executor -> model envelope.

Run: uv run python scripts/agent_web_quality.py
Optional baseline capture (developer use): --before-dir /tmp/praxis-web-before
The bundled tokenizer measures text consistently; these are not billing tokens.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib.util
import json
import re
import sys
from pathlib import Path
from typing import Any

import httpx
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agent_runtime.core.messages import tool_result_message  # noqa: E402
from agent_runtime.modeling.tokenization import TokenAccountingService, TokenizerContract  # noqa: E402
from agent_runtime.tools.builtins import web  # noqa: E402
from agent_runtime.tools.executor import ToolExecutor  # noqa: E402
from agent_runtime.tools.permissions import ToolExecutionContext  # noqa: E402
from agent_runtime.tools.tool import ToolCall, ToolCallOrigin  # noqa: E402
from agent_runtime.tools.web_content import extract_content  # noqa: E402
from agent_runtime.tools.web_http import PublicWebClient  # noqa: E402

FIXTURES = ROOT / 'tests/agent/fixtures/web'
ACCOUNTING = TokenAccountingService(TokenizerContract(
    embedding_model_name='', tokenizer_model_name='deepseek-v4-official',
    chunking_tokenizer_model_name='deepseek-v4-official',
))


def load_before(directory: Path) -> tuple[Any, Any]:
    modules = []
    for filename in ['web_content', 'web']:
        name = 'quality_before_' + filename
        spec = importlib.util.spec_from_file_location(name, directory / (filename + '.py'))
        assert spec and spec.loader
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        modules.append(module)

    async def before_parse(body: bytes, content_type: str, url: str) -> Any:
        return modules[0].extract_content(body, content_type, url)

    modules[1].extract_content_async = before_parse
    return modules[0].extract_content, modules[1]


def normalized(text: str) -> str:
    text = re.sub(r'\[([^]]*)\]\([^)]*\)', r'\1', text)
    text = re.sub(r'\[([^]]*)\]\[L\d+\]', r'\1', text)
    return re.sub(r'\s+', '', text)


def gold(body: bytes, name: str) -> list[str]:
    if name == 'raw_source':
        return [body.decode().strip()]
    soup = BeautifulSoup(body, 'html.parser')
    selector = {
        'news_29': '#artibody p[cms-style]', 'news_30': '#artibody p[cms-style]',
        'python_docs': '[role=main] p, [role=main] pre',
        'company_index': 'a[href*="doc-initqqvi2188200"], a[href*="doc-initpatz6145090"]',
    }[name]
    return [node.get_text() for node in soup.select(selector) if node.get_text(strip=True)]


async def measure(extractor: Any, web_module: Any) -> dict[str, Any]:
    records = {}
    manifest = json.loads((FIXTURES / 'manifest.json').read_text())
    for name, meta in manifest.items():
        body = (FIXTURES / (name + ('.html' if 'html' in meta['content_type'] else '.txt'))).read_bytes()
        extracted = extractor(body, meta['content_type'], meta['url'])
        expected = gold(body, name)
        weights = [len(normalized(text)) for text in expected]
        full = normalized(extracted.text)
        full_retention = sum(w for t, w in zip(expected, weights, strict=True)
                             if normalized(t) in full) / max(1, sum(weights))
        blobs: dict[str, bytes] = {}
        hits = []

        def save(content: bytes, target: dict[str, bytes] = blobs) -> str:
            identity = 'artifact_' + 'a' * 32
            target[identity] = content
            return identity

        client = PublicWebClient(transport=httpx.MockTransport(
            lambda req, calls=hits, data=body, media=meta['content_type']: calls.append(req) or httpx.Response(
                200, content=data, headers={'content-type': media}),
        ))
        try:
            tools = {t.definition.name: t for t in web_module.create_web_tools(
                client, save_source=save, load_source=blobs.__getitem__)}
            executor = ToolExecutor(tools)
            origin = ToolCallOrigin('request', 'revision', ('web_fetch',))
            execution = await executor.execute(ToolCall('fetch', 'web_fetch', {
                'url': meta['url'], 'max_lines': 80,
            }, origin), context=ToolExecutionContext(allow_web_tools=True))
            assert not execution.result.is_error, execution.result.error_code
            envelope_data = json.loads(tool_result_message(execution.result).content)
            # Normalize only the volatile timestamp for byte/token comparisons.
            envelope_data['structured_content']['fetched_at'] = '2026-10-05T00:00:00+00:00'
            envelope = json.dumps(envelope_data, sort_keys=True, ensure_ascii=False, separators=(',', ':'))
            output = envelope_data['structured_content']
            excerpt = normalized(re.sub(r'^\d+: ', '', output['content'], flags=re.M))
            excerpt_retention = sum(w for t, w in zip(expected, weights, strict=True)
                                    if normalized(t) in excerpt) / max(1, sum(weights))
            # Annotated markers for this fixed corpus, not production extraction rules.
            markers = {
                'news_29': ['新浪首页', '登录新浪财经APP', '海量资讯、精准解读', 'callback', 'document.write'],
                'news_30': ['新浪首页', '海量资讯、精准解读', 'callback', 'document.write'],
                'python_docs': ['Table of Contents', 'Quick search', 'Navigation'],
                'company_index': ['新浪首页', 'document.write', 'callback'],
                'raw_source': [],
            }[name]
            noise_hits = sum(marker in extracted.text for marker in markers)
            repeat = await executor.execute(ToolCall('repeat', 'web_fetch', {'url': meta['url']}, origin),
                                            context=ToolExecutionContext(allow_web_tools=True))
            assert not repeat.result.is_error
            continuation_ok = True
            if output['next_line'] is not None:
                continuation = await executor.execute(ToolCall('continue', 'web_fetch', {
                    'source_id': output['source_id'], 'start_line': output['next_line'], 'max_lines': 80,
                }, origin), context=ToolExecutionContext())
                continuation_ok = (not continuation.result.is_error
                                   and continuation.result.structured_content['content_hash'] == output['content_hash'])
            records[name] = {
                'fixture_sha256': hashlib.sha256(body).hexdigest(),
                'gold_blocks': len(expected), 'full_retention': round(full_retention, 4),
                'first_80_lines_retention': round(excerpt_retention, 4),
                'non_gold_text_share': round(max(0, 1 - sum(weights) * full_retention / max(1, len(full))), 4),
                'noise_marker_hits': noise_hits, 'annotated_noise_markers': len(markers),
                'model_bytes': len(envelope.encode()), 'model_text_tokens': ACCOUNTING.count(envelope),
                'model_links': len(output['links']), 'extracted_bytes': len(extracted.text.encode()),
                'fetch_repeat_network_calls': len(hits), 'continuation_ok': continuation_ok,
                'source_id_matches_artifact': output['source_id'] in blobs,
            }
        finally:
            await client.aclose()
    return records


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--before-dir', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    extractor, module = load_before(args.before_dir) if args.before_dir else (extract_content, web)
    records = await measure(extractor, module)
    result = {'tokenizer': ACCOUNTING.budget_count_source(), 'pages': records}
    if args.before_dir:
        result['baseline_source_sha256'] = {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in args.before_dir.glob('*.py')
        }
    else:
        baseline = json.loads((FIXTURES / 'baseline.json').read_text())['pages']
        for name, record in records.items():
            assert record['fixture_sha256'] == baseline[name]['fixture_sha256']
            assert record['full_retention'] >= 0.98, (name, 'lost body evidence', record)
            assert record['noise_marker_hits'] == 0, (name, 'known noise survived', record)
            assert record['model_links'] <= 8
            assert record['fetch_repeat_network_calls'] == 1
            assert record['continuation_ok'] and record['source_id_matches_artifact']
            if name.startswith('news_') or name == 'company_index':
                assert record['first_80_lines_retention'] == 1.0
                if name.startswith('news_'):
                    assert record['non_gold_text_share'] <= 0.1
                assert record['model_text_tokens'] <= baseline[name]['model_text_tokens'] * 0.4
            print(name, json.dumps(record, ensure_ascii=False))
    if args.output:
        args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n')
    elif args.before_dir:
        print(json.dumps(result, indent=2))


if __name__ == '__main__':
    asyncio.run(main())
