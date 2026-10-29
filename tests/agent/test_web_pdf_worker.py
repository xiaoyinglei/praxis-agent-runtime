from __future__ import annotations

import asyncio
import sys

import pymupdf
import pytest

from agent_runtime.tools import web_content
from agent_runtime.tools.web_http import PublicWebError


@pytest.mark.anyio
async def test_native_pdf_text_is_extracted_in_a_separate_worker_on_this_platform(monkeypatch):
    with pymupdf.open() as document:
        document.new_page().insert_text((72, 72), "First page financial evidence")
        document.new_page().insert_text((72, 72), "Second page original figures")
        body = document.tobytes()
    monkeypatch.setenv("PRAXIS_PDF_TEST_SECRET", "not-inherited-by-parser")
    spawn = asyncio.create_subprocess_exec
    launches = []

    async def record_spawn(*args, **kwargs):
        launches.append((args, kwargs))
        return await spawn(*args, **kwargs)

    monkeypatch.setattr(web_content.asyncio, "create_subprocess_exec", record_spawn)
    result = await web_content.extract_content_async(body, "application/pdf", "https://example.com/report.pdf")
    assert "Page 1" in result.text and "First page financial evidence" in result.text
    assert "Page 2" in result.text and "Second page original figures" in result.text
    assert len(launches) == 1 and launches[0][1]["start_new_session"]
    assert "-I" in launches[0][0]
    assert "PRAXIS_PDF_TEST_SECRET" not in launches[0][1]["env"]


@pytest.mark.anyio
async def test_parser_stops_oversized_stdout_before_worker_exit(monkeypatch):
    spawn = asyncio.create_subprocess_exec
    workers = []

    async def oversized_worker(*_args, **kwargs):
        process = await spawn(sys.executable, "-I", "-c", (
            "import sys,time; sys.stdin.buffer.read(); "
            "sys.stdout.buffer.write(b'x'*2000001); sys.stdout.flush(); time.sleep(30)"
        ), **kwargs)
        workers.append(process)
        return process

    monkeypatch.setattr(web_content.asyncio, "create_subprocess_exec", oversized_worker)
    monkeypatch.setattr(web_content, "_PARSER_TIMEOUT_SECONDS", 2)
    with pytest.raises(PublicWebError, match="resource limit"):
        await web_content.extract_content_async(b"text", "text/plain", "https://example.com")
    assert workers[0].returncode is not None


@pytest.mark.anyio
async def test_invalid_pdf_is_a_parser_failure_instead_of_a_platform_rejection():
    with pytest.raises(PublicWebError) as error:
        await web_content.extract_content_async(b"not a PDF", "application/pdf", "https://example.com/bad.pdf")
    assert error.value.code == "web_content_invalid"
