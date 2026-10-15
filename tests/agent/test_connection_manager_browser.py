from __future__ import annotations

import asyncio
import os
import threading

import pytest

from agent_runtime.connection_manager import ConnectionServer

pytestmark = pytest.mark.skipif(os.environ.get("PRAXIS_TEST_BROWSER") != "1", reason="Requires real Chromium")


def test_browser_connect_authorize_check_and_remove(tmp_path, monkeypatch):
    from playwright.sync_api import expect, sync_playwright

    entries = []
    observed = []
    consent = threading.Event()

    def register(name, **arguments):
        observed.append((name, arguments))
        entries.append({"name": name, "url": arguments["url"], "oauth": arguments["oauth"],
                        "read_only": arguments["read_only"]})

    async def login(name, *, workspace, redirect_handler):
        await redirect_handler("https://auth.example.com/authorize?state=browser-state")
        for _ in range(200):
            if consent.is_set():
                return
            await asyncio.sleep(0.02)
        raise AssertionError("test consent did not arrive")

    async def check(name, workspace):
        return {"connected": True, "tools": [f"mcp__{name}__search"], "diagnostics": []}

    monkeypatch.setattr("agent_runtime.mcp_admin.mcp_connections", lambda: list(entries))
    monkeypatch.setattr("agent_runtime.mcp_admin.register_remote_mcp", register)
    monkeypatch.setattr("agent_runtime.mcp_admin.remove_mcp_connection", lambda name: entries.clear())
    monkeypatch.setattr("agent_runtime.mcp_oauth.login_mcp_oauth", login)
    monkeypatch.setattr("agent_runtime.mcp_cli.check_mcp_connection", check)
    server = ConnectionServer(tmp_path)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        with sync_playwright() as playwright:
            executable = os.environ.get("PRAXIS_BROWSER_EXECUTABLE")
            browser = playwright.chromium.launch(
                headless=True, **({"executable_path": executable} if executable else {})
            )
            page = browser.new_page(viewport={"width": 1200, "height": 1000})
            errors = []
            page.on("pageerror", lambda error: errors.append(str(error)))
            page.goto(server.browser_url)
            expect(page.locator("#catalog .card")).to_have_count(3)
            assert "#" not in page.url  # The local capability is removed from browser history.
            page.locator(".card[data-id=notion] button").click()
            authorize = page.get_by_role("link", name="前往服务商授权")
            expect(authorize).to_be_visible()
            assert authorize.get_attribute("href").startswith("https://auth.example.com/")
            consent.set()
            expect(page.locator("#notice")).to_contain_text("notion 已连接", timeout=6000)
            assert observed[0][1]["oauth"] is True
            assert observed[0][1]["read_only"] is True
            row = page.locator(".connection").filter(has_text="notion")
            row.get_by_role("button", name="检查连接").click()
            expect(page.locator("#notice")).to_contain_text("1 个可用工具", timeout=6000)
            page.on("dialog", lambda dialog: dialog.accept())
            row.get_by_role("button", name="移除").click()
            expect(page.locator("#connections")).to_contain_text("还没有连接", timeout=6000)
            # Custom connections share the same registration path and preserve explicit action choice.
            page.get_by_label("连接名称").fill("docs")
            page.get_by_label("服务地址").fill("https://docs.example.com/mcp")
            page.get_by_label("认证方式").select_option("none")
            page.get_by_label("允许服务动作", exact=False).check()
            page.get_by_role("button", name="添加并连接").click()
            expect(page.locator("#notice")).to_contain_text("docs 已连接", timeout=6000)
            assert observed[-1][1]["read_only"] is False
            assert observed[-1][1]["oauth"] is False
            page.set_viewport_size({"width": 390, "height": 844})
            assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
            assert not errors
            browser.close()
    finally:
        consent.set()
        server.shutdown()
        server.server_close()
        worker.join(3)
