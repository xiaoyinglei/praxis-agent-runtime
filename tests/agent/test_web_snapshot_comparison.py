"""Saved-text comparisons need two complete snapshots of the same source over time."""

from __future__ import annotations

import hashlib

import pytest


def snapshot(text: str, **updates):
    return {
        "url": "https://example.com/page", "fetched_at": "2026-10-08T12:00:00+00:00",
        "text": text, "content_hash": hashlib.sha256(text.encode()).hexdigest(),
        "truncated": False, "extraction_method": "html", **updates,
    }


def compare(current, baseline):
    from agent_runtime.tools.web_snapshot import compare_saved_sources

    return compare_saved_sources(current, baseline, current_source_id="current-source",
                                 baseline_source_id="baseline-source")


@pytest.mark.parametrize("before,after", [
    ("Open roles: Engineer", "Open roles: Engineer, Analyst"),
    ("API: GET /v1/items", "API: GET /v2/items"),
    ("The museum opens at 9 AM.", "The museum opens at 10 AM."),
    ("Sign in to continue", "Sign in to continue."),
])
def test_changed_saved_text_has_provenance_without_business_verdict(before, after):
    baseline = snapshot(before)
    current = snapshot(after, fetched_at="2026-10-09T12:00:00Z")
    result = compare(current, baseline)
    assert result["status"] == "changed"
    assert result["scope"] == "saved_text"
    assert "business" in result["limitation"]
    for name, source_id, source in (("baseline", "baseline-source", baseline),
                                    ("current", "current-source", current)):
        assert result[name] == {"source_id": source_id, "url": source["url"],
                                "fetched_at": source["fetched_at"], "content_hash": source["content_hash"]}
    assert set(result) == {"status", "reason", "scope", "limitation", "baseline", "current"}
    assert before not in result.values() and after not in result.values()


def test_same_text_is_unchanged_even_if_non_text_metadata_changed():
    baseline = snapshot("Open roles: Engineer", title="Old title")
    current = snapshot("Open roles: Engineer", fetched_at="2026-10-09T12:00:00Z", title="New title")
    assert compare(current, baseline)["status"] == "unchanged"


def test_missing_baseline_never_infers_new_content():
    result = compare(snapshot("New jobs and API requirements"), None)
    assert result["status"] == "baseline_missing"
    assert result["baseline"] == {"source_id": "baseline-source", "url": None,
                                   "fetched_at": None, "content_hash": None}


def test_missing_baseline_without_requested_source_has_no_invented_id():
    from agent_runtime.tools.web_snapshot import compare_saved_sources

    result = compare_saved_sources(snapshot("Current text"), None, current_source_id="current-source")
    assert result["status"] == "baseline_missing"
    assert result["baseline"]["source_id"] is None


def test_url_normalization_removes_fragment_and_default_port():
    baseline = snapshot("Old", url="https://EXAMPLE.com:443/page#first")
    current = snapshot("New", url="https://example.com/page#second", fetched_at="2026-10-09T12:00:00Z")
    assert compare(current, baseline)["status"] == "changed"


@pytest.mark.parametrize("url", [
    "https://example.com/other", "https://example.com/Page", "https://example.com/page?q=1",
    "http://example.com/page", "https://other.example/page", "invalid URL",
])
def test_different_or_invalid_source_url_is_incomparable(url):
    assert compare(snapshot("New", url=url, fetched_at="2026-10-09T12:00:00Z"),
                   snapshot("Old"))["status"] == "incomparable"


@pytest.mark.parametrize("field,value", [
    ("truncated", True), ("extraction_error_code", "web_content_empty"),
    ("extraction_error_message", "Parser failed"), ("extraction_method", "failed"),
    ("extraction_method", "not_attempted"), ("content_hash", ""),
])
@pytest.mark.parametrize("side", ["baseline", "current"])
def test_partial_or_failed_snapshot_cannot_assert_same_or_changed(field, value, side):
    baseline = snapshot("Same")
    current = snapshot("Same", fetched_at="2026-10-09T12:00:00Z")
    {"baseline": baseline, "current": current}[side][field] = value
    assert compare(current, baseline)["status"] == "incomplete"


@pytest.mark.parametrize("value", [None, "", "not-a-date", "2026-10-08", "2026-10-08T12:00:00",
                                    "2026-13-08T12:00:00Z"])
@pytest.mark.parametrize("side", ["baseline", "current"])
def test_unknown_or_naive_retrieval_time_is_incomparable(value, side):
    baseline = snapshot("Old")
    current = snapshot("New", fetched_at="2026-10-09T12:00:00Z")
    {"baseline": baseline, "current": current}[side]["fetched_at"] = value
    assert compare(current, baseline)["status"] == "incomparable"


@pytest.mark.parametrize("baseline_time", ["2026-10-09T12:00:00Z", "2026-10-10T12:00:00Z"])
def test_baseline_must_be_strictly_earlier_than_current(baseline_time):
    assert compare(snapshot("New", fetched_at="2026-10-09T12:00:00Z"),
                   snapshot("Old", fetched_at=baseline_time))["status"] == "incomparable"


def test_retrieval_times_compare_instants_across_timezones():
    baseline = snapshot("Old", fetched_at="2026-10-09T13:00:00+08:00")
    current = snapshot("New", fetched_at="2026-10-09T06:00:00Z")
    assert compare(current, baseline)["status"] == "changed"


def test_equal_instants_in_different_timezones_are_incomparable():
    baseline = snapshot("Old", fetched_at="2026-10-09T13:00:00+08:00")
    current = snapshot("New", fetched_at="2026-10-09T05:00:00Z")
    assert compare(current, baseline)["status"] == "incomparable"


@pytest.mark.parametrize("diagnostics", [
    {"blocked_actions": {"method_not_get": 1}},
    {"resource_failures": {"http_error:Fetch": 1}},
    {"load_timeouts": ["networkidle"]},
    {"snapshot_available": False},
])
@pytest.mark.parametrize("side", ["baseline", "current"])
@pytest.mark.parametrize("current_text", ["Old", "New"])
def test_partial_browser_load_cannot_establish_saved_text_equality_or_change(diagnostics, side, current_text):
    baseline = snapshot("Old")
    current = snapshot(current_text, fetched_at="2026-10-09T12:00:00Z")
    {"baseline": baseline, "current": current}[side]["render_diagnostics"] = diagnostics
    assert compare(current, baseline)["status"] == "incomplete"


@pytest.mark.parametrize("current_text,status", [("Old", "unchanged"), ("New", "changed")])
def test_skipped_visual_resources_do_not_make_text_snapshot_incomplete(current_text, status):
    diagnostics = {"snapshot_available": True, "blocked_actions": {}, "resource_failures": {},
                   "load_timeouts": [], "skipped_resources": {"Image": 8, "Font": 2}}
    baseline = snapshot("Old", render_diagnostics=diagnostics)
    current = snapshot(current_text, fetched_at="2026-10-09T12:00:00Z", render_diagnostics=diagnostics)
    assert compare(current, baseline)["status"] == status


def test_incomparable_provenance_does_not_copy_unbounded_values():
    huge = "x" * 200_000
    result = compare(snapshot(huge, url=huge, fetched_at=huge), snapshot("Old"))
    assert result["status"] == "incomparable"
    assert len(str(result)) < 6000
