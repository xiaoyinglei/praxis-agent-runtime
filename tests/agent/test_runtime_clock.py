from datetime import UTC, datetime

from agent_runtime import runtime_clock


def test_clock_uses_configured_timezone_across_date_and_year_boundary(monkeypatch):
    class FixedDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime(2031, 12, 31, 18, 0, tzinfo=UTC)

    monkeypatch.setattr(runtime_clock, "datetime", FixedDatetime)
    monkeypatch.setenv("TZ", "Asia/Tokyo")
    clock = runtime_clock.capture_runtime_clock()
    assert clock == {
        "current_time": "2032-01-01T03:00:00+09:00",
        "current_date": "2032-01-01",
        "timezone": "Asia/Tokyo",
        "utc_offset": "+09:00",
    }


def test_invalid_timezone_retains_explicit_system_local_offset(monkeypatch):
    monkeypatch.setenv("TZ", "invalid/timezone")
    clock = runtime_clock.capture_runtime_clock()
    instant = datetime.fromisoformat(clock["current_time"])
    assert instant.utcoffset() is not None
    assert clock["current_date"] == instant.date().isoformat()
    assert clock["utc_offset"] == instant.strftime("%z")[:3] + ":" + instant.strftime("%z")[3:]
    assert clock["timezone"] != "invalid/timezone"
