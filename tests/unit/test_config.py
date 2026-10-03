from config import Settings


def test_defaults_are_safe():
    s = Settings(_env_file=None)
    assert s.paper_mode is True
    assert s.max_trade_inr <= s.daily_cap_inr


def test_env_override(monkeypatch):
    monkeypatch.setenv("MAX_TRADE_INR", "1234")
    assert Settings(_env_file=None).max_trade_inr == 1234
