import pytest
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import ValidationError

from src.config import Settings
from src.graph.llm import LLMConfigError, LLMRole, get_model, model_name_for


def cfg(**kw) -> Settings:
    """Settings that ignore the real .env file."""
    return Settings(_env_file=None, **kw)


def test_builds_gemini_model_from_settings():
    s = cfg(
        llm_api_key="fake-key",
        llm_model="some-model",
        llm_temperature=0.3,
        llm_timeout_s=12,
        llm_max_retries=1,
    )
    m = get_model(LLMRole.SYNTH, settings=s)
    assert isinstance(m, ChatGoogleGenerativeAI)
    assert m.model == "some-model"
    assert m.temperature == 0.3
    assert m.timeout == 12
    assert m.max_retries == 1


def test_defaults_are_conservative():
    s = cfg(llm_api_key="fake-key")
    assert s.llm_temperature == 0.0
    assert s.llm_max_retries == 2  # not the client's default of 6
    assert get_model(settings=s).max_retries == 2


@pytest.mark.parametrize("key", ["", "   "])
def test_missing_key_fails_loudly(key):
    with pytest.raises(LLMConfigError, match="LLM_API_KEY"):
        get_model(settings=cfg(llm_api_key=key))


def test_blank_model_name_fails():
    with pytest.raises(LLMConfigError):
        get_model(settings=cfg(llm_api_key="fake-key", llm_model=" "))


def test_role_accepts_string_and_rejects_unknown():
    s = cfg(llm_api_key="fake-key")
    assert get_model("critic", settings=s) is not None
    with pytest.raises(ValueError):
        get_model("wizard", settings=s)


def test_phase3_every_role_uses_the_single_configured_model():
    s = cfg(llm_api_key="fake-key", llm_model="m1")
    assert {model_name_for(r, s) for r in LLMRole} == {"m1"}


def test_api_key_is_not_leaked_in_repr():
    m = get_model(settings=cfg(llm_api_key="super-secret-key-123"))
    assert "super-secret-key-123" not in repr(m)
    assert "super-secret-key-123" not in str(m)


@pytest.mark.parametrize(
    "bad",
    [
        {"llm_temperature": -0.1},
        {"llm_temperature": 5},
        {"llm_timeout_s": 0},
        {"llm_max_retries": -1},
        {"llm_max_retries": 50},
    ],
)
def test_bad_llm_settings_rejected(bad):
    with pytest.raises(ValidationError):
        cfg(**bad)
