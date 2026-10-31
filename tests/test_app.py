"""Smoke tests: the Streamlit chat app runs end-to-end without a browser or API key."""
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parents[1] / "app.py")


@pytest.fixture(autouse=True)
def no_api_key(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)


def test_app_indexes_sample_docs():
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception
    assert any("Indexed **3** file(s)" in c.value for c in at.caption)


def test_question_without_key_shows_passages():
    at = AppTest.from_file(APP, default_timeout=60).run()
    at.chat_input[0].set_value("How many sick days do I get?").run()
    assert not at.exception
    assert len(at.chat_message) == 2  # user + assistant
    assert "Add a Groq API key" in at.chat_message[1].markdown[0].value
    assert "leave_policy.md" in at.chat_message[1].expander[0].markdown[0].value


def test_key_from_streamlit_secrets_is_used():
    at = AppTest.from_file(APP, default_timeout=60)
    at.secrets["GROQ_API_KEY"] = "gsk_test"
    at.run()
    assert not at.exception
    assert any("API key is configured" in s.value for s in at.success)
    assert len(at.text_input) == 0  # no key box needed
