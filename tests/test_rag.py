"""Tests run fully offline: the Groq LLM is replaced by a fake client, so no API key is needed."""

import io
import sys
from pathlib import Path
from types import SimpleNamespace

import groq
import httpx
import pytest
from pypdf import PdfWriter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import rag  # noqa: E402

SAMPLE_DOCS = Path(__file__).resolve().parents[1] / "sample_docs"


class FakeClient:
    """Mimics groq.Groq().chat.completions.create and records what was sent."""

    def __init__(self, text="Answer [1].", retired=(), available=()):
        self.calls = []
        self.retired = set(retired)
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))
        self.models = SimpleNamespace(
            list=lambda: SimpleNamespace(data=[SimpleNamespace(id=m) for m in available]))
        self._response = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=text), finish_reason="stop")]
        )

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        if kwargs["model"] in self.retired:
            response = httpx.Response(404, request=httpx.Request("POST", "https://api.groq.com"))
            raise groq.NotFoundError(f"The model `{kwargs['model']}` does not exist", response=response, body=None)
        return self._response


@pytest.fixture(scope="module")
def qa():
    return rag.DocumentQA.from_folder(SAMPLE_DOCS)


# ---------- reading & chunking ----------

def test_reads_text_and_markdown():
    assert rag.read_file("a.txt", b"hello") == "hello"
    assert rag.read_file("a.md", "₹500".encode()) == "₹500"


def test_reads_pdf():
    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)
    buf = io.BytesIO()
    writer.write(buf)
    assert rag.read_file("blank.pdf", buf.getvalue()) == ""  # valid PDF, no text


def test_rejects_unsupported_type():
    with pytest.raises(ValueError, match="Unsupported"):
        rag.read_file("photo.jpg", b"...")


def test_chunks_respect_size_and_keep_all_text():
    text = "\n\n".join(f"Paragraph {i} " + "word " * 30 for i in range(20))
    chunks = rag.chunk_text(text, chunk_size=400, overlap=50)
    assert len(chunks) > 1
    assert all(len(c) <= 400 for c in chunks)
    for i in range(20):
        assert any(f"Paragraph {i} " in c for c in chunks)


def test_long_paragraph_is_split_with_overlap():
    text = "x" * 1000
    chunks = rag.chunk_text(text, chunk_size=400, overlap=100)
    assert [len(c) for c in chunks] == [400, 400, 400]
    assert sum(len(c) for c in chunks) > len(text)  # overlap duplicates some characters


def test_empty_documents_raise():
    with pytest.raises(ValueError, match="No text"):
        rag.Retriever([])


# ---------- retrieval ----------

def test_loads_all_sample_docs(qa):
    assert [name for name, _ in qa.documents] == ["expense_policy.md", "leave_policy.md", "product_faq.md"]


@pytest.mark.parametrize("question, expected_file, expected_text", [
    ("How many sick leave days do I get?", "leave_policy.md", "12 days of paid sick leave"),
    ("What is the hotel limit per night in metro cities?", "expense_policy.md", "6,000 per night"),
    ("How long are deleted files kept?", "product_faq.md", "Deleted files"),
    ("What is the price of the Family plan?", "product_faq.md", "Family"),
])
def test_retrieves_the_right_passage(qa, question, expected_file, expected_text):
    top_chunk, score = qa.retrieve(question)[0]
    assert top_chunk.source == expected_file
    assert expected_text in top_chunk.text
    assert score > 0


def test_unrelated_question_finds_nothing(qa):
    assert qa.retrieve("zebra quantum volcano") == []


# ---------- generation (fake LLM) ----------

def test_prompt_contains_numbered_sources_and_question(qa):
    results = qa.retrieve("parental leave weeks")
    msg = rag.build_user_message("parental leave weeks", results)
    assert '<source id="1" file="leave_policy.md">' in msg
    assert msg.rstrip().endswith("Question: parental leave weeks")


def test_ask_sends_sources_to_llm_and_returns_text(qa):
    client = FakeClient("  You get 12 days [1].  ")
    result = qa.ask("How many sick days?", client=client)
    assert result["answer"] == "You get 12 days [1]."
    call = client.calls[0]
    assert call["model"] == rag.MODEL
    system, user = call["messages"]
    assert system == {"role": "system", "content": rag.SYSTEM_PROMPT}
    assert user["role"] == "user"
    assert "12 days of paid sick leave" in user["content"]
    assert [c.source for c, _ in result["sources"]][0] == "leave_policy.md"


def test_no_api_call_when_nothing_relevant(qa):
    client = FakeClient()
    result = qa.ask("zebra quantum volcano", client=client)
    assert "couldn't find" in result["answer"]
    assert client.calls == []


def test_empty_model_reply_is_handled(qa):
    assert qa.ask("How many sick days?", client=FakeClient(text=None))["answer"] == ""


def test_reasoning_models_get_low_effort(qa, monkeypatch):
    monkeypatch.setattr(rag, "MODEL", "openai/gpt-oss-120b")
    client = FakeClient()
    qa.ask("How many sick days?", client=client)
    assert client.calls[0]["reasoning_effort"] == "low"


def test_other_models_get_no_reasoning_option(qa, monkeypatch):
    monkeypatch.setattr(rag, "MODEL", "some/other-model")
    client = FakeClient()
    qa.ask("How many sick days?", client=client)
    assert "reasoning_effort" not in client.calls[0]


def test_retired_model_falls_back_automatically(qa, monkeypatch):
    monkeypatch.setattr(rag, "MODEL", "llama-3.3-70b-versatile")
    client = FakeClient("12 days [1].", retired={"llama-3.3-70b-versatile"},
                        available=["openai/gpt-oss-20b", "whisper-large-v3"])
    result = qa.ask("How many sick days?", client=client)
    assert result["answer"] == "12 days [1]."
    assert [c["model"] for c in client.calls] == ["llama-3.3-70b-versatile", "openai/gpt-oss-20b"]
    assert rag.MODEL == "openai/gpt-oss-20b"  # remembered for the next question


def test_error_when_no_fallback_is_available(qa, monkeypatch):
    monkeypatch.setattr(rag, "MODEL", "llama-3.3-70b-versatile")
    client = FakeClient(retired={"llama-3.3-70b-versatile"}, available=["whisper-large-v3"])
    with pytest.raises(groq.NotFoundError):
        qa.ask("How many sick days?", client=client)


# ---------- bundled test files (real PDFs) ----------

TEST_FILES = Path(__file__).resolve().parents[1] / "test_files"


def test_test_file_pdfs_have_text():
    docs = dict(rag.load_folder(TEST_FILES))
    assert set(docs) == {"it_security_policy.pdf", "health_insurance_guide.pdf", "onboarding_guide.txt"}
    assert "14 characters" in docs["it_security_policy.pdf"]
    assert "75,000" in docs["health_insurance_guide.pdf"]


@pytest.mark.parametrize("question, expected_file, expected_text", [
    ("How long must my password be?", "it_security_policy.pdf", "14 characters"),
    ("Is dental treatment covered?", "health_insurance_guide.pdf", "dental"),
    ("How long is the probation period?", "onboarding_guide.txt", "6 months"),
])
def test_retrieval_on_test_files(question, expected_file, expected_text):
    top_chunk, _ = rag.DocumentQA.from_folder(TEST_FILES).retrieve(question)[0]
    assert top_chunk.source == expected_file
    assert expected_text in top_chunk.text
