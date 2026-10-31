"""Retrieval-Augmented Generation (RAG) over your own documents.

Pipeline:  read files -> split into chunks -> TF-IDF index -> find the chunks most similar
to the question -> send only those chunks to an LLM (Groq) -> answer with [1], [2] citations.
"""

from __future__ import annotations

import io
import os
import re
from dataclasses import dataclass
from pathlib import Path

import groq
from groq import Groq
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

SUPPORTED_EXTENSIONS = (".pdf", ".txt", ".md")
MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")
# Used in this order if MODEL is retired by Groq (they deprecate models every few months).
FALLBACK_MODELS = ["openai/gpt-oss-120b", "openai/gpt-oss-20b", "qwen/qwen3.8-27b"]

SYSTEM_PROMPT = """You answer questions using only the numbered source passages provided by the user.

- Base every statement on the sources and cite them inline like [1] or [2][3].
- If the sources don't contain the answer, say you couldn't find it in the documents. Don't guess \
or use outside knowledge.
- Keep answers short and direct. Use a bullet list when listing several items."""


@dataclass
class Chunk:
    source: str  # file name
    index: int   # position of the chunk inside its file
    text: str


# ---------- 1. Read files ----------

def read_file(name: str, data: bytes) -> str:
    """Extract plain text from a PDF / TXT / MD file given its bytes."""
    ext = Path(name).suffix.lower()
    if ext == ".pdf":
        reader = PdfReader(io.BytesIO(data))
        return "\n\n".join(page.extract_text() or "" for page in reader.pages)
    if ext in (".txt", ".md"):
        return data.decode("utf-8", errors="replace")
    raise ValueError(f"Unsupported file type: {name} (supported: {', '.join(SUPPORTED_EXTENSIONS)})")


def load_folder(folder: str | Path) -> list[tuple[str, str]]:
    """Return [(file name, text)] for every supported file in a folder."""
    files = sorted(p for p in Path(folder).iterdir() if p.suffix.lower() in SUPPORTED_EXTENSIONS)
    return [(p.name, read_file(p.name, p.read_bytes())) for p in files]


# ---------- 2. Chunk ----------

def chunk_text(text: str, chunk_size: int = 800, overlap: int = 150) -> list[str]:
    """Split text into ~chunk_size-character pieces, keeping paragraphs together where possible.

    Paragraphs are packed into a chunk until it would get too long. A paragraph that is
    itself too long is cut into windows that overlap by `overlap` characters, so a sentence
    split at a boundary still appears whole in one of them.
    """
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks, current = [], ""
    for para in paragraphs:
        if len(para) > chunk_size:
            if current:
                chunks.append(current)
                current = ""
            step = chunk_size - overlap
            chunks.extend(para[i:i + chunk_size] for i in range(0, len(para) - overlap, step))
        elif len(current) + len(para) + 2 <= chunk_size:
            current = f"{current}\n\n{para}" if current else para
        else:
            chunks.append(current)
            current = para
    if current:
        chunks.append(current)
    return chunks


def build_chunks(documents: list[tuple[str, str]], **kwargs) -> list[Chunk]:
    return [Chunk(name, i, piece)
            for name, text in documents
            for i, piece in enumerate(chunk_text(text, **kwargs))]


# ---------- 3. Retrieve ----------

class Retriever:
    """TF-IDF keyword search. Runs fully offline and needs no embedding model download."""

    def __init__(self, chunks: list[Chunk]):
        if not chunks:
            raise ValueError("No text found in the documents.")
        self.chunks = chunks
        self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True)
        self.matrix = self.vectorizer.fit_transform(c.text for c in chunks)

    def search(self, query: str, k: int = 4, min_score: float = 0.02) -> list[tuple[Chunk, float]]:
        scores = cosine_similarity(self.vectorizer.transform([query]), self.matrix)[0]
        best = scores.argsort()[::-1][:k]
        return [(self.chunks[i], float(scores[i])) for i in best if scores[i] >= min_score]


# ---------- 4. Generate ----------

def build_user_message(question: str, results: list[tuple[Chunk, float]]) -> str:
    sources = "\n\n".join(
        f'<source id="{n}" file="{chunk.source}">\n{chunk.text}\n</source>'
        for n, (chunk, _) in enumerate(results, start=1)
    )
    return f"<sources>\n{sources}\n</sources>\n\nQuestion: {question}"


def _complete(client: Groq, model: str, messages: list[dict]):
    extra = {"reasoning_effort": "low"} if model.startswith("openai/gpt-oss") else {}
    return client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.1,             # low = stick closely to the sources
        max_completion_tokens=2048,  # room for the model's reasoning + the answer
        **extra,
    )


def _replacement_model(client: Groq, failed: str) -> str | None:
    """First fallback model that this Groq account can currently use."""
    available = {m.id for m in client.models.list().data}
    return next((m for m in FALLBACK_MODELS if m != failed and m in available), None)


def ask_llm(question: str, results: list[tuple[Chunk, float]], client: Groq | None = None) -> str:
    """Ask the Groq-hosted LLM to answer from the retrieved chunks. Returns the answer text."""
    global MODEL
    if not results:
        return "I couldn't find anything related to that in the documents."
    client = client or Groq()  # reads GROQ_API_KEY from the environment
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_message(question, results)},
    ]
    try:
        response = _complete(client, MODEL, messages)
    except groq.NotFoundError:  # model retired or not available on this account
        replacement = _replacement_model(client, MODEL)
        if replacement is None:
            raise
        MODEL = replacement
        response = _complete(client, MODEL, messages)
    return (response.choices[0].message.content or "").strip()


class DocumentQA:
    """Convenience wrapper: index documents once, then ask many questions."""

    def __init__(self, documents: list[tuple[str, str]], top_k: int = 4):
        self.documents = documents
        self.retriever = Retriever(build_chunks(documents))
        self.top_k = top_k

    @classmethod
    def from_folder(cls, folder: str | Path, **kwargs) -> "DocumentQA":
        return cls(load_folder(folder), **kwargs)

    def retrieve(self, question: str) -> list[tuple[Chunk, float]]:
        return self.retriever.search(question, k=self.top_k)

    def ask(self, question: str, client: Groq | None = None) -> dict:
        results = self.retrieve(question)
        return {"answer": ask_llm(question, results, client), "sources": results}
