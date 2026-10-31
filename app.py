"""Streamlit chat UI.  Run:  streamlit run app.py"""

import os
from pathlib import Path

import groq
import streamlit as st

import rag
from rag import DocumentQA, read_file

SAMPLE_DOCS = Path(__file__).parent / "sample_docs"

st.set_page_config(page_title="Chat with your Docs", page_icon="📄", layout="wide")


@st.cache_resource(show_spinner="Indexing documents...")
def build_index(files: tuple[tuple[str, bytes], ...]) -> DocumentQA:
    return DocumentQA([(name, read_file(name, data)) for name, data in files])


def configured_key() -> str:
    """Key set by the host: an environment variable, or a Streamlit Cloud secret."""
    if os.environ.get("GROQ_API_KEY"):
        return os.environ["GROQ_API_KEY"]
    try:
        return st.secrets.get("GROQ_API_KEY", "")
    except Exception:  # no secrets file configured
        return ""


with st.sidebar:
    st.header("1. API key")
    host_key = configured_key()
    if host_key:
        st.success("API key is configured. Just start asking.")
        api_key = host_key
    else:
        api_key = st.text_input("Groq API key", type="password",
                                placeholder="gsk_...  (or set GROQ_API_KEY)")
        st.caption("Free key: https://console.groq.com/keys")
    if not api_key:
        st.caption("Without a key you still see the matching passages, just no AI answer.")

    st.header("2. Documents")
    uploads = st.file_uploader("Upload PDF / TXT / MD", type=["pdf", "txt", "md"],
                               accept_multiple_files=True)
    use_samples = st.checkbox("Include sample documents", value=not uploads)

    files = [(f.name, f.getvalue()) for f in uploads or []]
    if use_samples:
        files += [(p.name, p.read_bytes()) for p in sorted(SAMPLE_DOCS.iterdir())]

    st.caption(f"Model: `{rag.MODEL}` (on Groq)")
    if st.button("Clear chat"):
        st.session_state.messages = []

st.title("📄 Chat with your Docs")
st.caption("Ask questions in plain English. Answers come only from your documents, with citations.")

if not files:
    st.info("Upload a document or tick **Include sample documents** in the sidebar.")
    st.stop()

try:
    qa = build_index(tuple(files))
except ValueError as e:
    st.error(str(e))
    st.stop()

st.caption(f"Indexed **{len(qa.documents)}** file(s) → **{len(qa.retriever.chunks)}** chunks: "
           + ", ".join(name for name, _ in qa.documents))

if "messages" not in st.session_state:
    st.session_state.messages = []


def render_sources(sources):
    if not sources:
        return
    with st.expander(f"Sources ({len(sources)})"):
        for n, (src, score, text) in enumerate(sources, start=1):
            st.markdown(f"**[{n}] {src}** · relevance {score:.2f}")
            st.text(text)


for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        render_sources(msg.get("sources"))

if not st.session_state.messages:
    st.markdown("**Try:** *How many sick days do I get?* · *What's the hotel limit in Bengaluru?* · "
                "*How long are deleted files kept on the Family plan?*")

if question := st.chat_input("Ask a question about your documents"):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    results = qa.retrieve(question)
    sources = [(c.source, s, c.text) for c, s in results]
    with st.chat_message("assistant"):
        if not api_key:
            answer = ("🔑 Add a Groq API key in the sidebar to get an AI answer. "
                      "Here are the most relevant passages I found:" if results
                      else "I couldn't find anything related to that in the documents.")
        else:
            with st.spinner("Thinking..."):
                try:
                    answer = qa.ask(question, client=groq.Groq(api_key=api_key))["answer"]
                except groq.AuthenticationError:
                    answer = "❌ That API key was rejected. Please check it."
                except groq.RateLimitError:
                    answer = "⏳ Free-tier rate limit reached. Wait a minute and try again."
                except groq.APIConnectionError:
                    answer = "🌐 Couldn't reach the Groq API. Check your internet connection."
                except groq.APIStatusError as e:
                    answer = f"⚠️ Groq API error ({e.status_code}): {e.message}"
        st.markdown(answer)
        render_sources(sources)
    st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})
