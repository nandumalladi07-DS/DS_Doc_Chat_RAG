---
title: Chat with your Docs
emoji: 📄
colorFrom: indigo
colorTo: purple
sdk: docker
app_port: 8501
pinned: false
---

# 📄 Chat with your Docs (RAG with Groq + Llama 3.3)

Upload PDFs, text or Markdown files and ask questions in plain English. The app finds the most relevant passages and has an LLM (**Llama 3.3 70B on Groq**, with a free API key) answer **only from those passages**, with `[1]` `[2]` citations you can expand to check.

This is **Retrieval-Augmented Generation (RAG)**, the standard pattern behind company knowledge-base chatbots.

## How it works

```
your files ──► extract text ──► split into ~800-char chunks ──► TF-IDF index
                                                                   │
question ─────────────────────────────► find top-4 similar chunks ◄┘
                                                │
                  LLM on Groq (answers only from these chunks, cites [n]) ──► answer + sources
```

| Step | Code | Detail |
|---|---|---|
| Read | `read_file()` | PDF via `pypdf`, TXT/MD decoded as UTF-8 |
| Chunk | `chunk_text()` | keeps paragraphs together and splits very long ones with overlap |
| Retrieve | `Retriever` | TF-IDF (1–2 word n-grams) + cosine similarity. Runs offline with no model download |
| Generate | `ask_llm()` | sends numbered `<source>` passages to Groq at low temperature; the system prompt forbids guessing beyond them |

Design choices worth mentioning in an interview:
- **No API call when nothing relevant is found.** This saves cost and stops made-up answers.
- **Works without an API key.** You still see the retrieved passages, which makes it easy to debug retrieval on its own.
- **Friendly errors.** A bad key, the free-tier rate limit and network problems each get a clear message instead of a crash.
- **Fully offline tests.** The LLM is replaced by a fake client, so the 18 tests need no key and cost nothing.

## Project structure

```
DS_Doc_Chat_RAG/
├── rag.py            # read → chunk → retrieve → ask the LLM (Groq)
├── app.py            # Streamlit chat UI (upload files, chat, see sources)
├── cli.py            # terminal version (+ --retrieve-only mode, no key needed)
├── sample_docs/      # 3 demo docs: leave policy, expense policy, product FAQ
├── tests/
│   ├── test_rag.py   # chunking, retrieval accuracy, prompt building, fake-LLM tests
│   └── test_app.py   # chat UI smoke tests (no browser, no key)
├── requirements.txt
└── Dockerfile        # for Hugging Face Spaces / Docker
```

## How to run (step by step)

You need **Python 3.10 or newer**. Check with `python --version` (on Windows you may need `py --version`).

### 1. Get the code

```bash
git clone https://github.com/<your-username>/DS_Doc_Chat_RAG.git
cd DS_Doc_Chat_RAG
```

### 2. Create a virtual environment and install packages

**Windows (PowerShell):**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```
If PowerShell blocks the activate script, run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once, then try again.

**macOS / Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Run the tests (no API key needed)

```bash
python -m pytest -v
```
Expected: `18 passed`.

### 4. Try retrieval without a key

```bash
python cli.py "How many sick days do I get?" --retrieve-only
```
This prints the passages that would be sent to the LLM.

### 5. Get a free Groq API key and set it

Sign up at https://console.groq.com, then **API Keys → Create API Key**. It's free (with rate limits). The key starts with `gsk_`.

**Windows (PowerShell):**
```powershell
$env:GROQ_API_KEY = "gsk_..."
```
**macOS / Linux:**
```bash
export GROQ_API_KEY="gsk_..."
```
(This lasts for the current terminal only. You can also paste the key into the app's sidebar instead.)

### 6. Ask from the terminal

```bash
python cli.py "What is the hotel limit in Bengaluru?"
python cli.py                                  # interactive mode
python cli.py "..." --docs path/to/your/folder # your own documents
```

### 7. Start the chat app

```bash
streamlit run app.py
```
Open http://localhost:8501, paste your key in the sidebar (or set it as in step 5), upload documents or use the samples, and start asking.

**Try:** *How many sick days do I get?* · *What's the hotel limit in Bengaluru?* · *How long are deleted files kept on the Family plan?* · *Who approves a ₹30,000 expense?*

## Configuration

| Environment variable | Default | Purpose |
|---|---|---|
| `GROQ_API_KEY` | — | your Groq API key |
| `GROQ_MODEL` | `llama-3.3-70b-versatile` | any chat model from https://console.groq.com/docs/models, e.g. `openai/gpt-oss-120b` or `llama-3.1-8b-instant` |

## Deploy free on Hugging Face Spaces

1. At https://huggingface.co go to **New → Space**, choose SDK **Docker** with the **Blank** template, then click **Create Space**.
2. In the Space go to **Settings → Variables and secrets → New secret**: name `GROQ_API_KEY`, value your `gsk_...` key. Use **secret**, not variable, so the key stays hidden. Visitors can then chat without a key. They share your free-tier rate limit, but nobody can see the key.
3. Push this repo to the Space:
   ```bash
   git remote add space https://huggingface.co/spaces/<hf-username>/<space-name>
   git push space main
   ```
   Use a Hugging Face **access token** with *write* permission as the password.
4. After a 2–3 minute build, the app is live at `https://huggingface.co/spaces/<hf-username>/<space-name>`.

## Run with Docker (optional)

```bash
docker build -t doc-chat .
docker run -p 8501:8501 -e GROQ_API_KEY=gsk_... doc-chat
```

## Ideas to extend

- Swap TF-IDF for semantic embeddings (e.g. `sentence-transformers`) plus a vector database (FAISS or Chroma).
- Stream the answer token by token.
- Keep chat history so follow-up questions work.

## Tech stack

Python · Groq API (Llama 3.3 70B) · scikit-learn (TF-IDF) · pypdf · Streamlit · pytest · Docker
