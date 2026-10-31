---
title: Chat with your Docs
emoji: 📄
colorFrom: indigo
colorTo: purple
sdk: docker
app_port: 8501
pinned: false
---

# 📄 Chat with your Docs (RAG with Groq + GPT-OSS 120B)

Upload PDFs, text or Markdown files and ask questions in plain English. The app finds the most relevant passages and has an LLM (**OpenAI GPT-OSS 120B on Groq**, with a free API key) answer **only from those passages**, with `[1]` `[2]` citations you can expand to check.

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
- **Fully offline tests.** The LLM is replaced by a fake client, so the 27 tests need no key and cost nothing.

## Project structure

```
DS_Doc_Chat_RAG/
├── rag.py            # read → chunk → retrieve → ask the LLM (Groq)
├── app.py            # Streamlit chat UI (upload files, chat, see sources)
├── cli.py            # terminal version (+ --retrieve-only mode, no key needed)
├── sample_docs/      # 3 demo docs: leave policy, expense policy, product FAQ
├── test_files/       # 2 PDFs + 1 TXT to try uploading (see "Test files" below)
├── tests/
│   ├── test_rag.py   # chunking, retrieval accuracy, prompt building, fake-LLM tests
│   └── test_app.py   # chat UI smoke tests (no browser, no key)
├── requirements.txt
└── Dockerfile        # to run anywhere with Docker
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
Expected: `27 passed`.

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
Open http://localhost:8501, paste your key in the sidebar (or set it as in step 5), upload documents or use the samples, and start asking. To test uploads, use the files in [`test_files/`](test_files/) (see [Test files](#test-files-documents-to-upload) for questions and expected answers).

**Try:** *How many sick days do I get?* · *What's the hotel limit in Bengaluru?* · *How long are deleted files kept on the Family plan?* · *Who approves a ₹30,000 expense?*

## Test files: documents to upload

Upload these in the app's sidebar (**Upload PDF / TXT / MD**) to test real file uploads, including PDF text extraction. Untick **Include sample documents** if you want answers from these files only.

| File | Type | About |
|---|---|---|
| `it_security_policy.pdf` | PDF, 2 pages | passwords, MFA, laptops, USB, VPN, phishing, data classification |
| `health_insurance_guide.pdf` | PDF, 2 pages | who is covered, sum insured, room rent, maternity, claims |
| `onboarding_guide.txt` | Text | first day, laptop, buddy, ID card, probation, payroll, Slack |

### Questions to try (and the answer you should get)

| Question | Expected answer |
|---|---|
| How long must my password be? | at least 14 characters |
| What should I do if I lose my laptop? | report it to the IT helpdesk within 2 hours |
| Are USB drives allowed? | no, USB storage is blocked |
| How do I report a phishing email? | Report Phishing button or security@nimbuslabs.example within 1 hour |
| What is the hospital room rent limit? | 1% of the sum insured per day (Rs. 5,000) |
| Is dental treatment covered? | no, except after an accident |
| How many days do I have to submit a reimbursement claim? | 30 days after discharge |
| What is the maternity cover? | up to Rs. 75,000 per delivery |
| What time should I arrive on my first day? | 9:30 AM at the Bengaluru office reception |
| How long is probation? | 6 months |
| Which Slack channel should I use for laptop problems? | #it-help |
| What is the company's stock price? | *should say it couldn't find this in the documents* |

From the terminal:
```bash
python cli.py "Is dental treatment covered?" --docs test_files
```

## Configuration

| Environment variable | Default | Purpose |
|---|---|---|
| `GROQ_API_KEY` | — | your Groq API key |
| `GROQ_MODEL` | `openai/gpt-oss-120b` | any chat model from https://console.groq.com/docs/models. If the model gets retired, the app switches to a working one automatically |

## Deploy free on Streamlit Community Cloud

1. Sign in at https://share.streamlit.io with your GitHub account.
2. Click **Create app → Deploy a public app from GitHub**.
3. Fill in **Repository** `<your-username>/DS_Doc_Chat_RAG`, **Branch** `main` and **Main file path** `app.py`.
4. Under **Advanced settings → Secrets**, paste:
   ```toml
   GROQ_API_KEY = "gsk_your_key_here"
   ```
   Visitors can then chat without entering a key, and the key is never shown. Skip this if visitors should paste their own key in the sidebar.
5. Click **Deploy**.

After a few minutes the app is live at a public `https://<name>.streamlit.app` link. Apps with no visitors for 12 hours go to sleep; the next visitor wakes them with one click.

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

Python · Groq API (GPT-OSS 120B) · scikit-learn (TF-IDF) · pypdf · Streamlit · pytest · Docker
