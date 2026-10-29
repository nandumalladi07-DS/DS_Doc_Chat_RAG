"""Ask questions from the terminal.

Usage:
    python cli.py "How many sick days do I get?"                 # uses sample_docs/
    python cli.py "What is the hotel limit?" --docs path/to/folder
    python cli.py "Refund policy?" --retrieve-only               # no API key needed
    python cli.py                                                # interactive mode
"""

import argparse
import os
from pathlib import Path

import groq

from rag import DocumentQA

SAMPLE_DOCS = Path(__file__).parent / "sample_docs"


def show(qa: DocumentQA, question: str, retrieve_only: bool):
    if retrieve_only:
        for n, (chunk, score) in enumerate(qa.retrieve(question), start=1):
            print(f"\n[{n}] {chunk.source}  (score {score:.2f})\n{chunk.text[:400]}")
        return
    try:
        result = qa.ask(question)
    except groq.AuthenticationError:
        print("Invalid API key. Check GROQ_API_KEY.")
        return
    except groq.RateLimitError:
        print("Rate limited (free tier). Wait a minute and try again.")
        return
    except groq.APIConnectionError:
        print("Network error: could not reach the Groq API.")
        return
    except groq.APIStatusError as e:
        print(f"Groq API error ({e.status_code}): {e.message}")
        return
    print(f"\n{result['answer']}\n")
    for n, (chunk, score) in enumerate(result["sources"], start=1):
        print(f"  [{n}] {chunk.source} (score {score:.2f})")


def main():
    parser = argparse.ArgumentParser(description="Chat with your documents")
    parser.add_argument("question", nargs="?", help="leave out for interactive mode")
    parser.add_argument("--docs", default=str(SAMPLE_DOCS), help="folder with .pdf/.txt/.md files")
    parser.add_argument("--retrieve-only", action="store_true",
                        help="only show the matching passages (no API key needed)")
    args = parser.parse_args()

    if not args.retrieve_only and not os.environ.get("GROQ_API_KEY"):
        parser.error("set GROQ_API_KEY first, or use --retrieve-only")

    qa = DocumentQA.from_folder(args.docs)
    print(f"Indexed {len(qa.documents)} file(s), {len(qa.retriever.chunks)} chunks from {args.docs}")

    if args.question:
        show(qa, args.question, args.retrieve_only)
        return
    print("Type a question (empty line to quit).")
    while question := input("\n> ").strip():
        show(qa, question, args.retrieve_only)


if __name__ == "__main__":
    main()
