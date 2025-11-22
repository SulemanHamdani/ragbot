"""CLI for querying the chatbot."""
from __future__ import annotations

import argparse
import asyncio
from datetime import datetime
from pathlib import Path
from typing import List

from src.config import settings
from src.rag.chatbot import RAGChatbot


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ask questions against the Qdrant-backed RAG chatbot.")
    parser.add_argument("questions", nargs="*", help="Questions to ask")
    parser.add_argument("--file", type=Path, help="Optional file with one question per line")
    parser.add_argument("--collection", type=str, default=settings.qdrant.collection_name, help="Qdrant collection name")
    parser.add_argument("--limit", type=int, default=5, help="Number of context chunks to retrieve")
    return parser.parse_args()


def load_questions(args: argparse.Namespace) -> List[str]:
    questions: List[str] = []
    if args.file:
        questions.extend([line.strip() for line in args.file.read_text().splitlines() if line.strip()])
    questions.extend(args.questions)
    return questions


async def main() -> None:
    args = parse_args()
    settings.qdrant.collection_name = args.collection
    questions = load_questions(args)
    if not questions:
        raise SystemExit("Provide questions via CLI or --file.")
    chatbot = RAGChatbot()
    timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    log_path = Path("logs") / f"answers-{timestamp}.log"
    responses = []
    for question in questions:
        answer = await chatbot.answer(question, limit=args.limit)
        responses.append((question, answer))
    log_lines = [f"Q: {q}\nA: {a}\n" for q, a in responses]
    log_path.write_text("\n".join(log_lines))
    print(f"Saved {len(responses)} answers to {log_path}")


if __name__ == "__main__":
    asyncio.run(main())
