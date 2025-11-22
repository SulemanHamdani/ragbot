"""Retrieval and generation for the chatbot."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

from openai import AsyncOpenAI

from src.config import settings
from src.embeddings.openai_embeddings import embed_texts
from src.vectorstore.qdrant_store import create_client, search_similar


@dataclass(slots=True)
class RetrievedContext:
    text: str
    score: float
    source: str
    filename: str
    chunk_id: int


class RAGChatbot:
    def __init__(self, client: AsyncOpenAI | None = None):
        self.client = client or AsyncOpenAI(api_key=settings.openai.api_key)
        self.qdrant = create_client()

    async def retrieve(self, question: str, limit: int = 5) -> List[RetrievedContext]:
        [query_embedding] = await embed_texts([question], client=self.client)
        results = search_similar(self.qdrant, query_embedding, limit=limit)
        contexts: List[RetrievedContext] = []
        for hit in results:
            payload = hit.payload or {}
            contexts.append(
                RetrievedContext(
                    text=str(payload.get("text", "")),
                    score=hit.score,
                    source=str(payload.get("source", "")),
                    filename=str(payload.get("filename", "")),
                    chunk_id=int(payload.get("chunk_id", 0)),
                )
            )
        return contexts

    async def answer(
        self,
        question: str,
        limit: int = 5,
        conversation_history: Sequence[Tuple[str, str]] | None = None,
    ) -> str:
        contexts = await self.retrieve(question, limit=limit)
        context_block = "\n\n".join(
            [f"[source={c.source} file={c.filename} chunk={c.chunk_id}] {c.text}" for c in contexts]
        )
        history_text = _format_history(conversation_history)
        prompt = f"""
        You are a concise, highly reliable assistant.

        Use ONLY the information in the provided context and conversation history to answer the question.
        Do NOT invent facts, names, numbers, or assumptions that are not supported by the context.

        If the answer is not clearly supported by the context, reply exactly:
        "I do not know based on the provided context."

        Resolve references like "he", "she", "they", "it", or "this" using the conversation history and context.
        - If you can clearly resolve the reference, use the correct entity.
        - If the reference is ambiguous or cannot be resolved with high confidence, say that it is ambiguous and do NOT guess.

        Follow these rules:
        1. Base your answer only on the context and conversation history below.
        2. If multiple interpretations are possible, briefly mention the ambiguity.
        3. Be as concise as possible while still being clear.
        4. Do NOT answer with information from outside the context, even if you know it.

        Conversation history:
        {history_text}

        Retrieved context:
        {context_block}

        Question:
        {question}

        Answer:
        """

        completion = await self.client.chat.completions.create(
            model=settings.openai.chat_model,
            messages=[{"role": "user", "content": prompt}],
        )
        return completion.choices[0].message.content or ""


def _format_history(history: Sequence[Tuple[str, str]] | None) -> str:
    if not history:
        return "None."
    lines: List[str] = []
    for question, answer in history:
        lines.append(f"User: {question}")
        lines.append(f"Assistant: {answer}")
    return "\n".join(lines)
