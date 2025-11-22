"""Retrieval and generation for the chatbot."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

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

    async def answer(self, question: str, limit: int = 5) -> str:
        contexts = await self.retrieve(question, limit=limit)
        context_block = "\n\n".join(
            [f"[source={c.source} file={c.filename} chunk={c.chunk_id}] {c.text}" for c in contexts]
        )
        prompt = (
            "You are a concise assistant. Use the provided context to answer the question. "
            "If the answer is not in the context, say you do not know.\n\n"
            f"Context:\n{context_block}\n\nQuestion: {question}\nAnswer:"
        )
        completion = await self.client.chat.completions.create(
            model=settings.openai.chat_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
        return completion.choices[0].message.content or ""
