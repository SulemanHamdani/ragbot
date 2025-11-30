"""User-facing runner that executes the RAG agent."""
from __future__ import annotations

from typing import List, Sequence, Tuple

from .agent import build_agent
from .tools import AgentDeps
from .metrics import compute_metrics


class AgentRunner:
    def __init__(self, agent=None, deps: AgentDeps | None = None):
        if agent is None and deps is None:
            agent, deps = build_agent()
        if agent is None or deps is None:
            raise ValueError("Both agent and deps must be provided together.")
        self.agent = agent
        self.deps = deps

    async def answer(
        self,
        question: str,
        limit: int = 5,
        conversation_history: Sequence[Tuple[str, str]] | None = None,
    ) -> str:
        prompt = self._build_prompt(question, conversation_history, limit)
        result = await self.agent.run(prompt, deps=self.deps)
        answer = result.output
        print(f"[answer] {answer}")
        print("[metrics] running evals...")
        metrics = await compute_metrics(question, answer or "")
        print(f"[metrics] {metrics}")
        return answer

    def _build_prompt(
        self,
        question: str,
        history: Sequence[Tuple[str, str]] | None,
        limit: int,
    ) -> str:
        history_text = _format_history(history)
        return (
            f"Conversation so far:\n{history_text}\n\n"
            f"User question: {question}\n"
            f"First call `vector_search` with query=the question text and limit={limit} to fetch context. "
            "If vector search returns nothing useful, call `web_search` to gather public web snippets. "
            "Then answer concisely using only the retrieved context. If nothing relevant is returned, say 'I do not know based on the provided context.'"
        )


def _format_history(history: Sequence[Tuple[str, str]] | None) -> str:
    if not history:
        return "(no prior turns)"
    lines: List[str] = []
    for question, answer in history:
        lines.append(f"User: {question}")
        lines.append(f"Assistant: {answer}")
    return "\n".join(lines)


__all__ = ["AgentRunner"]
