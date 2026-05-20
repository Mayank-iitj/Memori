"""CrewAI memory integration backed by Memori."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from memori import Memori


class CrewAIMemory:
    """Persist and recall CrewAI context with Memori.

    This helper captures CrewAI task/output pairs through Memori's agent
    ingestion endpoint and exposes simple recall/context helpers for later runs.
    """

    def __init__(
        self,
        memori: Memori,
        *,
        project_id: str,
        session_id: str | None = None,
        signal: str = "decision",
        source: str = "fact",
    ) -> None:
        """Create a CrewAI memory bridge.

        Args:
            memori: Configured Memori instance.
            project_id: Memori Cloud project identifier used for agent capture/recall.
            session_id: Optional fixed session identifier for this memory instance.
            signal: Optional agent recall signal filter used in cloud recall.
            source: Optional agent recall source filter used in cloud recall.
        """
        if not isinstance(project_id, str) or not project_id.strip():
            raise ValueError("project_id must be a non-empty string")

        self.memori = memori
        self.project_id = project_id
        self.session_id = session_id
        self.signal = signal
        self.source = source

    def save(
        self,
        task: str,
        output: str,
        *,
        agent_role: str | None = None,
        crew_name: str | None = None,
        trace: dict[str, Any] | None = None,
        summary: str | None = None,
        session_id: str | None = None,
        provider: str | None = None,
        model: str | None = None,
        provider_sdk_version: str | None = None,
    ) -> None:
        """Persist a CrewAI execution trace with Memori augmentation.

        Args:
            task: Task prompt/input assigned to the agent.
            output: Agent output/decision for the task.
            agent_role: Optional CrewAI role for contextual attribution.
            crew_name: Optional crew/team name for context.
            trace: Optional additional trace payload merged into base context.
            summary: Optional session summary metadata.
            session_id: Optional explicit session identifier override.
            provider: Optional LLM provider for augmentation metadata.
            model: Optional LLM model/version for augmentation metadata.
            provider_sdk_version: Optional provider SDK version metadata.
        """
        resolved_trace: dict[str, Any] = {}
        if trace is not None:
            resolved_trace.update(trace)
        if agent_role is not None:
            resolved_trace.setdefault("agent_role", agent_role)
        if crew_name is not None:
            resolved_trace.setdefault("crew_name", crew_name)

        self.memori.capture_agent_turn(
            user_content=task,
            assistant_content=output,
            project_id=self.project_id,
            session_id=session_id or self.session_id,
            platform="crewai",
            trace=resolved_trace or None,
            summary=summary,
            provider=provider,
            model=model,
            provider_sdk_version=provider_sdk_version,
        )

    async def asave(self, *args: Any, **kwargs: Any) -> None:
        """Async wrapper for :meth:`save`."""
        await asyncio.to_thread(self.save, *args, **kwargs)

    def recall(
        self,
        query: str,
        *,
        limit: int | None = None,
        session_id: str | None = None,
    ) -> Any:
        """Recall memories for a CrewAI query.

        Cloud mode uses the agent recall endpoint scoped by project/session.
        BYODB mode uses ``memori.recall(query, limit=...)``.
        """
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")

        if self.memori.config.cloud:
            return self.memori.agent_recall(
                query=query,
                project_id=self.project_id,
                session_id=session_id or self.session_id,
                signal=self.signal,
                source=self.source,
            )
        return self.memori.recall(query, limit=limit)

    async def arecall(self, *args: Any, **kwargs: Any) -> Any:
        """Async wrapper for :meth:`recall`."""
        return await asyncio.to_thread(self.recall, *args, **kwargs)

    def build_context(self, query: str, *, limit: int | None = None) -> str:
        """Build a newline-delimited memory context string for CrewAI prompts."""
        payload = self.recall(query, limit=limit)
        contents = self._extract_contents(payload)
        return "\n".join(f"- {content}" for content in contents)

    async def abuild_context(self, query: str, *, limit: int | None = None) -> str:
        """Async wrapper for :meth:`build_context`."""
        return await asyncio.to_thread(self.build_context, query, limit=limit)

    @staticmethod
    def _extract_contents(payload: Any) -> list[str]:
        if isinstance(payload, dict):
            facts = payload.get("facts", [])
            if isinstance(facts, list):
                return [
                    str(item["content"])
                    for item in facts
                    if isinstance(item, dict) and "content" in item
                ]
            return []

        if isinstance(payload, list):
            extracted: list[str] = []
            for item in payload:
                if isinstance(item, dict) and "content" in item:
                    extracted.append(str(item["content"]))
            return extracted

        return []
