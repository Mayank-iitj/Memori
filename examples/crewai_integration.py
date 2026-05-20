"""Memori + CrewAI integration example with shared persistent memory."""

from __future__ import annotations

import os

from memori import Memori
from memori.llm.clients import CrewAIMemory


def main() -> None:
    """Run a minimal CrewAI + Memori memory flow."""
    try:
        from crewai import Agent, Crew, Process, Task
    except ImportError as exc:  # pragma: no cover - example only
        raise RuntimeError(
            "CrewAI is required for this example. Install with: pip install crewai"
        ) from exc

    memori = Memori(api_key=os.environ["MEMORI_API_KEY"]).attribution(
        entity_id="customer-123",
        process_id="crewai-support",
    )
    memory = CrewAIMemory(memori, project_id="support-project")

    researcher = Agent(
        role="Research Specialist",
        goal="Gather user-specific context before replying",
        backstory="You collect relevant historical details for the support team.",
        verbose=True,
    )
    responder = Agent(
        role="Support Specialist",
        goal="Answer with personalized context",
        backstory="You deliver concise support responses based on prior memory.",
        verbose=True,
    )

    query = "How should I respond to this customer renewal question?"
    context = memory.build_context(query)
    task = Task(
        description=(
            "Use the following prior context to craft a customer response:\n"
            f"{context or '- No prior context found.'}"
        ),
        expected_output="A concise response tailored to the customer context.",
        agent=responder,
    )

    crew = Crew(
        agents=[researcher, responder],
        tasks=[task],
        process=Process.sequential,
        verbose=True,
    )
    result = crew.kickoff()

    memory.save(
        task.description,
        str(result),
        agent_role=responder.role,
        crew_name="support_crew",
        trace={"process": "sequential"},
    )
    memori.augmentation.wait()
    print(result)


if __name__ == "__main__":
    main()
