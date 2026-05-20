from types import SimpleNamespace

import pytest

from memori.llm.clients import CrewAIMemory


def _mock_memori(mocker, *, cloud: bool):
    return SimpleNamespace(
        config=SimpleNamespace(cloud=cloud),
        capture_agent_turn=mocker.MagicMock(),
        agent_recall=mocker.MagicMock(return_value={"facts": []}),
        recall=mocker.MagicMock(return_value=[]),
    )


def test_crewai_memory_requires_non_empty_project_id(mocker):
    memori = _mock_memori(mocker, cloud=True)

    with pytest.raises(ValueError, match="project_id must be a non-empty string"):
        CrewAIMemory(memori, project_id="")


def test_crewai_memory_save_captures_agent_turn(mocker):
    memori = _mock_memori(mocker, cloud=True)
    memory = CrewAIMemory(memori, project_id="proj-1", session_id="session-1")

    memory.save(
        "Summarize latest incident",
        "Incident linked to stale deploy cache",
        agent_role="incident_commander",
        crew_name="ops_crew",
        trace={"tool_calls": ["pagerduty_lookup"]},
    )

    memori.capture_agent_turn.assert_called_once_with(
        user_content="Summarize latest incident",
        assistant_content="Incident linked to stale deploy cache",
        project_id="proj-1",
        session_id="session-1",
        platform="crewai",
        trace={
            "tool_calls": ["pagerduty_lookup"],
            "agent_role": "incident_commander",
            "crew_name": "ops_crew",
        },
        summary=None,
        provider=None,
        model=None,
        provider_sdk_version=None,
    )


def test_crewai_memory_recall_uses_agent_endpoint_in_cloud_mode(mocker):
    memori = _mock_memori(mocker, cloud=True)
    memory = CrewAIMemory(
        memori,
        project_id="proj-1",
        session_id="session-1",
        signal="decision",
        source="fact",
    )

    memory.recall("deployment rollback")

    memori.agent_recall.assert_called_once_with(
        query="deployment rollback",
        project_id="proj-1",
        session_id="session-1",
        signal="decision",
        source="fact",
    )
    memori.recall.assert_not_called()


def test_crewai_memory_recall_uses_byodb_recall(mocker):
    memori = _mock_memori(mocker, cloud=False)
    memory = CrewAIMemory(memori, project_id="proj-1")

    memory.recall("customer prefers sms", limit=3)

    memori.recall.assert_called_once_with("customer prefers sms", limit=3)
    memori.agent_recall.assert_not_called()


def test_crewai_memory_build_context_formats_facts(mocker):
    memori = _mock_memori(mocker, cloud=True)
    memori.agent_recall.return_value = {
        "facts": [{"content": "User prefers email"}, {"content": "Plan is annual"}]
    }
    memory = CrewAIMemory(memori, project_id="proj-1")

    context = memory.build_context("how should I respond?")

    assert context == "- User prefers email\n- Plan is annual"


@pytest.mark.asyncio
async def test_crewai_memory_async_wrappers(mocker):
    memori = _mock_memori(mocker, cloud=True)
    memory = CrewAIMemory(memori, project_id="proj-1")
    to_thread = mocker.patch("memori.llm.clients.crewai.asyncio.to_thread")

    await memory.asave("task", "output")
    await memory.arecall("query")
    await memory.abuild_context("query")

    assert to_thread.call_count == 3
