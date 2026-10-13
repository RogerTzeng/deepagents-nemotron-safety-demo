import asyncio
import json

import pytest
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from pydantic import Field

from execution_rails_demo import build_execution_agent, inspect_with_rails, run_demo


class RecordingModel(FakeMessagesListChatModel):
    calls: list = Field(default_factory=list)

    def bind_tools(self, tools, **kwargs):
        return self

    async def _agenerate(self, messages, **kwargs):
        self.calls.append(list(messages))
        return await super()._agenerate(messages, **kwargs)


def scripted_model(scenario, name="fetch_reference"):
    return RecordingModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[{"name": name, "args": {"scenario": scenario}, "id": "agent-call-123"}],
            ),
            AIMessage(content="A public reference summary."),
        ]
    )


@pytest.mark.parametrize(
    "scenario,status,calls",
    [
        ("public", "passed", 2),
        ("credentials", "modified", 2),
        ("injection", "blocked", 1),
        ("malformed", "blocked", 1),
        ("tool_error", "blocked", 1),
    ],
)
def test_real_agent_tool_loop(scenario, status, calls):
    model = scripted_model(scenario)
    verdict, answer = asyncio.run(run_demo(scenario, model))
    assert verdict["status"] == status
    assert verdict["tool_call_id"] == "agent-call-123"
    assert len(model.calls) == calls
    assert not any(isinstance(m, ToolMessage) for m in model.calls[0])
    if calls == 2:
        assert answer == "A public reference summary."
        result = model.calls[1][-1]
        assert isinstance(result, ToolMessage)
        assert result.tool_call_id == "agent-call-123"
        assert result.name == "fetch_reference"
        assert result.content == verdict["content"]
        assert "DEMO_ONLY_TOKEN_123456" not in result.content
        if scenario == "credentials":
            assert "[REDACTED]" in result.content
    else:
        assert answer is None
        assert "content" not in verdict


def test_execution_order_and_sanitized_graph_state(monkeypatch):
    import execution_rails_demo as demo

    model = scripted_model("credentials")
    original = demo.fetch_reference
    executed = []

    def fetch_reference(scenario: str) -> str:
        """Return the synthetic reference."""
        assert len(model.calls) == 1
        executed.append(scenario)
        return original(scenario)

    monkeypatch.setattr(demo, "fetch_reference", fetch_reference)
    agent, guard = build_execution_agent(model)
    state = asyncio.run(agent.ainvoke({"messages": [HumanMessage(content="credentials")]}))
    assert executed == ["credentials"]
    assert [(v["stage"], v["status"]) for v in guard.verdicts] == [
        ("tool_input", "passed"),
        ("tool_output", "modified"),
    ]
    assert "DEMO_ONLY_TOKEN_123456" not in repr(state)
    assert "[REDACTED]" in repr(state)


def test_uninspected_fields_are_not_forwarded():
    raw = json.dumps({"status": "ok", "content": "Public information.", "secret": "DO_NOT_FORWARD"})
    verdict = asyncio.run(inspect_with_rails(raw))
    assert verdict["allowed"]
    assert "DO_NOT_FORWARD" not in verdict["content"]


def test_inspection_failure_stops_agent_continuation(monkeypatch):
    async def fail(tool_message):
        raise RuntimeError("scripted inspector failure")

    monkeypatch.setattr("execution_rails_demo.inspect_reference_result", fail)
    model = scripted_model("public")
    verdict, answer = asyncio.run(run_demo("public", model))
    assert verdict["status"] == "error"
    assert answer is None
    assert len(model.calls) == 1


def test_other_tools_cannot_bypass_inspection():
    model = scripted_model("public", name="write_file")
    verdict, answer = asyncio.run(run_demo("public", model))
    assert verdict["status"] == "blocked"
    assert verdict["reason"] == "Tool not enabled in demo"
    assert answer is None
    assert len(model.calls) == 1


def test_no_tool_call_is_not_reported_as_success():
    model = RecordingModel(responses=[AIMessage(content="Skipped the tool")])
    with pytest.raises(RuntimeError, match="did not call"):
        asyncio.run(run_demo("public", model))


@pytest.mark.parametrize(
    "args",
    [
        {},
        {"scenario": 123},
        {"scenario": "input_blocked"},
        {"scenario": "public", "sql": "SELECT * FROM secrets"},
    ],
)
def test_input_rejection_prevents_tool_execution(monkeypatch, args):
    executed = []

    def fetch_reference(scenario: str) -> str:
        """Spy on actual tool execution."""
        executed.append(scenario)
        return "must not execute"

    monkeypatch.setattr("execution_rails_demo.fetch_reference", fetch_reference)
    model = scripted_model("public")
    model.responses[0].tool_calls[0]["args"] = args
    verdict, answer = asyncio.run(run_demo("public", model))
    assert verdict["stage"] == "tool_input"
    assert verdict["status"] == "blocked"
    assert verdict["reason"] == "Unsupported tool arguments"
    assert not executed
    assert answer is None
    assert len(model.calls) == 1


def test_input_inspection_failure_prevents_tool_execution(monkeypatch):
    inspected = []
    executed = []

    async def fail(tool_calls):
        inspected.append(tool_calls)
        raise RuntimeError("scripted input inspector failure")

    def fetch_reference(scenario: str) -> str:
        """Spy on actual tool execution."""
        executed.append(scenario)
        return "must not execute"

    monkeypatch.setattr("execution_rails_demo.inspect_reference_call", fail)
    monkeypatch.setattr("execution_rails_demo.fetch_reference", fetch_reference)
    model = scripted_model("public")
    verdict, answer = asyncio.run(run_demo("public", model))
    assert inspected
    assert not executed
    assert verdict["stage"] == "tool_input"
    assert verdict["status"] == "error"
    assert answer is None
    assert len(model.calls) == 1
