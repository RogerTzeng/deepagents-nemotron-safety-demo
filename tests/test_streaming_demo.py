import asyncio

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, AIMessageChunk
from langchain_core.outputs import ChatGenerationChunk
from nemoguardrails.actions.rail_outcome import RailOutcome
from nemoguardrails.testing.fake_model import FakeLLMModel

from execution_rails_demo import build_execution_agent
from streaming_demo import StreamRejected, build_streaming_rails, guarded_stream


class StreamingModel(GenericFakeChatModel):
    def bind_tools(self, tools, **kwargs):
        return self

    async def _astream(self, messages, **kwargs):
        message = next(self.messages)
        if message.tool_calls:
            yield ChatGenerationChunk(
                message=AIMessageChunk(content="", tool_calls=message.tool_calls)
            )
        else:
            for word in message.content.split(" "):
                yield ChatGenerationChunk(message=AIMessageChunk(content=word + " "))


def make_rails(monkeypatch):
    monkeypatch.setenv("SAFETY_API_KEY", "test-key")
    monkeypatch.setattr(
        "nemoguardrails.rails.llm.llmrails.init_llm_model",
        lambda *args, **kwargs: FakeLLMModel(responses=["Response Safety: safe"]),
    )
    rails = build_streaming_rails()
    rails.config.rails.output.streaming.chunk_size = 4
    rails.config.rails.output.streaming.context_size = 1
    return rails


@pytest.mark.parametrize("mode", ["allow", "block", "error"])
def test_real_deep_agent_stream_inspected_before_release(monkeypatch, mode):
    rails = make_rails(monkeypatch)
    checked = []
    emitted = []

    async def inspect(context, **kwargs):
        text = context["bot_message"]
        checked.append(text)
        if "SECRET" in text:
            if mode == "error":
                raise RuntimeError("Inspector unavailable")
            if mode == "block":
                return RailOutcome.block()
        return RailOutcome.allow()

    rails.register_action(inspect, name="content_safety_check_output")
    model = StreamingModel(
        messages=iter(
            [
                AIMessage(
                    content="",
                    tool_calls=[
                        {"name": "fetch_reference", "args": {"scenario": "public"}, "id": "call-1"}
                    ],
                ),
                AIMessage(content="Public information is useful. SECRET must stay private."),
            ]
        )
    )
    agent, guard = build_execution_agent(model)

    async def consume():
        async for chunk in guarded_stream(agent, rails, "Fetch public"):
            assert checked, "No raw text may be emitted before inspection"
            emitted.append(chunk)

    if mode == "allow":
        asyncio.run(consume())
        assert "".join(emitted).strip() == "Public information is useful. SECRET must stay private."
    else:
        with pytest.raises(
            StreamRejected,
            match=("content_blocked" if mode == "block" else "rail_execution_failure"),
        ):
            asyncio.run(consume())
        assert emitted  # Earlier approved output cannot be retracted.
        assert "SECRET" not in "".join(emitted)
    assert len(checked) > 1
    assert [v["stage"] for v in guard.verdicts] == ["tool_input", "tool_output"]


def test_streaming_reuses_custom_policy(monkeypatch):
    rails = make_rails(monkeypatch)
    model = next(m for m in rails.config.models if m.type == "content_safety")
    assert model.parameters["chat_template_kwargs"]["custom_policy"]
    assert rails.config.rails.output.streaming.stream_first is False


@pytest.mark.parametrize("safety", ["safe", "unsafe"])
def test_native_content_safety_action(monkeypatch, safety):
    rails = make_rails(monkeypatch)
    classifier = FakeLLMModel(responses=[f"User Safety: safe\nResponse Safety: {safety}"])
    rails.register_action_param("llms", {"content_safety": classifier})
    agent, _ = build_execution_agent(
        StreamingModel(messages=iter([AIMessage(content="A short answer.")]))
    )

    async def consume():
        return [chunk async for chunk in guarded_stream(agent, rails, "Explain lithography")]

    if safety == "unsafe":
        with pytest.raises(StreamRejected, match="content_blocked"):
            asyncio.run(consume())
    else:
        assert "".join(asyncio.run(consume())).strip() == "A short answer."
    assert classifier.inference_count > 0
