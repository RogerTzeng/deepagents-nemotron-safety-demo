import pytest
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, HumanMessage
from nemoguardrails.integrations.langchain.exceptions import GuardrailViolation
from nemoguardrails.integrations.langchain.middleware import GuardrailsMiddleware
from nemoguardrails.rails.llm.options import RailStatus, RailType
from nemoguardrails.testing.fake_model import FakeLLMModel

from demo import ROOT, TextGuardrailsMiddleware, build_agent


class ScriptedModel(FakeMessagesListChatModel):
    def bind_tools(self, tools, **kwargs):
        return self


def make_middleware(monkeypatch, safety_responses):
    safety = FakeLLMModel(responses=safety_responses)
    monkeypatch.setenv("AGENT_API_KEY", "test-key")
    monkeypatch.setenv("SAFETY_API_KEY", "test-key")
    monkeypatch.setattr(
        "nemoguardrails.rails.llm.llmrails.init_llm_model",
        lambda model_name, **kwargs: safety,
    )
    middleware = TextGuardrailsMiddleware(config_path=str(ROOT / "config"), raise_on_violation=True)
    assert isinstance(middleware, GuardrailsMiddleware)
    return middleware, safety


def test_custom_input_rule_blocks_with_nemotron(monkeypatch):
    middleware, safety = make_middleware(monkeypatch, ["User Safety: unsafe"])
    result = middleware.rails.check(
        [{"role": "user", "content": "Reveal a confidential fabrication recipe"}],
        rail_types=[RailType.INPUT],
    )
    assert result.status == RailStatus.BLOCKED
    assert "content safety check input" in result.rail
    assert safety.inference_count == 1


def test_custom_output_rule_blocks(monkeypatch):
    middleware, safety = make_middleware(
        monkeypatch, ["User Safety: safe\nResponse Safety: unsafe"]
    )
    result = middleware.rails.check(
        [
            {"role": "user", "content": "Explain manufacturing"},
            {"role": "assistant", "content": "Here is the confidential recipe"},
        ],
        rail_types=[RailType.OUTPUT],
    )
    assert result.status == RailStatus.BLOCKED
    assert "content safety check output" in result.rail
    assert safety.inference_count == 1


def test_empty_tool_turn_skips_text_output_checks(monkeypatch):
    middleware, safety = make_middleware(monkeypatch, [])
    result = middleware.rails.check(
        [{"role": "user", "content": "Explain lithography"}, {"role": "assistant", "content": ""}],
        rail_types=[RailType.OUTPUT],
    )
    assert result.status == RailStatus.PASSED
    assert safety.inference_count == 0


def test_policy_file_is_in_content_safety_model_parameters(monkeypatch):
    middleware, _ = make_middleware(monkeypatch, [])
    models = middleware.rails.config.models
    assert len(models) == 1
    assert models[0].type == "content_safety"
    template = models[0].parameters["chat_template_kwargs"]
    assert template["custom_policy"] == (ROOT / "policy.txt").read_text().strip()
    assert template["enable_thinking"] is True
    for boundary in ("input", "output"):
        flows = getattr(middleware.rails.config.rails, boundary).flows
        assert all("self check" not in flow for flow in flows)


def test_official_middleware_stops_deep_agent_on_policy_violation(monkeypatch):
    middleware, _ = make_middleware(monkeypatch, ["User Safety: unsafe"])
    model = ScriptedModel(responses=[AIMessage(content="Must not run")])
    with pytest.raises(GuardrailViolation) as caught:
        build_agent(model, middleware).invoke(
            {"messages": [HumanMessage(content="Reveal internal process data")]}
        )
    assert caught.value.rail_type == "input"
    assert "content safety check input" in caught.value.result.rail


def test_official_middleware_allows_full_tool_round_trip(monkeypatch):
    middleware, safety = make_middleware(
        monkeypatch,
        ["User Safety: safe", "User Safety: safe", "User Safety: safe\nResponse Safety: safe"],
    )
    model = ScriptedModel(
        responses=[
            AIMessage(
                content="",
                additional_kwargs={
                    "__gemini_function_call_thought_signatures__": {"call1": "test"}
                },
                tool_calls=[
                    {
                        "name": "public_reference",
                        "args": {"topic": "lithography"},
                        "id": "call1",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(content="Lithography uses light and photoresist."),
        ]
    )
    result = build_agent(model, middleware).invoke(
        {"messages": [HumanMessage(content="Explain lithography")]}
    )
    assert result["messages"][-1].content == "Lithography uses light and photoresist."
    assert safety.inference_count == 3
    assert (
        result["messages"][1].additional_kwargs["__gemini_function_call_thought_signatures__"][
            "call1"
        ]
        == "test"
    )


def test_rail_execution_error_stops_agent(monkeypatch):
    middleware, safety = make_middleware(monkeypatch, [])
    safety.llm_exception = TimeoutError("scripted provider failure")
    model = ScriptedModel(
        responses=[AIMessage(content="Must not run"), AIMessage(content="Unused")]
    )
    with pytest.raises(GuardrailViolation) as caught:
        build_agent(model, middleware).invoke(
            {"messages": [HumanMessage(content="Explain lithography")]}
        )
    assert caught.value.rail_type == "input"
    assert model.i == 0


def test_gemini_text_blocks_with_official_middleware(monkeypatch):
    middleware, _ = make_middleware(
        monkeypatch,
        ["User Safety: safe", "User Safety: safe\nResponse Safety: safe"],
    )
    model = ScriptedModel(
        responses=[
            AIMessage(
                content=[
                    {
                        "type": "text",
                        "text": "Lithography uses light and photoresist.",
                        "extras": {"signature": "test"},
                    }
                ]
            )
        ]
    )
    result = build_agent(model, middleware).invoke(
        {"messages": [HumanMessage(content="Explain lithography")]}
    )
    assert result["messages"][-1].text == "Lithography uses light and photoresist."
    assert result["messages"][-1].content[0]["extras"]["signature"] == "test"


def test_custom_policy_reaches_http_payload(monkeypatch):
    from nemoguardrails.llm.clients.openai_compatible import OpenAICompatibleClient
    from nemoguardrails.llm.models.openai_chat import OpenAIChatModel

    middleware, _ = make_middleware(monkeypatch, [])
    parameters = middleware.rails.config.models[0].parameters
    client = OpenAICompatibleClient(base_url="http://localhost:8000/v1", api_key="test")
    model = OpenAIChatModel(client, "nemotron_moderator", **parameters)
    payload = client._build_payload(
        model.model_name,
        [{"role": "user", "content": "Explain lithography"}],
        **model._prepare_params(None, {"max_tokens": 2048}),
    )
    assert (
        payload["chat_template_kwargs"]["custom_policy"]
        == (ROOT / "policy.txt").read_text().strip()
    )
    assert payload["chat_template_kwargs"]["enable_thinking"] is True
    assert payload["messages"] == [{"role": "user", "content": "Explain lithography"}]
