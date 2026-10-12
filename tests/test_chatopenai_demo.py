import asyncio

import httpx
import pytest
from langchain_openai import ChatOpenAI
from nemoguardrails.testing.fake_model import FakeLLMModel

from chatopenai_demo import build_chain


def make_chat_model(calls):
    async def respond(request):
        calls.append(request)
        return httpx.Response(
            200,
            json={
                "id": "test-completion",
                "object": "chat.completion",
                "model": "test-model",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "Lithography uses light and photoresist.",
                        },
                        "finish_reason": "stop",
                    }
                ],
            },
        )

    return ChatOpenAI(
        model="test-model",
        api_key="test-key",
        max_retries=0,
        http_async_client=httpx.AsyncClient(transport=httpx.MockTransport(respond)),
    )


async def invoke_chain(model_calls):
    chain = build_chain(make_chat_model(model_calls))
    return await asyncio.wait_for(chain.ainvoke({"input": "Explain lithography"}), 10)


@pytest.mark.parametrize(
    "classifications, expected, calls",
    [
        (["User Safety: unsafe"], "I cannot provide that information due to company policy.", 0),
        (
            ["User Safety: safe", "User Safety: safe\nResponse Safety: unsafe"],
            "I cannot provide that information due to company policy.",
            1,
        ),
        (
            ["User Safety: safe", "User Safety: safe\nResponse Safety: safe"],
            "Lithography uses light and photoresist.",
            1,
        ),
    ],
)
def test_chain_checks_input_and_output(monkeypatch, classifications, expected, calls):
    safety = FakeLLMModel(responses=classifications)
    monkeypatch.setenv("SAFETY_API_KEY", "test-key")
    monkeypatch.setattr(
        "nemoguardrails.rails.llm.llmrails.init_llm_model", lambda *args, **kwargs: safety
    )
    model_calls = []

    result = asyncio.run(invoke_chain(model_calls))
    output = result["output"] if isinstance(result, dict) else result
    assert output == expected
    assert len(model_calls) == calls
    assert safety.inference_count == len(classifications)


def test_safety_provider_error_does_not_call_chat_model(monkeypatch):
    safety = FakeLLMModel(responses=[])
    safety.llm_exception = TimeoutError("scripted safety provider failure")
    monkeypatch.setenv("SAFETY_API_KEY", "test-key")
    monkeypatch.setattr(
        "nemoguardrails.rails.llm.llmrails.init_llm_model", lambda *args, **kwargs: safety
    )
    model_calls = []

    with pytest.raises((ValueError, TimeoutError)):
        asyncio.run(invoke_chain(model_calls))
    assert not model_calls
