"""Stream a Deep Agent through NeMo's native buffered output rails."""

import argparse
import asyncio
import json
import os
from contextlib import aclosing

from dotenv import load_dotenv
from langchain_core.messages import AIMessageChunk, HumanMessage
from langchain_openai import ChatOpenAI
from nemoguardrails import LLMRails, RailsConfig

from demo import ROOT
from execution_rails_demo import ToolResultRejected, build_execution_agent


class StreamRejected(RuntimeError):
    pass


def build_streaming_rails():
    # Reuse the Nemotron model, custom policy initializer, and output Colang flow.
    config = RailsConfig.from_path(str(ROOT / "config"))
    config.rails.input.flows = []
    config.streaming = True
    config.rails.output.streaming.enabled = True
    config.rails.output.streaming.chunk_size = 16
    config.rails.output.streaming.context_size = 4
    config.rails.output.streaming.stream_first = False
    return LLMRails(config)


async def agent_text(agent, prompt):
    async with aclosing(
        agent.astream(
            {"messages": [HumanMessage(content=prompt)]},
            config={"recursion_limit": 10},
            stream_mode="messages",
        )
    ) as stream:
        async for message, metadata in stream:
            if (
                metadata.get("langgraph_node") == "model"
                and isinstance(message, AIMessageChunk)
                and message.text
            ):
                yield message.text


async def guarded_stream(agent, rails, prompt):
    async with (
        aclosing(agent_text(agent, prompt)) as source,
        aclosing(
            rails.stream_async(generator=source, messages=[{"role": "user", "content": prompt}])
        ) as stream,
    ):
        async for chunk in stream:
            # NeMo emits a JSON error item for a blocked chunk or rail failure.
            try:
                payload = json.loads(chunk)
            except (ValueError, TypeError):
                payload = None
            if isinstance(payload, dict) and isinstance(payload.get("error"), dict):
                raise StreamRejected(payload["error"].get("code", "stream_error"))
            yield chunk


async def run(model, rails, prompt):
    agent, _ = build_execution_agent(model)
    try:
        async for text in guarded_stream(agent, rails, prompt):
            print(text, end="", flush=True)
    except (StreamRejected, ToolResultRejected) as exc:
        print(f"\n[Stopped: {exc}]")
        return 2
    print()
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prompt", nargs="?", default="Fetch and summarize scenario: public")
    args = parser.parse_args()
    load_dotenv(ROOT / ".env")
    safety_key = os.getenv("SAFETY_API_KEY") or os.getenv("NVIDIA_API_KEY")
    agent_key = os.getenv("STREAM_API_KEY") or os.getenv("NVIDIA_API_KEY")
    if not safety_key or not agent_key:
        parser.error("Set SAFETY_API_KEY and STREAM_API_KEY (or NVIDIA_API_KEY) in .env")
    os.environ["SAFETY_API_KEY"] = safety_key
    model = ChatOpenAI(
        model=os.getenv("STREAM_MODEL", "z-ai/glm-5.3"),
        base_url=os.getenv("STREAM_BASE_URL", "https://integrate.api.nvidia.com/v1"),
        api_key=agent_key,
        timeout=300,
        max_retries=0,
    )
    return asyncio.run(run(model, build_streaming_rails(), args.prompt))


if __name__ == "__main__":
    raise SystemExit(main())
