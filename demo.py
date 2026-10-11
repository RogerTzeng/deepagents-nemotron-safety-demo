import argparse
import os
from pathlib import Path

from deepagents import create_deep_agent
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from nemoguardrails.integrations.langchain.exceptions import GuardrailViolation
from nemoguardrails.integrations.langchain.middleware import GuardrailsMiddleware
from nemoguardrails.rails.llm.options import RailType

ROOT = Path(__file__).resolve().parent


class TextGuardrailsMiddleware(GuardrailsMiddleware):
    """Use official rail hooks with text-only current-turn classification messages."""

    def _convert_to_rails_messages(self, messages):
        user = next((m for m in reversed(messages) if isinstance(m, HumanMessage)), None)
        result = [{"role": "user", "content": user.text if user else ""}]
        if messages and isinstance(messages[-1], AIMessage):
            result.append({"role": "assistant", "content": messages[-1].text})
        return result


def public_reference(topic: str) -> str:
    """Look up public educational semiconductor information: lithography or interlocks."""
    return {
        "lithography": "Lithography transfers patterns to a wafer using light and photoresist.",
        "interlocks": "Interlocks stop equipment when unsafe conditions are detected. "
        "Follow approved maintenance procedures and report faults to authorized staff.",
    }.get(topic.lower(), "No public reference available for that topic.")


def build_agent(model, guardrails):
    return create_deep_agent(
        model=model,
        tools=[public_reference],
        middleware=[guardrails],
        subagents=[
            {
                "name": "general-purpose",
                "description": "Help with public semiconductor education.",
                "system_prompt": "Use only public information and follow the company policy.",
                "tools": [public_reference],
                "middleware": [guardrails],
            }
        ],
        system_prompt="You are a semiconductor education assistant. Use public_reference "
        "for lithography and interlocks. Provide public educational information only. "
        "Never invent proprietary data or credentials.",
    )


def main():
    parser = argparse.ArgumentParser(
        description="Deep Agents + official NeMo Guardrails middleware"
    )
    parser.add_argument("prompt", nargs="?", default="Use public_reference to explain lithography.")
    parser.add_argument("--config", type=Path, default=ROOT / "config")
    parser.add_argument("--check", action="store_true", help="Run input rails without the agent")
    parser.add_argument("--response", help="With --check, run output rails on this response")
    args = parser.parse_args()
    if args.response is not None and not args.check:
        parser.error("--response requires --check")
    load_dotenv(ROOT / ".env")
    safety_key = os.getenv("SAFETY_API_KEY") or os.getenv("NVIDIA_API_KEY")
    agent_key = os.getenv("AGENT_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not safety_key:
        parser.error("Set SAFETY_API_KEY (or NVIDIA_API_KEY) in .env")
    if not args.check and not agent_key:
        parser.error("Set AGENT_API_KEY (or GOOGLE_API_KEY) in .env")
    os.environ["SAFETY_API_KEY"] = safety_key
    if agent_key:
        os.environ["AGENT_API_KEY"] = agent_key
    guardrails = TextGuardrailsMiddleware(config_path=str(args.config), raise_on_violation=True)
    try:
        if args.check:
            messages = [{"role": "user", "content": args.prompt}]
            rail_types = [RailType.INPUT]
            if args.response is not None:
                messages.append({"role": "assistant", "content": args.response})
                rail_types = [RailType.OUTPUT]
            result = guardrails.rails.check(messages, rail_types=rail_types)
            print(f"status={result.status.value}; rail={result.rail or 'none'}")
            return 2 if result.status.value == "blocked" else 0
        model_name = os.getenv("AGENT_MODEL", "gemini-3.5-flash-lite")
        if model_name.startswith("gemini-"):
            model = ChatGoogleGenerativeAI(
                model=model_name, api_key=agent_key, timeout=60, max_retries=0
            )
        else:
            model = ChatOpenAI(
                model=model_name,
                api_key=agent_key,
                base_url=os.getenv("AGENT_BASE_URL"),
                timeout=60,
                max_retries=0,
            )
        result = build_agent(model, guardrails).invoke(
            {"messages": [HumanMessage(content=args.prompt)]}, {"recursion_limit": 30}
        )
        print(result["messages"][-1].text)
    except GuardrailViolation as exc:
        print(
            f"Blocked at {exc.rail_type}; rail={exc.result.rail if exc.result else 'execution error'}"
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
