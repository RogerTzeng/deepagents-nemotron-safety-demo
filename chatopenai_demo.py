"""A plain LangChain ChatOpenAI chain protected by the shared custom-policy rails."""

import argparse
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from nemoguardrails import RailsConfig
from nemoguardrails.integrations.langchain.runnable_rails import RunnableRails

ROOT = Path(__file__).resolve().parent
GOOGLE_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"


def build_chain(model, config_path=ROOT / "config"):
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a semiconductor education assistant. Use public information only."),
            ("human", "{input}"),
        ]
    )
    return RunnableRails(
        RailsConfig.from_path(str(config_path)),
        runnable=prompt | model | StrOutputParser(),
        passthrough=True,
    )


def main():
    parser = argparse.ArgumentParser(
        description="LangChain ChatOpenAI + Nemotron custom-policy rails"
    )
    parser.add_argument("prompt", nargs="?", default="Explain photolithography in simple terms.")
    args = parser.parse_args()
    load_dotenv(ROOT / ".env")
    safety_key = os.getenv("SAFETY_API_KEY") or os.getenv("NVIDIA_API_KEY")
    # Keep the chat provider's settings separate from the Deep Agents demo.
    chat_key = (
        os.getenv("CHAT_API_KEY") or os.getenv("GOOGLE_API_KEY") or os.getenv("AGENT_API_KEY")
    )
    if not safety_key:
        parser.error("Set SAFETY_API_KEY (or NVIDIA_API_KEY) in .env")
    if not chat_key:
        parser.error(
            "Set CHAT_API_KEY, GOOGLE_API_KEY, or the existing Gemini AGENT_API_KEY in .env"
        )
    os.environ["SAFETY_API_KEY"] = safety_key
    model = ChatOpenAI(
        model=os.getenv("CHAT_MODEL", "gemini-3.5-flash-lite"),
        base_url=os.getenv("CHAT_BASE_URL", GOOGLE_BASE_URL),
        api_key=chat_key,
        timeout=60,
        max_retries=0,
    )
    result = build_chain(model).invoke({"input": args.prompt})
    print(result["output"] if isinstance(result, dict) else result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
