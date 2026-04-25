"""Smoke test: verify AI Builders gateway works via the project's LLM factory
with OpenAI-style function calling (required for TradingAgents)."""
from dotenv import load_dotenv
load_dotenv()

from langchain_core.tools import tool
from tradingagents.llm_clients.factory import create_llm_client


@tool
def get_weather(city: str) -> str:
    """Get current weather for a city."""
    return f"sunny, 20C in {city}"


def main():
    client = create_llm_client(
        provider="aibuilders",
        model="grok-4-fast",
        base_url="https://space.ai-builders.com/backend/v1",
    )
    llm = client.get_llm().bind_tools([get_weather])

    print("--- plain reply ---")
    r1 = llm.invoke("Say hi in 3 words.")
    print(repr(r1.content)[:200])

    print("--- tool calling ---")
    r2 = llm.invoke("What's the weather in Shanghai? Use the tool.")
    print("content:", repr(r2.content)[:200])
    print("tool_calls:", r2.tool_calls)

    assert r2.tool_calls, "LLM did not produce tool_calls — TradingAgents will not work!"
    print("\n[OK] Tool calling works. Ready to run TradingAgents.")


if __name__ == "__main__":
    main()
