import json
import os
import re

import instructor
from dotenv import load_dotenv
from litellm import Router

load_dotenv()

MODEL_LIST = [
    {
        "model_name": "gpt-4o-mini",
        "litellm_params": {
            "model": "openai/gpt-4o-mini",
            "api_key": os.getenv("OPENROUTER_KEY"),
            "base_url": "https://openrouter.ai/api/v1",
            "rpm": 6,
        },
    },
]


# Tools
def get_stock_price(ticker: str) -> dict:
    prices = {
        "AAPL": {"price": 189.25, "change_pct": 1.2, "volume": "52M"},
        "GOOGL": {"price": 141.80, "change_pct": -0.8, "volume": "21M"},
        "MSFT": {"price": 378.50, "change_pct": 0.5, "volume": "18M"},
    }
    ticker = ticker.upper()
    if ticker not in prices:
        return {"error": f"Ticker '{ticker}' not found"}
    return {"ticker": ticker, **prices[ticker]}


def get_company_news(ticker: str) -> dict:
    news = {
        "AAPL": [
            "Vision Pro sales exceeded Q1 targets",
            "iPhone 16 supply chain ramp-up confirmed",
        ],
        "MSFT": [
            "Copilot reaches 1M enterprise users",
            "Azure OpenAI expands to 12 regions",
        ],
        "GOOGL": [
            "Gemini Ultra integrated into Workspace",
            "Antitrust ruling puts ads under scrutiny",
        ],
    }
    ticker = ticker.upper()
    if ticker not in news:
        return {"error": f"No news for '{ticker}'"}
    return {"ticker": ticker, "headlines": news[ticker]}


def compare_stocks(tickers: list[str]) -> dict:
    results = {
        t: get_stock_price(t) for t in tickers if "error" not in get_stock_price(t)
    }
    if not results:
        return {"error": "No valid tickers"}
    best = max(results, key=lambda t: results[t]["change_pct"])
    return {"comparison": results, "best_performer_today": best}


TOOLS = {
    "get_stock_price": get_stock_price,
    "get_company_news": get_company_news,
    "compare_stocks": compare_stocks,
}

SYSTEM_PROMPT = """You are a stock research assistant with access to:

- get_stock_price(ticker: str) → price, change_pct, volume
- get_company_news(ticker: str) → recent headlines
- compare_stocks(tickers: list[str]) → side-by-side comparison

Respond ONLY in this format:

Thought: <what you know and what you need>
Action: <tool_name>
Action Input: <valid JSON arguments>

When done:

Thought: I have all the information needed.
Final Answer: <your complete response>

Rules: always think before acting, use exact tool names, Action Input must be valid JSON.
"""


def parse_action(text: str) -> tuple[str | None, dict | None]:
    action_match = re.search(r"Action:\s*(\w+)", text)
    input_match = re.search(r"Action Input:\s*(\{.*?\}|\[.*?\])", text, re.DOTALL)
    if not action_match or not input_match:
        return None, None
    try:
        args = json.loads(input_match.group(1).strip())
    except json.JSONDecodeError:
        return action_match.group(1).strip(), None
    return action_match.group(1).strip(), args


def call_tool(tool_name: str, args: dict) -> str:
    if tool_name not in TOOLS:
        return f"Error: Unknown tool '{tool_name}'. Available: {list(TOOLS.keys())}"
    try:
        result = (
            TOOLS[tool_name](**args)
            if isinstance(args, dict)
            else TOOLS[tool_name](args)
        )
        return json.dumps(result, indent=2)
    except Exception as e:
        return f"Error calling {tool_name}: {str(e)}"


def react_agent(
    client: instructor.Instructor, question: str, max_iterations: int = 6
) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    print(f"\nQuestion: {question}\n{'='*60}")

    for i in range(max_iterations):
        # instructor is used here only for the routing/client; no response_model
        # since the ReAct format is free-text, not structured output
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            response_model=None,
            temperature=0,
            max_tokens=500,
            stop=["Observation:"],
        )
        text = response.choices[0].message.content.strip()
        print(f"\n[Step {i+1}]\n{text}")

        if "Final Answer:" in text:
            return text.split("Final Answer:")[-1].strip()

        tool_name, args = parse_action(text)
        if not tool_name:
            messages.append({"role": "assistant", "content": text})
            messages.append(
                {
                    "role": "user",
                    "content": "Follow the format: Thought / Action / Action Input.",
                }
            )
            continue

        obs = (
            call_tool(tool_name, args)
            if args is not None
            else f"Error: could not parse args for '{tool_name}'"
        )
        print(f"\nObservation: {obs}")

        messages.append({"role": "assistant", "content": text})
        messages.append({"role": "user", "content": f"Observation: {obs}"})

    return "Max iterations reached."


if __name__ == "__main__":
    router = Router(model_list=MODEL_LIST)
    client = instructor.from_litellm(router.completion)

    react_agent(
        client,
        "Which is performing better today Google or Microsoft? Give me a brief recommendation",
    )
