# Reasoning with LLM

A direct prompt works well with LLM for a simple task, but when a task needs multiple steps to arrive at a particular solution, a direct prompt is likely to fail because of LLM hallucination and assumptions. To reduce this failure, we need to force the model to *reason before it acts*. The concepts of Chain-of-Thought and ReAct try to solve this kind of problem differently. But let's be clear on when to use each.

| Pattern | Reasoning | External Data | Best For |
|---|---|---|---|
| Direct prompt | ❌ | ❌ | Simple Q&A |
| Chain-of-Thought | ✅ | ❌ | Classification, triage, scoring |
| ReAct | ✅ | ✅ | Research, workflows, agents |


## Part 1: Chain-of-Thought - When Your Prompt Has Everything It Needs

Chain-of-Thought forces the model to reason step-by-step before arriving at the solution. It doesn't call any new tools or external data, but forces the LLM to think logically before giving the final output. Let's see this with a simple example of support ticket triage, which categorizes and prioritizes tickets based on urgency.

```python
import os

import instructor
from dotenv import load_dotenv
from litellm import Router
from pydantic import BaseModel

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


class SupportTriage(BaseModel):
    reasoning: str
    priority: str  # critical / high / medium / low
    category: str  # billing / technical / account / general


TRIAGE_PROMPT = """You are a senior support engineer triaging incoming tickets.
Think through each ticket before classifying it:
- What is the user actually experiencing?
- What is the business impact if unresolved?
- Which team has the expertise to resolve this?
- How urgently does this need attention?
Reason through these questions explicitly before producing your output.
"""


def triage_ticket(client: instructor.Instructor, ticket: str) -> SupportTriage:
    return client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": TRIAGE_PROMPT},
            {"role": "user", "content": f"Ticket:\n{ticket}"},
        ],
        response_model=SupportTriage,
        temperature=0,
        max_retries=3,
    )


if __name__ == "__main__":
    router = Router(model_list=MODEL_LIST)
    client = instructor.from_litellm(router.completion)

    tickets = [
        "Our entire team is locked out since the deployment 20 minutes ago. Client demo in 2 hours.",
        "I was charged twice for my subscription this month. Please refund the duplicate.",
    ]

    for ticket in tickets:
        result = triage_ticket(client, ticket)
        print(f"Ticket   : {ticket[:60]}...")
        print(f"Reasoning: {result.reasoning}")
        print(f"Priority : {result.priority}")
        print(f"Category : {result.category}")
        print()
```

Here, the `reasoning` field is the audit trail. So when a ticket gets mis-triaged, you can see exactly where the model's logic broke down and fix the prompt.

CoT is enough when the model already has all the facts, but the moment it needs to *fetch* something — like prices, documents, or database records — we need something called ReAct.

## Part 2: ReAct - When the Model Needs to Get Data

ReAct stands for **Reason + Act**. Here, the model has the ability to call an external service or tool to get the final output. So the model enters a loop instead of producing a single-shot output.

The workflow of ReAct looks like this:
```
Thought → Action → Observation → Thought → Action → ... → Final Answer
```

In each cycle, the model thinks about what it needs, calls a tool, reads the result, and decides what to do next.

Let's understand this concept with an example that analyses and compares different stock prices and news, then gives a final recommendation:

```python
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

```

Here, `get_stock_price`, `get_company_news`, and `compare_stocks` are the tools that can be called by the LLM to get the relevant information. Also, observations go in as user messages, not as assistant messages.

`stop=["Observation:"]` tells the LLM to wait for the tool output rather than making assumptions.

`max_iterations=6` tells the LLM to iterate the cycle only up to 6 times to reduce cost by avoiding infinite loops.


## Part 3: Building a ReAct Agent with LangChain

As you continue to build the ReAct loop from scratch, you might spend most of your time debugging it rather than formulating the logic. So in production, we use frameworks to facilitate the ReAct agent, such as LangChain or CrewAI. These frameworks give you the freedom to spend time implementing logic rather than building infrastructure like token management, debugging, and error recovery.

Here is the same example using LangChain to build the ReAct agent rather than building it from scratch. The code below uses `create_react_agent` from LangGraph. It is the current recommended approach.

```python
import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

load_dotenv()


@tool
def get_stock_price(ticker: str) -> dict:
    """Get the current price and daily change for a stock ticker"""
    prices = {
        "AAPL": {"price": 189.25, "change_pct": 1.2, "volume": "52M"},
        "GOOGL": {"price": 141.80, "change_pct": -0.8, "volume": "21M"},
        "MSFT": {"price": 378.50, "change_pct": 0.5, "volume": "18M"},
    }
    ticker = ticker.upper()
    if ticker not in prices:
        return {
            "error": f"Ticker '{ticker}' not found. Supported: {list(prices.keys())}"
        }
    return {"ticker": ticker, **prices[ticker]}


@tool
def get_company_news(ticker: str) -> dict:
    """Get the latest news headlines for a company by stock ticker."""
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


@tool
def compare_stocks(tickers: list[str]) -> dict:
    """Compare daily performance across multiple stock tickers."""
    prices = {
        "AAPL": {"price": 189.25, "change_pct": 1.2},
        "GOOGL": {"price": 141.80, "change_pct": -0.8},
        "MSFT": {"price": 378.50, "change_pct": 0.5},
    }
    results = {t.upper(): prices[t.upper()] for t in tickers if t.upper() in prices}
    if not results:
        return {"error": "No valid tickers provided"}
    best = max(results, key=lambda t: results[t]["change_pct"])
    return {"comparison": results, "best_performer_today": best}


llm = ChatOpenAI(
    model="openai/gpt-4o-mini",
    temperature=0,
    openai_api_key=os.getenv("OPENROUTER_KEY"),
    openai_api_base="https://openrouter.ai/api/v1",
)

tools = [get_stock_price, get_company_news, compare_stocks]
agent = create_agent(llm, tools)


def run_agent_streamed(question: str) -> None:
    print(f"\nQuestion: {question}\n{'='*60}")
    for chunk in agent.stream({"messages": [HumanMessage(content=question)]}):
        if "model" in chunk:
            for msg in chunk["model"]["messages"]:
                if msg.tool_calls:
                    for tc in msg.tool_calls:
                        print(f"  → Calling: {tc['name']}({tc['args']})")
                elif msg.content:
                    print(f"\nFinal Answer:\n{msg.content}")
        elif "tools" in chunk:
            for msg in chunk["tools"]["messages"]:
                print(f"  ← Result  : {msg.content[:120]}...")


if __name__ == "__main__":
    run_agent_streamed("Which is performing better today Google or Microsoft?")
    run_agent_streamed("What's the latest on Apple and is the stock up or down?")
```

As you can see from the code above, tool schemas are auto-generated from type hints and docstrings. The model gets a precise description of every tool, which reduces wrong-tool calls in production. Using `agent.stream()` pushes each step of analysis in real time, which is useful for presenting it in the UI. Using these types of frameworks is great in production. However, they also add abstraction layers and occasionally obscure costs, and when something breaks, you're debugging through them. So use them wisely and understand the trade-offs before jumping to code.