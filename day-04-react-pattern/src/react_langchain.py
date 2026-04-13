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
