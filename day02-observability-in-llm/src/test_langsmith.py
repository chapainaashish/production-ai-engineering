import asyncio
import os

from dotenv import load_dotenv
from langsmith import traceable
from openai import APIConnectionError, APIError, AsyncOpenAI, RateLimitError, Timeout

load_dotenv()

# Initialize OpenRouter client
client = AsyncOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_KEY"),
    max_retries=3,  # Automatic exponential backoff
    timeout=30.0,
)


@traceable(run_type="llm")
async def call_llm(
    messages: list[dict], model: str = "gpt-4o-mini", temperature: float = 0.7
):
    """
    Call LLM with messages and automatically log to LangSmith.
    """
    try:
        response = await client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
        )
        return response
    except (APIConnectionError, APIError, RateLimitError, Timeout) as e:
        print(f"LLM API error: {e}")
        return None


async def main():
    messages = [{"role": "user", "content": "Where is Nepal located?"}]

    response = await call_llm(messages)

    if response:
        content = response.choices[0].message.content
        print("Response:", content)
    else:
        print("Failed to get response from LLM.")


if __name__ == "__main__":
    asyncio.run(main())
