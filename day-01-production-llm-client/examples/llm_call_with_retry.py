import asyncio
import os
import random
from typing import Optional

from dotenv import load_dotenv
from openai import (
    APIError,
    APITimeoutError,
    AsyncOpenAI,
    AuthenticationError,
    RateLimitError,
)

# Load env vars
load_dotenv()


async def call_llm_with_retry(
    client: AsyncOpenAI,
    messages: list[dict],
    model: str = "gpt-4o-mini",
    max_retries: int = 3,
    temperature: float = 0.7,
) -> Optional[dict]:
    """
    Call LLM API with exponential backoff retry logic

    This handles transient failures and prevents
    overwhelming the API with rapid retries
    """

    for attempt in range(max_retries):
        try:
            response = await client.chat.completions.create(
                model=model, messages=messages, temperature=temperature, timeout=30.0
            )

            return {
                "success": True,
                "content": response.choices[0].message.content,
                "tokens": response.usage.total_tokens,
            }

        except RateLimitError as e:
            if attempt == max_retries - 1:
                return {"success": False, "error": "Rate limit exceeded"}

            # Exponential backoff with jitter
            wait_time = (2**attempt) + random.uniform(0, 1)
            print(f"Rate limited. Retrying in {wait_time:.2f}s...")
            await asyncio.sleep(wait_time)

        except APITimeoutError as e:
            if attempt == max_retries - 1:
                return {"success": False, "error": "Request timeout"}

            print(f"Timeout. Retrying... (Attempt {attempt + 1}/{max_retries})")

        except APIError as e:
            # Server error - don't retry immediately
            return {"success": False, "error": f"API error: {str(e)}"}

        except AuthenticationError as e:
            # Auth error - don't retry
            return {"success": False, "error": "Invalid API key"}

        except Exception as e:
            return {"success": False, "error": f"Unexpected error: {str(e)}"}

    return {"success": False, "error": "Max retries exceeded"}


# Test it
async def test_retry():
    client = AsyncOpenAI(
        base_url="https://openrouter.ai/api/v1", api_key=os.getenv("OPENROUTER_KEY")
    )

    result = await call_llm_with_retry(
        client=client,
        messages=[{"role": "user", "content": "Is Pokhara beautiful?"}],
    )

    if result["success"]:
        print(f"Response: {result['content']}")
        print(f"Tokens: {result['tokens']}")
    else:
        print(f"Error: {result['error']}")


asyncio.run(test_retry())
