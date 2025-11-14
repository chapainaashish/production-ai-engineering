import asyncio
import os

from dotenv import load_dotenv
from openai import APIConnectionError, APIError, AsyncOpenAI, RateLimitError, Timeout

load_dotenv()

client = AsyncOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_KEY"),
    max_retries=3,  # Automatic exponential backoff with jitter
    timeout=30.0,
)


async def simple_llm_call():
    """OpenAI API Call with Safe Error Handling"""
    try:
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Where is Nepal located?"}],
            temperature=0.7,
        )

        print("Response:", response.choices[0].message.content)
        print("\nToken Usage:")
        print(f"  Prompt: {response.usage.prompt_tokens}")
        print(f"  Completion: {response.usage.completion_tokens}")
        print(f"  Total: {response.usage.total_tokens}")
        print(f"  Finish Reason: {response.choices[0].finish_reason}")

    except RateLimitError as e:
        print("Rate limit exceeded after retries:", e)

    except Timeout as e:
        print("Request timed out after retries:", e)

    except APIConnectionError as e:
        print("Connection error after retries:", e)

    except APIError as e:
        print("API returned an error even after retries:", e)

    except Exception as e:
        print("Unexpected Error:", type(e).__name__, str(e))


asyncio.run(simple_llm_call())
