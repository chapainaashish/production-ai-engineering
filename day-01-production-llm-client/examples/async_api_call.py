import asyncio
import os

from dotenv import load_dotenv
from openai import AsyncOpenAI

# Load env vars
load_dotenv()


async def simple_llm_call():
    """Simple API call"""

    # Using openrouter here as they provide free models (upto certain limits)
    client = AsyncOpenAI(
        base_url="https://openrouter.ai/api/v1", api_key=os.getenv("OPENROUTER_KEY")
    )

    response = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Where is Nepal located ?"}],
        temperature=0.7,
        timeout=30.0,
    )

    print("Response:", response.choices[0].message.content)
    print("\nToken Usage:")
    print(f"  Prompt: {response.usage.prompt_tokens}")
    print(f"  Completion: {response.usage.completion_tokens}")
    print(f"  Total: {response.usage.total_tokens}")


# Run the async function
asyncio.run(simple_llm_call())
