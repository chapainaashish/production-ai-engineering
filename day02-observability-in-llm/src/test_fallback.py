import asyncio
import os

from dotenv import load_dotenv
from langsmith import traceable
from litellm import Router

load_dotenv()

model_list = [
    {
        "model_name": "gpt-4o-mini",
        "litellm_params": {
            "model": "gpt-4o-mini",
            "api_key": os.getenv("OPENROUTER_KEY"),
            "base_url": "https://openrouter.ai/api/v1",
            "rpm": 6,
        },
    },
    {
        "model_name": "claude-haiku-4.5",
        "litellm_params": {
            "model": "claude-haiku-4-5",
            "api_key": os.getenv("ANTHROPIC_API_KEY"),
            "rpm": 6,
        },
    },
]

router = Router(
    model_list=model_list,
    fallbacks=[{"gpt-4o-mini": ["claude-haiku-4.5"]}],
    retry_after=10,
    allowed_fails=3,
)


@traceable(run_type="llm")
async def call_model(messages: list[dict], model: str = "gpt-4o-mini"):
    response = await router.acompletion(
        model=model,
        messages=messages,
    )
    return response


async def main():
    messages = [{"role": "user", "content": "Write a short poem about Kathmandu."}]

    try:
        response = await call_model(messages=messages, model="gpt-4o-mini")
        print(response)
    except Exception as e:
        print("Error:", e)


asyncio.run(main())
