import asyncio
import os

from dotenv import load_dotenv
from openai import AsyncOpenAI
from traceloop.sdk import Traceloop
from traceloop.sdk.decorators import workflow

load_dotenv()

Traceloop.init(app_name="production_api", api_key=os.getenv("TRACELOOP_APIKEY"))

client = AsyncOpenAI(
    api_key=os.getenv("OPENROUTER_KEY"),
    base_url="https://openrouter.ai/api/v1",
)


@workflow(name="llm_completion")
async def call_llm(messages: list[dict]):
    response = await client.chat.completions.create(
        model="gpt-4o-mini", messages=messages
    )
    return response


async def main():
    messages = [{"role": "user", "content": "Explain Nepal in 50 words"}]

    try:
        response = await call_llm(messages)
        print("Response:", response.choices[0].message.content)
    except Exception as e:
        print("Error:", e)


asyncio.run(main())
