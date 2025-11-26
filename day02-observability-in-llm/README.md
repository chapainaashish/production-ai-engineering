You’ve built your first AI application, and now it’s time to move it to production. But how do you track costs, latency, and errors across thousands of requests? This is where **observability** comes in. it helps you see what’s happening in your system and saves you hours of debugging. Learn more about observability [here](https://opentelemetry.io/docs/concepts/observability-primer)

In this post, I’ll show a production observability stack that you can use. LangSmithfor for detailed tracing, OpenLLMetry for vendor-neutral observability. We will go even further and learn about LiteLLM for multi-provider routing with budget controls. These tools give you the insights you need without building custom logging from scratch.

Now, let’s look at the problems and how these tools solve them.

## Problem 1: You Can't See What's Happening

When your LLM API fails, you need answers like:

* Which model was being called?
    
* How long did the request take?
    
* What was the specific error message?
    

Without observability, you basically need to guess the problem. So, you need to add observability tools like LangSmith or OpenLLMetry.

## Solution 1: Observability with LangSmith

LangSmith is a managed SaaS platform from LangChain that gives you complete observability. It's very simple to get started like shown in the code below:

```python
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
```

After you run this, head over to [smith.langchain.com](https://smith.langchain.com) and you'll see:

* The complete trace with full request and response
    
* Token counts broken down by input and output
    
* Latency measured in milliseconds
    
* Cost calculated automatically for you
    

That `@traceable` decorator basically does all this under the hood. The tradeoff of using LangSmith in the long run on big projects is you will be vendor-locked with this platform.

## Solution 2: Vendor-Neutral Observability with OpenLLMetry

If you already have observability infrastructure (Datadog, Grafana, Honeycomb, etc.) or need complete control over your data, OpenLLMetry is your answer. It's an open-source SDK built on OpenTelemetry that sends traces to any backend you want, and like LangSmith, you are not left with only one choice.

```python
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
```

For now, I am sending the data to traceloop but you can customize it to send to your any observability back-end.

## Problem 2: You Need Cost Visibility and Control

Alright, most developers stop at the observability part, but in production you need more than that. Sure, observability tells you what is happening inside your system like latency and errors, but it doesn't stop things from going wrong. So, you need some kind of way to act on that data. One tool you can use to complement observability is LiteLLM.

Observability reveals cost spikes and provider failures, and LiteLLM prevents them and routes around them. They give you both visibility and control, which is what a real production-ready LLM stack needs.

## Solution 1: LiteLLM for Cost Control

LiteLLM lets you control the budget and limits in production. This is very useful when you have a small budget and multiple operations. Here is the sample code that you can use for budget controlling using LiteLLM:

```python
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
]

router = Router(
    model_list=model_list,
    fallbacks=[],
)

customer_budgets = {
    "customer_123": {"limit": 50.00, "spent": 0.00},
    "customer_456": {"limit": 100.00, "spent": 0.00},
}


@traceable(run_type="llm")
async def call_with_budget(
    customer_id: str, messages: list[dict], model: str = "gpt-4o-mini"
):
    budget = customer_budgets.get(customer_id)

    if budget["spent"] >= budget["limit"]:
        raise Exception(
            f"Budget exceeded for {customer_id}: "
            f"${budget['spent']:.2f} / ${budget['limit']:.2f}"
        )

    response = await router.acompletion(
        model=model,
        messages=messages,
    )

    cost = response._hidden_params.get("response_cost", 0)
    customer_budgets[customer_id]["spent"] += cost

    return response


async def main():
    messages = [{"role": "user", "content": "Write a short poem about Kathmandu."}]
    customer_id = "customer_123"

    try:
        response = await call_with_budget(
            customer_id=customer_id, messages=messages, model="gpt-4o-mini"
        )
        print(response)
    except Exception as e:
        print("Error:", e)


asyncio.run(main())
```

In production, you should store these budgets in some kind of database instead of a dict. Reset them monthly or per billing cycle and set relevant alerts.

## Solution 2: High Availability Through Multi-Provider Fallbacks

Since we already implemented LiteLLM, we can leverage it further to reduce our application downtime. So, if your application depends on one provider only, your entire application goes down with them. To solve this, we can use fallback providers without any significant code changes.

See the below example, when OpenAI goes down, LiteLLM will automatically route to Anthropic:

```python
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
```

## Resources

* [https://docs.smith.langchain.com](https://docs.smith.langchain.com)
    
* [https://smith.langchain.com/pricing](https://smith.langchain.com/pricing)
    
* [https://www.traceloop.com/docs/openllmetry](https://www.traceloop.com/docs/openllmetry)
    
* [https://github.com/traceloop/openllmetry](https://github.com/traceloop/openllmetry)
    
* [https://www.traceloop.com/docs/openllmetry/integrations](https://www.traceloop.com/docs/openllmetry/integrations)
    
* [https://github.com/BerriAI/litellm](https://github.com/BerriAI/litellm)
    
* [https://docs.litellm.ai](https://docs.litellm.ai)
    
* [https://openai.com/api/pricing](https://openai.com/api/pricing)
    
* [https://anthropic.com/pricing](https://anthropic.com/pricing)
    
* [https://opentelemetry.io/docs/concepts/observability-primer/](https://opentelemetry.io/docs/concepts/observability-primer/)