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
