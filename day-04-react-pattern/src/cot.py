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
