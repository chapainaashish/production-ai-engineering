import asyncio
import logging
import os
import random
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, TypedDict

from dotenv import load_dotenv
from openai import (
    APIError,
    APITimeoutError,
    AsyncOpenAI,
    AuthenticationError,
    RateLimitError,
)

load_dotenv()


class APIResult(TypedDict):
    success: bool
    content: Optional[str]
    tokens: Optional[int]
    cost: Optional[float]
    error: Optional[str]


class StatsDict(TypedDict):
    total_calls: int
    total_tokens: int
    total_cost: float
    avg_tokens_per_call: float
    avg_cost_per_call: float


@dataclass
class LLMResponse:
    """Structured LLM response with metadata"""

    content: str
    model: str
    tokens: int
    cost: float
    latency_ms: int
    timestamp: str


def calculate_cost(
    input_tokens: int, output_tokens: int, model: str = "gpt-5-nano"
) -> float:
    """Calculate the cost of an API call"""
    pricing = {
        "gpt-5-nano": {"input": 0.05, "output": 0.40},
        "gpt-5": {"input": 1.25, "output": 10.00},
        "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    }

    if model not in pricing:
        model = "gpt-5-nano"

    price = pricing[model]
    input_cost = (input_tokens / 1_000_000) * price["input"]
    output_cost = (output_tokens / 1_000_000) * price["output"]

    return input_cost + output_cost


async def call_llm_with_retry(
    client: AsyncOpenAI,
    messages: list[dict[str, str]],
    model: str = "gpt-4o-mini",
    max_retries: int = 3,
    temperature: float = 0.7,
) -> APIResult:
    """
    Call LLM API with exponential backoff retry logic
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
                "cost": calculate_cost(
                    response.usage.prompt_tokens,
                    response.usage.completion_tokens,
                    model,
                ),
                "error": None,
            }

        except RateLimitError:
            if attempt == max_retries - 1:
                return {
                    "success": False,
                    "content": None,
                    "tokens": None,
                    "cost": None,
                    "error": "Rate limit exceeded",
                }

            wait_time = (2**attempt) + random.uniform(0, 1)
            await asyncio.sleep(wait_time)

        except APITimeoutError:
            if attempt == max_retries - 1:
                return {
                    "success": False,
                    "content": None,
                    "tokens": None,
                    "cost": None,
                    "error": "Request timeout",
                }

        except AuthenticationError:
            return {
                "success": False,
                "content": None,
                "tokens": None,
                "cost": None,
                "error": "Invalid API key",
            }

        except APIError as e:
            return {
                "success": False,
                "content": None,
                "tokens": None,
                "cost": None,
                "error": f"API error: {str(e)}",
            }

        except Exception as e:
            return {
                "success": False,
                "content": None,
                "tokens": None,
                "cost": None,
                "error": f"Unexpected error: {str(e)}",
            }

    return {
        "success": False,
        "content": None,
        "tokens": None,
        "cost": None,
        "error": "Max retries exceeded",
    }


class ProductionLLMClient:
    """
    Production-ready LLM API client with logging, retries and cost tracking
    """

    def __init__(
        self,
        api_key: str,
        default_model: str = "gpt-4o-mini",
        log_file: str = "llm_calls.log",
        base_url: Optional[str] = None,
    ):
        client_kwargs = {"api_key": api_key}
        client_kwargs["base_url"] = base_url if base_url else None

        self.client = AsyncOpenAI(**client_kwargs)
        self.default_model = default_model

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
        )
        self.logger = logging.getLogger(__name__)

        self.total_cost = 0.0
        self.total_tokens = 0
        self.call_count = 0

        self.logger.info(f"LLM Client initialized with model: {default_model}")

    async def call(
        self,
        prompt: str,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_retries: int = 3,
    ) -> Optional[LLMResponse]:
        """
        Make an LLM API call with full production features
        """

        model = model or self.default_model
        messages = [{"role": "user", "content": prompt}]

        start_time = time.time()

        result = await call_llm_with_retry(
            client=self.client,
            messages=messages,
            model=model,
            max_retries=max_retries,
            temperature=temperature,
        )

        latency = int((time.time() - start_time) * 1000)

        if not result["success"]:
            self.logger.error(f"LLM call failed: {result['error']}")
            return None

        self.call_count += 1
        self.total_tokens += result["tokens"]
        self.total_cost += result["cost"]

        response = LLMResponse(
            content=result["content"],
            model=model,
            tokens=result["tokens"],
            cost=result["cost"],
            latency_ms=latency,
            timestamp=datetime.now().isoformat(),
        )

        self.logger.info(
            f"LLM Call #{self.call_count} | "
            f"Model: {model} | "
            f"Tokens: {result['tokens']} | "
            f"Cost: ${result['cost']:.6f} | "
            f"Latency: {latency}ms"
        )

        return response

    def get_stats(self) -> StatsDict:
        """Get usage statistics for this session"""
        return {
            "total_calls": self.call_count,
            "total_tokens": self.total_tokens,
            "total_cost": round(self.total_cost, 6),
            "avg_tokens_per_call": round(
                self.total_tokens / max(self.call_count, 1), 2
            ),
            "avg_cost_per_call": round(self.total_cost / max(self.call_count, 1), 6),
        }

    def reset_stats(self) -> None:
        """Reset usage statistics"""
        self.logger.info("Resetting session statistics")
        self.total_cost = 0.0
        self.total_tokens = 0
        self.call_count = 0


async def main() -> None:
    client = ProductionLLMClient(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_KEY") or "",
        default_model="gpt-4o-mini",
    )

    response1 = await client.call("What is the capital city of Nepal?")
    print(response1)

    stats = client.get_stats()
    print(f"\nSession Stats:")
    print(f"  Total Calls: {stats['total_calls']}")
    print(f"  Total Tokens: {stats['total_tokens']}")
    print(f"  Total Cost: ${stats['total_cost']}")
    print(f"  Avg Cost/Call: ${stats['avg_cost_per_call']}")


if __name__ == "__main__":
    asyncio.run(main())
