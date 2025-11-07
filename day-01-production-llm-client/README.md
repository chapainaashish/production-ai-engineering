## Overview

The first way to interact with any LLM is through the API in development. This is pretty basic which you will master easily but you can miss some critical aspect which I will cover in this post.

**Topic Covered:**

* LLM API client with error handling
    
* Token counter to track tokens
    
* Cost calculator to monitor spending
    
* Exponential backoff retry logic to handle API failures
    
* Logging system for debugging and monitoring
    

---

## Part 1: Understand Tokens Before Getting Excited

Before calling the LLM, tokens are important to understand. So, when you send "Hello, world!" to GPT, the model doesn't process it as two words. It breaks it down into smaller units called tokens. Tokens directly impact everything in LLM development.

OpenAI uses a tokenization algorithm called **Byte Pair Encoding (BPE)**, implemented in their `tiktoken` library. BPE algorithm doesn't split text into tokens using spaces or punctuation, It's more complex. You can read more about it [here](https://www.kaggle.com/code/qmarva/1-bpe-tokenization-algorithm-eng)

**Some approximations:**

* 1 token ≈ 4 characters in English
    
* Non-English text uses significantly more tokens (sometimes 2-3x more)
    

### Tokens are Currency

**1\. Pricing is Per-Token**

Every API call costs money based on token count. If you miscalculate tokens by 50%, you're miscalculating your budget by 50%. At scale, this means thousands of dollars in unexpected costs.

**2\. Context Windows Are Token-Limited**

When you see "GPT-4 has an 8K context window," that means 8,000 tokens total for your input AND the response. If you run out of tokens mid-conversation, the API fails. Your application breaks and god knows what will happen next.

**3\. Rate Limits Are Token-Based**

OpenAI limits you by tokens per minute (TPM), more than requests per minute. A single large request with 10K tokens counts the same as 10 small requests with 1K tokens each.

Okay enough theory, Let's write code to see how tokenization works:

```python
import tiktoken


def count_tokens(text: str, model: str = "gpt-4") -> int:
    """Count tokens in text for a specific model"""
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))


# Test different inputs
examples = ["Hello", "Hello, world!", "Artificial Intelligence", "AI"]

for text in examples:
    tokens = count_tokens(text)
    print(f"'{text}' = {tokens} tokens")
```

**Output:**

```plaintext
'Hello' = 1 tokens
'Hello, world!' = 4 tokens
'Artificial Intelligence' = 3 tokens
'AI' = 1 tokens
```

Notice that "Hello, world!" is 4 tokens, not 2. The comma and space are separate tokens. This is why you can't estimate tokens by counting words.

---

## Part 2: Calling the LLM API Right Way

### Hide Secrets in .env

Before we call the OpenAI API, we need to set up our API key properly. Never hardcode API keys in your code. Sounds pretty basic but most of the people miss it. This is how keys get leaked to GitHub and you end up with unexpected bills.

Create a `.env` file in your project root:

```bash
OPENAI_API_KEY=sk-your-actual-api-key-here
```

Now add this file to `.gitignore` so it never gets committed:

```bash
echo ".env" >> .gitignore
```

Look, it is that simple, ALWAYS DO IT….

### Get Advantage of Async Call

OpenAI can be called asynchronously. You can send multiple requests in parallel and get advantage of async call.

```python
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
```

When you run this, you'll get a response from GPT along with token usage statistics.

```plaintext
Response: Nepal is a landlocked country located in South Asia, situated mainly in the Himalayas. It is bordered by China to the north and India to the south, east, and west. Nepal is known for its diverse geography, which includes plains, hills, and the towering peaks of the Himalayas, including Mount Everest, the highest point on Earth. The capital city of Nepal is Kathmandu.

Token Usage:
  Prompt: 12
  Completion: 79
  Total: 91
```

### Understanding the Response

If you view the API response on details, it returns a structured object with the following key fields:

* `choices[0].message.content`: The actual text response from the model
    
* `usage.prompt_tokens`: How many tokens your input used
    
* `usage.completion_tokens`: How many tokens the response used
    
* `usage.total_tokens`: Sum of both (this is what you pay for)
    
* `finish_reason`:
    
    * `"stop"`: Model completed the response naturally
        
    * `"length"`: Hit the token limit (response was cut off)
        
    * `"content_filter"`: Response was blocked by safety filters
        

That `finish_reason` field matters in production. if you see `"length"`, the response was truncated and your application needs to handle that.

---

## Part 3: Calculating The Cost

LLM API calls can get expensive fast. Before deploying your app into production, understand token-based pricing first.

### Current Pricing

As of November 2025, OpenAI charges:

| Model | Input (per 1M tokens) | Output (per 1M tokens) |
| --- | --- | --- |
| gpt-5-nano | $0.05 | $0.40 |
| gpt-5 | $1.25 | $10.00 |

**What you can understand from this table:**

* Output tokens cost more than input tokens (8x more for both models)
    
* GPT-5 is 25x more expensive than GPT-5 nano for both input and output
    

Alright, Now you have seen the cost, the model choice depends on your usecase. For straightforward tasks like summarization and classification, you can use GPT nano version and for complex task, you can use GPT 5 version.

### Building a Cost Calculator

```python
def calculate_cost(
    input_tokens: int, output_tokens: int, model: str = "gpt-5-nano"
) -> dict:
    """Calculate the cost of an API call"""

    pricing = {
        "gpt-5-nano": {"input": 0.05, "output": 0.40},
        "gpt-5": {"input": 1.25, "output": 10.00},
    }

    if model not in pricing:
        model = "gpt-5-nano"

    price = pricing[model]
    input_cost = (input_tokens / 1_000_000) * price["input"]
    output_cost = (output_tokens / 1_000_000) * price["output"]

    return {
        "input_cost": input_cost,
        "output_cost": output_cost,
        "total_cost": input_cost + output_cost,
        "model": model,
    }


# Example usage
cost = calculate_cost(100, 200, "gpt-5-nano")
print(f"Cost: ${cost['total_cost']:.6f}")

# Compare with GPT-5
cost_gpt5 = calculate_cost(100, 200, "gpt-5")
print(f"GPT-5 would cost: ${cost_gpt5['total_cost']:.6f}")
```

**Output:**

```plaintext
Cost: $0.000085
GPT-5 would cost: $0.002125
```

### Real-World Cost Projection

Let's calculate what a real production system would cost. Imagine you're building a customer support chatbot:

**Scenario:**

* 10,000 requests per day
    
* Average 300 input tokens and 200 output tokens per request
    

```python
daily_requests = 10_000
input_tokens_per_request = 300
output_tokens_per_request = 200

# Calculate monthly costs
monthly_requests = daily_requests * 30
monthly_input_tokens = monthly_requests * input_tokens_per_request
monthly_output_tokens = monthly_requests * output_tokens_per_request

# GPT-5 nano costs
nano_input_cost = (monthly_input_tokens / 1_000_000) * 0.05
nano_output_cost = (monthly_output_tokens / 1_000_000) * 0.40
nano_total = nano_input_cost + nano_output_cost

# GPT-5 costs
gpt5_input_cost = (monthly_input_tokens / 1_000_000) * 1.25
gpt5_output_cost = (monthly_output_tokens / 1_000_000) * 10.00
gpt5_total = gpt5_input_cost + gpt5_output_cost

print(f"GPT-5 Nano Monthly: ${nano_total:,.2f}")
print(f"GPT-5 Monthly: ${gpt5_total:,.2f}")
print(f"Savings with GPT-5 Nano: ${gpt5_total - nano_total:,.2f}/month")
```

**Output:**

```plaintext
GPT-5 Nano Monthly: $28.50
GPT-5 Monthly: $712.50
Savings with GPT-5 Nano: $684.00/month
```

---

## Part 4: Error Handling & Retries are Important

LLM APIs might fail. So, you shouldn't assume they are immune to errors. You should treat them like any other third party APIs or even give more importance if you are building the system around it.

### Some Common API Errors

1\. Rate Limit Errors (HTTP 429)

2\. Timeout Errors

3\. API Errors (HTTP 500-599)

4\. Authentication Errors (HTTP 401)

5\. Invalid Request Errors (HTTP 400)

### Implementing Exponential Backoff

The standard approach to handling retries is **exponential backoff with jitter**

```python
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
```

### Why Add Jitter?

You might wonder about this line:

```python
wait_time = (2 ** attempt) + random.uniform(0, 1)
```

Why not just use `2 ** attempt`?

**Without jitter:**

* Retry after 1s, 2s, 4s, 8s...
    
* If 100 requests hit rate limits simultaneously, they all retry at the same time
    
* This creates a **thundering herd** which means all requests slam the API at once
    
* The API gets overwhelmed again, causing another round of failures
    

**With jitter:**

* Retry after 1.0-2.0s, 2.0-3.0s, 4.0-5.0s...
    
* Requests spread out over time and API load distributes evenly
    

For more details on this pattern, see [AWS's exponential backoff and jitter article](https://aws.amazon.com/blogs/architecture/exponential-backoff-and-jitter/).

---

## Part 5: Production-Ready Client

Alright, let’s combine everything and build a production-ready LLM client. It handles:

* Automatic retries with exponential backoff
    
* Cost and token tracking across all calls
    
* Comprehensive logging for debugging
    
* Graceful error handling
    
* Session statistics
    

### Complete Implementation

```python
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
```

**Output:**

```plaintext
2025-11-07 10:48:13,828 - INFO - LLM Client initialized with model: gpt-4o-mini
2025-11-07 10:48:17,194 - INFO - HTTP Request: POST https://openrouter.ai/api/v1/chat/completions "HTTP/1.1 200 OK"
2025-11-07 10:48:17,242 - INFO - LLM Call #1 | Model: gpt-4o-mini | Tokens: 23 | Cost: $0.000007 | Latency: 3412ms
LLMResponse(content='The capital city of Nepal is Kathmandu.', model='gpt-4o-mini', tokens=23, cost=7.05e-06, latency_ms=3412, timestamp='2025-11-07T10:48:17.242526')

Session Stats:
  Total Calls: 1
  Total Tokens: 23
  Total Cost: $7e-06
  Avg Cost/Call: $7e-06
```

## Resources

### Official Documentation

* [OpenAI API Reference](https://platform.openai.com/docs/api-reference) — Complete API documentation
    
* [OpenAI Tokenizer](https://platform.openai.com/tokenizer) — Visualize how text tokenizes
    
* [tiktoken GitHub](https://github.com/openai/tiktoken) — Token counting library
    
* [OpenAI Usage Tiers](https://platform.openai.com/docs/guides/rate-limits) — Understand your limits
    
* [OpenAI Pricing](https://openai.com/pricing) — Current pricing for all models
    
* [Production Best Practices](https://platform.openai.com/docs/guides/production-best-practices) — OpenAI's recommendations
    
* [Safety Best Practices](https://platform.openai.com/docs/guides/safety-best-practices) — Building safe AI systems
    
* [AWS Exponential Backoff and Jitter](https://aws.amazon.com/blogs/architecture/exponential-backoff-and-jitter/) — Retry pattern details
    
* [Real Python Async IO Guide](https://realpython.com/async-io-python/) — Async patterns in Python
    

---

*Questions or feedback? Open an issue on* [*Github*](https://github.com/chapainaashish/production-ai-engineering) *or reach out on* [*LinkedIn*](https://www.linkedin.com/in/chapainaashish/)