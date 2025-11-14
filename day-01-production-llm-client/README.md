## Overview

The first way to interact with any LLM is through the API in development. This is pretty basic which you will master easily but you can miss some critical aspect which I will cover in this post.

## Part 1: Understand Tokens Before Getting Excited

Before calling the LLM, tokens are important to understand. So, when you send "Hello, world!" to GPT, the model doesn't process it as two words. It breaks it down into smaller units called tokens. Tokens directly impact everything in LLM development.

OpenAI uses a tokenization algorithm called **Byte Pair Encoding (BPE)**, implemented in their `tiktoken` library. BPE algorithm doesn't split text into tokens using spaces or punctuation, It's more complex. You can read more about it [here](https://www.kaggle.com/code/qmarva/1-bpe-tokenization-algorithm-eng)

**Some approximations:**

* 1 token ≈ 4 characters in English
    
* Non-English text uses significantly more tokens (sometimes 2-3x more)
    

### Tokens are Currency

**1\. Pricing is Per-Token**

Every API call costs money based on token count. So, if you miscalculate tokens by 50%, you're miscalculating your budget by 50%. At scale, this means thousands of dollars in unexpected costs.

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

Look "Hello, world!" is 4 tokens, not 2. The comma and space are separate tokens. This is why you can't estimate tokens by counting words.

---

## Part 2: Calling the LLM API Right Way

### Hide Secrets in .env

Before we call the OpenAI API, we need to set up our API key properly. Never hardcode API keys in your code. Sounds pretty basic but most of the people miss it.

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

Before deploying your app into production, understand token-based pricing first because LLM API calls can get expensive as you scale.

### Current Pricing

As of November 2025, OpenAI charges:

| Model | Input (per 1M tokens) | Output (per 1M tokens) |
| --- | --- | --- |
| gpt-5-nano | $0.05 | $0.40 |
| gpt-5 | $1.25 | $10.00 |

**What you can understand from this table:**

* Output tokens cost more than input tokens (8x more for both models)
    
* GPT-5 is 25x more expensive than GPT-5 nano for both input and output
    

### Real-World Cost Projection

Let's calculate what a real production system would cost. Imagine you're building a customer support chatbot with this specific scenario:

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

Alright, Now you have seen the cost, the model choice depends on your use-case. For straightforward tasks like summarization and classification, you can use GPT nano version and for complex task, you can use GPT 5 version. Furthermore, you can use [Openrouter](https://openrouter.ai/rankings) to compare models according to your use-case.

## Part 4: Error Handling & Retries are Important

LLM APIs might fail. So, you shouldn't assume they are immune to errors. You should treat them like any other third party APIs or even give more importance if you are building the system around it.

### Some Common API Errors

1\. Rate Limit Errors (HTTP 429)

2\. Timeout Errors

3\. API Errors (HTTP 500-599)

4\. Authentication Errors (HTTP 401)

5\. Invalid Request Errors (HTTP 400)

### Implementing Exponential Backoff and Timeout

The standard approach to handling retries is **exponential backoff with jitter.** OpenAI standard SDK offers in-built retries and timeout. So, you can leverage that.

```python
import asyncio
import os
from dotenv import load_dotenv
from openai import AsyncOpenAI
from openai import APIError, APIConnectionError, RateLimitError, Timeout

load_dotenv()

client = AsyncOpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    max_retries=3,  # Automatic exponential backoff with jitter
    timeout=30.0
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
```

You might wonder about how `max_retires` works under the hood. It use something call **exponential backoff with jitter** which retry the API every few random seconds. This few random seconds are calculated by multiplying the attempt with two and adding jitter;

```python
wait_time = (2 ** attempt) + random.uniform(0, 1)
```

The first expression give the result like (1s, 2s, 4s) and second expression is jitter which is random value between 0 and 1. You might be wondering why not just use `2 ** attempt` and get rid of jitter?

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