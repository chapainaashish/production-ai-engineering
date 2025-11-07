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
