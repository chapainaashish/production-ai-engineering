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
