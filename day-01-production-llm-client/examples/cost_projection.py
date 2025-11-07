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
