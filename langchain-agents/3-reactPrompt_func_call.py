import os
from dotenv import load_dotenv
import ollama
from langsmith import traceable
import json

load_dotenv()

MAX_ITERATIONS = 10
MODEL="qwen3:1.7b"

@traceable(run_type="tool")
def get_product_price(product:str) -> float:
    """Looks up the price of a product."""
    # Simulate fetching product price
    print(f"Executing get_product_price for {product}...")
    prices = {
        "laptop": 999.99,
        "keyboard": 499.99,
        "headphones": 199.99
    }
    return prices.get(product, 0.0)

@traceable(run_type="tool") 
def apply_discount(price: float, discount_tier: str) -> float:
    """Applies a discount to the given price and return the final price.
    Available tiers: bronze, silver, gold."""
    print(f"Executing apply_discount for price: {price} and discount_tier: {discount_tier}...")
    discounts_percentages = {
        "bronze": 5,
        "silver": 12,
        "gold": 23
    }
    discount_rate = discounts_percentages.get(discount_tier.lower(), 0.0) / 100
    return round(price * (1 - discount_rate), 2)

#Ollama can still call the functions without tool schema, by using goolge's docstring format, something as below.
# def get_product_price(product:str) -> float:
#     """Looks up the price of a product.
#     Args:
#         product (str): The name of the product to look up the price for.
#     Returns:
#         float: The price of the product.
#     """
#     # Simulate fetching product price

tools_for_llm = [{
      "type": "function",
      "function": {
        "name": "get_product_price",
        "description": "Looks up the price of a product.",
        "parameters": {
          "type": "object",
          "required": ["product"],
          "properties": {
            "product": {"type": "string", "description": "The name of the product to look up the price for"}
          }
        }
      }
},
{
      "type": "function",
      "function": {
        "name": "apply_discount",
        "description": "Applies a discount to the given price and return the final price. Available tiers: bronze, silver, gold.",
        "parameters": {
          "type": "object",
          "required": ["price", "discount_tier"],
          "properties": {
            "price": {"type": "number", "description": "The price of the product"},
            "discount_tier": {"type": "string", "description": "The discount tier to apply (bronze, silver, gold)"}
          }
        }
      }
    }
]

@traceable(name="Ollama chat", run_type="llm")
def ollama_chat_traced(messages):
    return ollama.chat(model=MODEL, messages=messages, tools=tools_for_llm)

@traceable(name="Ollama agent loop")
def run_agent(question: str):
    tools=[get_product_price, apply_discount]
    tools_dict = {
        "get_product_price": get_product_price,
        "apply_discount": apply_discount
    }

   
    print(f"Question: {question}")
    messages = [
        {"role": "system", "content": "You are a helpful shopping assistant." \
    "You have access to a product catalog tool " \
    "and a discount tool.\n\n" \
    "STRICT RULES - you must follow these exactly:\n" \
    "1. NEVER guess or assume any product price. You MUST call get_product_price first to get the real price.\n" \
    "2. Only call apply_discount AFTER you have received a price from get_product_price. Pass the exact price returned by get_product_price to apply_discount.- do not pass a made-up number\n"
    "3. NEVER calculate discounts yourself using math. Alway use the apply_discount tool.\n" \
    "4. if the user does not specify a discount tier, ask them which tier to use - Do NOT assume one."},{
                   "role": "user", "content": question
              }]
    


    for i in range(MAX_ITERATIONS-1):
        print(f"\n--- Iteration {i+1} ---")
        ai_message = ollama_chat_traced(messages = messages).message
        tool_calls = ai_message.tool_calls
        if not tool_calls:
            print("AI response:", ai_message.content)
            return ai_message.content

        tool_call = tool_calls[0]
        tool_name = tool_call.function.name
        tool_args = tool_call.function.arguments
        print(f"AI wants to call tool: {tool_name} with args: {tool_args}")

        tool_to_use = tools_dict.get(tool_name)
        if tool_to_use is None:
            print(f"Tool {tool_name} not found. Skipping tool call.")
            continue
        observation = tool_to_use(**tool_args)
        print(f"Observation from tool {tool_name}: {observation}")
        messages.append(ai_message)
        messages.append({"role": "tool", "content": str(observation)})
        # print(f"-----------------{messages}-----------------")



def main():
    print("Initializing the chat model...")
    print(run_agent("What is the price of a laptop with a silver discount?"))

if __name__ == "__main__":
    main()
