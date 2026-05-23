import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain.messages import SystemMessage, HumanMessage, ToolMessage
from langsmith import traceable

load_dotenv()

MAX_ITERATIONS = 10
MODEL="qwen3:1.7b"

@tool
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

@tool
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


@traceable(name="Langchain agent loop")
def run_agent(question: str):
    tools=[get_product_price, apply_discount]
    tools_dict = {tool.name: tool for tool in tools}
    llm = init_chat_model(f"ollama:{MODEL}",temprature=0)
    llm_with_tools = llm.bind_tools(tools)

    print(f"Question: {question}")
    messages = [SystemMessage(content="You are a helpful shopping assistant." \
    "You have access to a product catalog tool " \
    "and a discount tool.\n\n" \
    "STRICT RULES - you must follow these exactly:\n" \
    "1. NEVER guess or assume any product price. You MUST call get_product_price first to get the real price.\n" \
    "2. Only call apply_discount AFTER you have received a price from get_product_price. Pass the exact price returned by get_product_price to apply_discount.- do not pass a made-up number\n"
    "3. NEVER calculate discounts yourself using math. Alway use the apply_discount tool.\n" \
    "4. if the user does not specify a discount tier, ask them which tier to use - Do NOT assume one."),
                HumanMessage(content=question)]


def main():
    print("Initializing the chat model...")
    run_agent("What is the price of a laptop with a silver discount?")

if __name__ == "__main__":
    main()
