import ollama
import re
import inspect
from dotenv import load_dotenv
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


tools = {
    "get_product_price": get_product_price,
    "apply_discount": apply_discount
}

def get_tool_descriptions(tools):
    descriptions = []
    for tool_name, tool_func in tools.items():
        function = getattr(tool_func,"__wrapped__", tool_func)
        sig = inspect.signature(function)
        docstring = inspect.getdoc(function) or ""
        descriptions.append(f"{tool_name}{sig} - {docstring}")
    
    return "\n".join(descriptions)

tool_descriptions = get_tool_descriptions(tools)
tool_names = ", ".join(tools.keys())

prompt = f"""Answer the following questions as best you can. You have access to the following tools:

{tool_descriptions}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {{question}}
Thought:"""


@traceable(name="Ollama chat", run_type="llm")
def ollama_chat_traced(model, messages, options):
    return ollama.chat(model=model, messages=messages, options=options)



@traceable(name="Ollama agent loop")
def run_agent(question: str):
    print(f"Question: {question}")
    react_prompt = prompt.format(question=question)
    scratchpad = ""

    for i in range(MAX_ITERATIONS-1):
        print(f"\n--- Iteration {i+1} ---")
        full_prompt = react_prompt+scratchpad
        ai_message = ollama_chat_traced(
            model=MODEL,
            messages = [{"role": "system", "content": full_prompt}],
            options={"stop":  ["\nObservation"],"temperature": 0}
            ).message
        output = ai_message.content
        final_answer_match = re.search(r"Final Answer:\s*(.+)", output)
        if final_answer_match:
            final_answer = final_answer_match.group(1).strip()
            print("Final Answer:", final_answer)
            return final_answer


        action_match = re.search(r"Action:\s*(.+)", output)
        action_input_match = re.search(r"Action Input:\s*(.+)", output)

        if not action_match or not action_input_match:
            print("AI response:", output)
            return output
        
        tool_name = action_match.group(1).strip()
        tool_input = action_input_match.group(1).strip()

        print(f"AI wants to call tool: {tool_name} with input: {tool_input}")

        raw_args = [x.strip() for x in tool_input.split(",")]
        args = [x.split("=",1)[-1].strip().strip("'\"") for x in raw_args]

        if tool_name not in tools:
            observation = f"Tool {tool_name} not found."
        else:
            observation = str(tools[tool_name](*args))

    scratchpad += f"{output}\nObservation: {observation}\nThought:"




def main():
    print("Initializing the chat model...")
    print(run_agent("What is the price of a laptop with a silver discount?"))

if __name__ == "__main__":
    main()
