import os
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
# from tavily import TavilyClient
from langchain_tavily import TavilySearch

load_dotenv()

# tavily = TavilyClient()

# @tool
# def search(query: str) -> str:
#     """
#     Tool that searches over internet
#     Args:        
#         query: search query
#     Returns:
#         The search results
#     """
#     print(f"Searching for: {query}")
#     return tavily.search(query=query)

class Source(BaseModel):
    """Schema for a source used by the agent."""
    url: str = Field(description="The URL of the source")

class AgentResponse(BaseModel):
    """Schema for the agent's response with answer and sources."""
    answer: str = Field(description="The answer provided to the query")
    sources: list[Source] = Field(default_factory=list, description="List of sources used to generate the answer")


llm = ChatOpenAI(model="gpt-5")
# tools = [search]
tools = [TavilySearch()]
agent = create_agent(model=llm, tools=tools,response_format=AgentResponse)

def main():
    print("Hello from react-search-agent!")
    result = agent.invoke({"messages": [HumanMessage(content="Find a 3 job openings for Langchain engineers in Milton, Ontario Canada in LinkedIn")]})
    print(result)


if __name__ == "__main__":
    main()
