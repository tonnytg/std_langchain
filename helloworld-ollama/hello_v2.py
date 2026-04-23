from langchain.agents import create_agent
from langchain_ollama import ChatOllama
from langchain.tools import tool
import json

@tool(return_direct=True)
def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"

llm = ChatOllama(
        model="llama3.2:3b",
        temperature=1
)

agent = create_agent(
    model=llm,
    tools=[get_weather],
    system_prompt = """
You are a helpful assistant.

When you need to use a tool:
- ONLY use the tool calling system
- DO NOT output JSON
- DO NOT describe the tool
- DO NOT return tool calls as text

Always use the tool properly.
"""
)

# Run the agent
response = agent.invoke(
    {"messages": [{"role": "user", "content": "what is the weather in sf"}]}
)

last = response["messages"][-1].content
try:
    data = json.loads(last)
    if "parameters" in data:
        result = get_weather.invoke(data["parameters"])
        print(result)
except:
    print(last)

#print(response["messages"][-1].content)
