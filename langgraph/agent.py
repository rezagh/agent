from langchain.chat_models import init_chat_model
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import InMemorySaver
from config import get_api_key

def get_weather(city: str) -> str:  
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"

agent = create_react_agent(
    model="anthropic:claude-3-7-sonnet-latest",  
    tools=[get_weather],  
    prompt="You are a helpful assistant"  
)

# Run the agent
# agent.invoke(
#     {"messages": [{"role": "user", "content": "what is the weather in sf"}]}
# )







checkpointer = InMemorySaver()



model = init_chat_model(
    "anthropic:claude-3-7-sonnet-latest",
    temperature=0
)

agent = create_react_agent(
    model=model,
    tools=[get_weather],
    # A static prompt that never changes - this is a system message
    prompt="Never answer questions about the weather.",
    checkpointer=checkpointer
)


# agent.invoke(
#     {"messages": [{"role": "user", "content": "what is the weather in sf"}]}
# )


"""To allow multi-turn conversations with an agent, you need to enable persistence by providing a checkpointer when creating an agent. 
At runtime, you need to provide a config containing thread_id — a unique identifier for the conversation (session):
"""


config = {"configurable": {"thread_id": "1"}}
sf_response = agent.invoke(
    {"messages": [{"role": "user", "content": "what is the weather in sf"}]},
    config  
)

print(sf_response)
ny_response = agent.invoke(
    {"messages": [{"role": "user", "content": "what about new york?"}]},
    config
)
print(ny_response)