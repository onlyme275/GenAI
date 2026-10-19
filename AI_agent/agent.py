
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_classic.agents import (
    AgentExecutor,
    create_tool_calling_agent,
)
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

# 1. Search tool
search_tool = DuckDuckGoSearchRun()
tools = [search_tool]

# 2. Groq model
llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0.2,
)

# 3. Chat prompt for native tool calling
prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a helpful assistant. Use the search tool when "
        "current or external information is needed. "
        "Answer accurately and clearly."
    ),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])

agent = create_tool_calling_agent(
    llm=llm,
    tools=tools,
    prompt=prompt,
)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    max_iterations=5,
)

response = agent_executor.invoke({
    "input": "Explain 3 ways to travel from morang to kathmandu."
})

print("\nFinal Answer:")
print(response["output"])
