from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-72B-Instruct",
    task="text-generation",
    max_new_tokens=100,
    temperature=0.7,
)
model = ChatHuggingFace(llm=llm)

message = [
    SystemMessage(content='You are a helpful assistant'),
    HumanMessage(content='Tell me about Langchain')
]

result = model.invoke(message)

message.append(AIMessage(result.content))

print(message)

