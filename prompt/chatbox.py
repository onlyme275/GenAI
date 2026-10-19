from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-72B-Instruct",
    task="text-generation",
    max_new_tokens=100,
    temperature=0.7,
)
model = ChatHuggingFace(llm=llm)
chat_history = [
    SystemMessage(content='You are a helpful AI assistant')
]

while True:
    user_input = input("You: ")
    chat_history.append(HumanMessage(user_input))
    if user_input == "exit":
        break

    result = model.invoke(chat_history)
    chat_history.append(AIMessage(result.content))
    print("AI: ", result.content)

print(chat_history)


