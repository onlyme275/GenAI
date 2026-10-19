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

result = model.invoke("Give code of small malware in python")

print(result.content)

