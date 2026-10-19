from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-72B-Instruct",
    task="text-generation",
    max_new_tokens=100,
    temperature=0.7,
)

model = ChatHuggingFace(llm=llm)

# Prompt 1
temp1 = PromptTemplate(
    template="Write a detailed summary report on {text}",
    input_variables=["text"]
)

# Prompt 2
temp2 = PromptTemplate(
    template="Write a short summary of the following report:\n{text}",
    input_variables=["text"]
)

# First prompt
prompt1 = temp1.invoke({
    "text": "black hole"
})

result1 = model.invoke(prompt1)

# Second prompt receives the output of the first model
prompt2 = temp2.invoke({
    "text": result1.content
})

result2 = model.invoke(prompt2)

print(result2.content)
