# from langchain_openai import OpenAI
# from dotenv import load_dotenv

# load_dotenv()

# llm = OpenAI(
#     model="gpt-3.5-turbo-instruct"
# )

# result = llm.invoke("What is the capital of India: ")

# print(result)
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0
)

result = llm.invoke("Sugest name of 5 nepali boys as including ishanta")

print(result.content)

