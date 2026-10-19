from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import streamlit as st

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-0.5B-Instruct",
    task="text-generation",
    max_new_tokens=100,
    temperature=0.7,
)

model = ChatHuggingFace(llm=llm)

st.header("Research Tool")

user_input = st.text_input("Enter your prompt:")

if st.button("Summarize"):
    if user_input:
        result = model.invoke(user_input)
        st.write(result.content)
    else:
        st.warning("Please enter a prompt.")
