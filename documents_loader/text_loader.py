from langchain_community.document_loaders import TextLoader
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

loader = TextLoader('cricket.txt', encoding='utf-8')
docs = loader.load()
print(type(docs))   
print(len(docs))
# print(docs[0])

prompt1 = PromptTemplate(
    template="write a summary for the following poem - \n{poem}",
    input_variables=['poem']
)

model = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0.7
)
parser = StrOutputParser()

chain = prompt1 | model | parser
print(chain.invoke({'poem': docs[0].page_content}))