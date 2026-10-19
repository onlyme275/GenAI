from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

model = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0.7,
)

parser = StrOutputParser()

prompt = PromptTemplate(
    template="Generate 5 interesting facts about {topic}",
    input_variables=['topic']
)

chain = prompt | model | parser

result = chain.invoke({
    'topic': 'cricket'
})
print(result)
chain.get_graph().print_ascii()


