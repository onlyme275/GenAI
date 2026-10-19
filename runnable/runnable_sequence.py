from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence
from langchain_core.output_parsers import StrOutputParser
load_dotenv()

prompt = PromptTemplate(
    template="Write a joke about {topic}",
    input_variables=['topic']
)

prompt1 = PromptTemplate(
    template="Explain about joke {topic}",
    input_variables=['topic']
)

model = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0.7
)

parser = StrOutputParser()

chain = RunnableSequence(prompt, model, parser, prompt1, model, parser)
print(chain.invoke({'topic': 'AI'}))
