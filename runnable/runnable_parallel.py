from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnableSequence
load_dotenv()
prompt1 = PromptTemplate(
    template="Generate a tweet about {topic}",
    input_variables=['topic']
)
prompt2 = PromptTemplate(
    template="Generate a linkedin post about {topic}",
    input_variables=['topic']
)
model = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0.7
)
parser = StrOutputParser()

parallel_chain = RunnableParallel({
    'tweet': RunnableSequence(prompt1, model, parser),
    'linkedin': RunnableSequence(prompt2, model, parser)
})
result = parallel_chain.invoke({'topic': 'AI'})
print(result)

