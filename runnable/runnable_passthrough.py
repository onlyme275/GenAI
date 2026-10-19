from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.runnables import RunnableSequence, RunnableParallel, RunnablePassthrough
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

prompt1 = PromptTemplate(
    template="write a joke about {topic}",
    input_variables=['topic']
)

model = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0.7
)

parser = StrOutputParser()

prompt2 = PromptTemplate(
    template='Explain the following joke - {text}',
    input_variables=['text']
)

# chain = RunnableSequence(prompt1, model, parser, prompt2, model, parser)

# print(chain.invoke({'topic': 'AI'}))

joke_gen_chain = RunnableSequence(prompt1, model, parser)

parallel_chain = RunnableParallel({
    'joke': RunnablePassthrough(),
    'explaination': RunnableSequence(prompt2, model, parser)
})

final_chain = RunnableSequence(joke_gen_chain, parallel_chain)
print(final_chain.invoke({'topic': 'cricket'}))

