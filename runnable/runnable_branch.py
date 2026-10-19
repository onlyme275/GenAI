from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence, RunnableParallel, RunnableLambda, RunnablePassthrough, RunnableBranch
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

prompt1 = PromptTemplate(
    template="Write a detailed report on {topic}",
    input_variables=['topic']
)
prompt2 = PromptTemplate(
    template="Summarize the following text \n {text}",
    input_variables=['text']
)
model = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0.7
)
parser = StrOutputParser()
report_gen_chain = RunnableSequence(prompt1, model, parser)

branch_chain = RunnableBranch(
    (lambda x: len(x.split())>500, 
    RunnableSequence(
        RunnablePassthrough(lambda x : {'text': x}), prompt2, model, parser)),
    (RunnablePassthrough())
)

final_chain = report_gen_chain | branch_chain

print(final_chain.invoke({'topic': 'Russia vs Ukraine'}))

