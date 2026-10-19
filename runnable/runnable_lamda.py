from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence, RunnablePassthrough, RunnableParallel, RunnableLambda

from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate

load_dotenv()

def word_counts(text):
    return len(text.split())

prompt = PromptTemplate(
    template="Write a joke about {topic}",
    input_variables=['text']
)
model = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0.7
)
parser = StrOutputParser()
joke_gen_chain = RunnableSequence(prompt, model, parser)
# parallel_chain = RunnableParallel({
#     'joke': RunnablePassthrough(),
#     'word_count': RunnableLambda(lambda x: len(x.split()))
# })
parallel_chain = RunnableParallel({
    'joke': RunnablePassthrough(),
    'word_count': RunnableLambda(word_counts)
})

final_chain = RunnableSequence(joke_gen_chain, parallel_chain)

result = final_chain.invoke({'topic': 'AI'})
final_result = """{} \n word counts - {}""".format(result['joke'], result['word_count'])
print(final_result)
