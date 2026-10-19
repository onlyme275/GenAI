from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel
load_dotenv()

prompt1 = PromptTemplate(
    template="Generate short and simple notes following text \n {text}",
    input_variables=['text']
)

prompt2 = PromptTemplate(
    template='Generate 5 short question answer the following text \n {text}',
    input_variables=['text']
)

prompt3 = PromptTemplate(
    template='Merge the provided notes an quiz into an single document \n notes -> {notes} and quiz -> {quiz}',
    input_variables=['notes', 'quiz']
)

parser = StrOutputParser()

model1 = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0.7
)

model2 = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0.7
)

parallel_chains = RunnableParallel({
    'notes': prompt1 | model1 | parser,
    'quiz': prompt2 | model2 | parser
})

merge_chains = prompt3 | model1 | parser
chain = parallel_chains | merge_chains

text = """
potato, (Solanum tuberosum), annual plant in the nightshade family (Solanaceae), grown for its starchy edible tubers. The potato is native to the Peruvian-Bolivian Andes and is now cultivated on every inhabited continent and is one of the world’s main food crops, valued for its high yields and adaptability to diverse climates. Potatoes are consumed in a variety of forms, including as cooked vegetables, as processed foods such as french fries and potato chips, and as ingredients such as potato starch and potato flour, which are used in cooking, baking, and food manufacturing. They are also used to make vodka and aquavit. The tubers are highly digestible and supply vitamin C, potassium, thiamin, and niacin; when eaten with their skins, they also provide dietary fiber.
"""

result = chain.invoke({'text': text})
print(result)

chain.get_graph().print_ascii()