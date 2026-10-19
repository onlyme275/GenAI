from langchain_groq  import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
load_dotenv()


model = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0.7,
)

class Person(BaseModel):
    name: str = Field(description="Name of the person")
    age:  str = Field(description="Age of the person")
    city: str = Field(description="Name of the city")

parser = PydanticOutputParser(pydantic_object=Person)

temp = PromptTemplate(
    template="Generate the name, age and city of a fictional {place} person \n {format_instruction}",
    input_variables=['place'],
    partial_variables={'format_instruction': parser.get_format_instructions()}
)

# prompt = temp.invoke({
#     'place': 'american'
# })
# print(prompt)
# result = model.invoke(prompt)
# final_result = parser.parse(result.content)

chain = temp | model | parser
print(chain)
final_result = chain.invoke({
    'place': 'nepali'
})
print(final_result)


