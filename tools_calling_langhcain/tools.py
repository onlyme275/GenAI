import os
import json
from langchain_groq import ChatGroq
from langchain_core.tools import tool, InjectedToolArg
from typing import Annotated
from langchain_core.messages import HumanMessage
import requests
from dotenv import load_dotenv

load_dotenv()

@tool
def multiply( a: int, b: int) -> int:
    """Give two number as a and b"""
    return a * b

print(multiply.invoke({'a': 2, 'b': 3}))

model = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0.7,
)

llm_with_tools = model.bind_tools([multiply])
print(llm_with_tools.invoke('can you multiply 3 with 10').tool_calls[0])


query = HumanMessage('can you multilply 3 with 10')
message = [query]
result = llm_with_tools.invoke(message)

tool_result = multiply.invoke(result.tool_calls[0])
message.append(tool_result)
print(message)

@tool
def get_conversion_factor(base_currency: str, target_currency: str) -> dict:
    """ Fetches exchange rate between base_currency and target_currency (use 3-letter currency codes like USD, NPR). """
    url = f"https://v6.exchangerate-api.com/v6/d33e25cafe04311f2bf66fd1/pair/{base_currency}/{target_currency}"
    response = requests.get(url)
    return response.json()

@tool
def convert(base_currency_value: int, conversion_rate: Annotated[float, InjectedToolArg]):
    """ Given a currency conversion """
    return base_currency_value * conversion_rate

print(get_conversion_factor.invoke({'base_currency': 'USD', 'target_currency': 'NPR'}))

print(convert.invoke({
    'base_currency_value': 10, 
    'conversion_rate': 85.16
}))

model_with_tools = model.bind_tools([get_conversion_factor, convert])
messages = [HumanMessage('what is the conversion factor of the USD and Nepali, and that you can convert 10 usd into nep')]
ai_message = model_with_tools.invoke(messages)

print(ai_message)
print(ai_message.tool_calls)

messages.append(ai_message)
conversion_rate = None
for tool_call in ai_message.tool_calls:
    if tool_call['name'] == 'get_conversion_factor':
        tool_message1 = get_conversion_factor.invoke(tool_call)
        conversion_rate = json.loads(tool_message1.content)['conversion_rate']
        messages.append(tool_message1)
        print("get_conversion_factor output:", tool_message1.content)

    if tool_call['name'] == 'convert':
        tool_call['args']['conversion_rate'] = conversion_rate
        tool_message2 = convert.invoke(tool_call)
        messages.append(tool_message2)
        print("convert output:", tool_message2.content)

# Follow up with the model to get the final answer if tool messages were added
if len(messages) > 2:
    final_response = model_with_tools.invoke(messages)
    print("\nFinal LLM Response:")
    print(final_response.content)

        