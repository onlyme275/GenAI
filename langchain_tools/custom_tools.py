from langchain_core.tools import tool

def multiply(a, b):
    """Multiply two integers and return the result........"""
    print("first multiply")
    return a * b

#add type hints
def multiply(a: int, b: int) -> int:
    """Multiply two integers and return the result..."""
    print("second multiply")
    return  a * b

@tool
def multiply(a: int, b: int) -> int:
    """Multiply two integers and return the result."""
    print("third multiply")
    return a * b

result = multiply.invoke({'a': 2, 'b': 3})
print(result)


print(multiply.name)
print(multiply.description)
print(multiply.args)
print("multiply: \n")
print(multiply.args_schema.model_json_schema())