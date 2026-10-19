
from langchain.tools import BaseTool
from pydantic import BaseModel, Field


# 1. Define the input schema
class MultiplyInput(BaseModel):
    a: int = Field(description="The first number to multiply")
    b: int = Field(description="The second number to multiply")


# 2. Define the actual tool
class MultiplyTool(BaseTool):
    name: str = "multiply"
    description: str = "Multiply two integers and return the result."
    args_schema: type[BaseModel] = MultiplyInput

    def _run(self, a: int, b: int) -> int:
        return a * b


# 3. Create and execute the tool
multiply_tool = MultiplyTool()

result = multiply_tool.invoke({"a": 3, "b": 2})

print("Result:", result)
print("Name:", multiply_tool.name)
print("Description:", multiply_tool.description)
print("Arguments:", multiply_tool.args)
