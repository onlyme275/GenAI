# from langchain_openai import OpenAIEmbeddings
# from dotenv import load_dotenv

# load_dotenv()

# embedding = OpenAIEmbeddings(model='text-embedding-3-large', dimensions=32)

# result = embedding.embed_query("Nepal is the capital of nepal")

# print(str(result))

from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()

embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

result = embedding.embed_query("Nepal is a beautiful country")

print(result)
print(len(result))