import os
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.documents import Document
from dotenv import load_dotenv

try:
    from langchain.retrievers import ContextualCompressionRetriever
    from langchain.retrievers.document_compressors import LLMChainExtractor
except ModuleNotFoundError:
    from langchain_classic.retrievers import ContextualCompressionRetriever
    from langchain_classic.retrievers.document_compressors import LLMChainExtractor

load_dotenv()

# Use OpenAIEmbeddings if OPENAI_API_KEY is provided, otherwise fallback to HuggingFaceEmbeddings
if os.getenv("OPENAI_API_KEY"):
    from langchain_openai import OpenAIEmbeddings
    embedding_model = OpenAIEmbeddings()
else:
    from langchain_huggingface import HuggingFaceEmbeddings
    if os.getenv("HUGGINGFACEHUB_API_TOKEN"):
        os.environ.setdefault("HF_TOKEN", os.environ["HUGGINGFACEHUB_API_TOKEN"])
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

docs = [
    Document(page_content=(
        """
        assembly hello world
    """
    ), metadata={'source': 'Doc1'}),
    Document(page_content=(
        """
        assembly doing file system
    """
    ))
]

vectorstore = FAISS.from_documents(docs, embedding_model)

base_retriver = vectorstore.as_retriever(search_kwargs={'k': 5})

llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0.7
)
compressor = LLMChainExtractor.from_llm(llm)

compression_retriever = ContextualCompressionRetriever(
    base_retriever=base_retriver,
    base_compressor=compressor
)
query = "What is assembly?"
compressed_results = compression_retriever.invoke(query)

if not compressed_results:
    print(f"\nNo relevant content found for query: '{query}' (Contextual compressor filtered out irrelevant docs)")
else:
    for i, doc in enumerate(compressed_results):
        print(f"\n--- Result {i+1} ---")
        print(doc.page_content)
