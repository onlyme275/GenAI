from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled, NoTranscriptFound, RequestBlocked
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
import os
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()
video_id = "T2fjQrsKbAM"
try:
    ytt_api = YouTubeTranscriptApi()
    transcript_list = ytt_api.fetch( video_id,  languages=["en"] )
    transcript = " ".join(   chunk.text for chunk in transcript_list )
    print(transcript)
except RequestBlocked:
    print("YouTube blocked this request. Try again later.")
except TranscriptsDisabled:
    print("No captions available for this video")
except NoTranscriptFound:
    print("No English transcript found for this video.")

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.create_documents([transcript])
print("length of chunks: \n", len(chunks))
print("chunks first part: \n", chunks[0])

# embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
if os.getenv("HUGGINGFACEHUB_API_TOKEN"):
    os.environ.setdefault("HF_TOKEN", os.environ["HUGGINGFACEHUB_API_TOKEN"])
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
vector_store = FAISS.from_documents(chunks, embeddings)

print("vector_store: \n", vector_store.index_to_docstore_id)
print("vector_store_get_by_ids: \n", vector_store.get_by_ids(["40265d81-74c0-47be-ae0f-0536b299c77c"]))

retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})
print("Retriever: \n", retriever)

retriever.invoke("What is dog?")
print("Retriever invoke: \n", retriever.invoke("What is dog?"))

llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0.7
)

prompt = PromptTemplate(
    template="""
        You are a helpful assistant.
        Answer ONLY from the provided transcript context.
        If the context is insufficient, just say you don't know.

        {context}
        Question: {question}
    """,
    input_variables=['context', 'question']
)

question = "is the topic of aliens discussed in this video? if yes then what was discussed"
retrieved_docs = retriever.invoke(question)

context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)

final_prompt = prompt.invoke({"context": context_text, "question": question})

answer = llm.invoke(final_prompt)
print("Answer: \n", answer.content)

# Building Chain
print("\n============================\n")
print("****** CHAIN BUILDINGS ******\n")
print("=============================\n")

from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

def format_docs(retrieved_docs):
    context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
    return context_text

parallel_chain = RunnableParallel({
    'context': retriever | RunnableLambda(format_docs),
    'question': RunnablePassthrough()
})
parser = StrOutputParser()
main_chain = parallel_chain | prompt | llm | parser

answer2 = parallel_chain.invoke("About cats")
print("\nAnswer2: \n", answer2)

final_answer = main_chain.invoke("Can you summarize the video")
print("\nFinal Answer: \n", final_answer)

