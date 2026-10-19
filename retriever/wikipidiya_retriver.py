import requests
from langchain_community.retrievers import WikipediaRetriever

# Add a User-Agent to requests made by the wikipedia package.
original_request = requests.Session.request

def custom_request(self, method, url, **kwargs):
    headers = dict(kwargs.get("headers") or {})
    headers.setdefault(
        "User-Agent",
        "GenAIStudentProject/1.0 (educational use)"
    )
    kwargs["headers"] = headers
    return original_request(self, method, url, **kwargs)

requests.Session.request = custom_request

retriever = WikipediaRetriever(
    top_k_results=2,
    lang="en"
)

query = "Nepal geographical location and physical features"

docs = retriever.invoke(query)

for doc in docs:
    print("Title:", doc.metadata.get("title", "Unknown"))
    print(doc.page_content)
    print("-" * 50)