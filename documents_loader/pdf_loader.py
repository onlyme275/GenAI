from langchain_community.document_loaders import PyMuPDFLoader

loader = PyMuPDFLoader('assembly_language_guide.pdf')
docs = loader.load()
print(len(docs))