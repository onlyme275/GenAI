from langchain_text_splitters import CharacterTextSplitter   
from langchain_community.document_loaders import PyMuPDFLoader

loader = PyMuPDFLoader('../documents_loader/assembly_language_guide.pdf')
docs = loader.load()

text = """
    Cricket is one of the most popular sports in the world, especially in countries like Nepal, India, Australia, England, and Pakistan. It is a team sport played between two teams, with eleven players on each side. The main objective of cricket is to score more runs than the opposing team.

Cricket has three major formats: Test cricket, One Day International (ODI), and Twenty20 (T20). Test cricket can last up to five days, ODI matches consist of 50 overs per team, and T20 matches consist of 20 overs per team.

"""

splitter = CharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=0,
    separator=''
)
result = splitter.split_text(text)
result1 = splitter.split_documents(docs)
print(result)
print('\n')
print("result 1:", result1[0])
