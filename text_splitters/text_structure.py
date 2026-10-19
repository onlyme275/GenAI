from langchain_text_splitters import RecursiveCharacterTextSplitter

text = """
    Cricket is one of the most popular sports in the world, especially in countries like Nepal, India, Australia, England, and Pakistan. It is a team sport played between two teams, with eleven players on each side. The main objective of cricket is to score more runs than the opposing team.
    
    Cricket has three major formats: Test cricket, One Day International (ODI), and Twenty20 (T20). Test cricket can last up to five days, ODI matches consist of 50 overs per team, and T20 matches consist of 20 overs per team.
"""


splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=0,
)
chunks = splitter.split_text(text)
print(len(chunks))
print(chunks)
