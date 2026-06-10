from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load PDF
loader = PyPDFLoader("data/sample.pdf")
docs = loader.load()

print("Total Pages:", len(docs))

# Chunking
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = splitter.split_documents(docs)

print("\nTotal Chunks:", len(chunks))

print("\nFirst Chunk:\n")
print(chunks[0].page_content)