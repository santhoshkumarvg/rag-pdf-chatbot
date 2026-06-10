import os
from dotenv import load_dotenv

import google.generativeai as genai

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# Load .env
load_dotenv()

# Gemini API Key
genai.configure(
    api_key=os.getenv("GOOGLE_API_KEY")
)

# Gemini Model
model = genai.GenerativeModel("gemini-2.5-flash")

# Embeddings
embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Load Chroma DB
vectordb = Chroma(
    persist_directory="db",
    embedding_function=embedding
)

while True:

    question = input("\nAsk a Question (type exit to quit): ")

    if question.lower() == "exit":
        break

    # Retrieve relevant chunks
    docs = vectordb.similarity_search(
        question,
        k=5
    )

    context = "\n\n".join(
        [doc.page_content for doc in docs]
    )

    prompt = f"""
You are a helpful assistant.

Answer ONLY using the context below.

Context:
{context}

Question:
{question}

If the answer is not present in the context,
say:
"I could not find that information in the document."
"""

    response = model.generate_content(prompt)

    print("\nAnswer:\n")
    print(response.text)