from pdf_processor import process_pdfs
import os
from dotenv import load_dotenv

import streamlit as st

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

st.sidebar.title("RAG PDF Chatbot")
st.sidebar.write("Upload PDFs and ask questions")

if st.sidebar.button("Clear Chat"):
    st.session_state.chat_history = []

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


import google.generativeai as genai

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma


load_dotenv()

genai.configure(
    api_key=os.getenv("GOOGLE_API_KEY")
)

model = genai.GenerativeModel("gemini-2.5-flash")

@st.cache_resource
def load_embedding():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

embedding = load_embedding()

vectordb = Chroma(
    persist_directory="db",
    embedding_function=embedding
)

st.title("🤖 AI-Powered Document Assistant")

st.caption(
    "Built with Streamlit, LangChain, ChromaDB, HuggingFace Embeddings and Gemini"
)

uploaded_files = st.file_uploader(
    "Upload PDFs",
    type=["pdf"],
    accept_multiple_files=True
)

if uploaded_files:
    for uploaded_file in uploaded_files:

        with open(uploaded_file.name, "wb") as f:
            f.write(uploaded_file.getbuffer())

    st.success("PDFs Uploaded Successfully")

question = st.chat_input("Ask a Question")

if uploaded_files:

    if st.button("Process PDF"):

        pdf_paths = []

        for uploaded_file in uploaded_files:
            pdf_paths.append(uploaded_file.name)

        with st.spinner("Processing PDFs..."):
            result = process_pdfs(pdf_paths)

        st.success(result)

    if question:

        docs = vectordb.similarity_search(
            question,
            k=3
        )

        context = "\n\n".join(
            [doc.page_content for doc in docs]
        )

        history = "\n".join(st.session_state.chat_history)

        prompt = f"""
        You are a helpful AI assistant.

        Answer ONLY from the provided context.

        If the answer is not available in the context, say:

        "Answer not found in the uploaded documents."

        Context:
        {context}

        Question:
        {question}
        """

        response = model.generate_content(prompt)

        answer = response.text

        st.session_state.messages.append(
            {"role": "user", "content": question}
        )

        st.session_state.messages.append(
            {"role": "assistant", "content": answer}
        )

        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            st.write(answer)

        st.session_state.chat_history.append(
            f"User: {question}"
        )

        st.session_state.chat_history.append(
            f"Assistant: {answer}"
        )

        st.divider()
        st.subheader("Sources Used")

        for i, doc in enumerate(docs):
            with st.expander(f"Source {i+1}"):

                st.write(
                    f"PDF: {doc.metadata.get('source', 'Unknown')}"
                )

                st.write(doc.page_content)