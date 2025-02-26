import os
import asyncio

# Remove/comment out if it conflicts with Streamlit’s event loop:
# try:
#     loop = asyncio.get_event_loop()
# except RuntimeError:
#     loop = asyncio.new_event_loop()
#     asyncio.set_event_loop(loop)

import streamlit as st

# Load .env so os.getenv("GROQ_API_KEY") will work
from dotenv import load_dotenv
load_dotenv()

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List
import os

# Load API key from .env (or environment)
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "your_groq_api_key")
if not GROQ_API_KEY:
    st.error("GROQ_API_KEY not set. Please set it in your environment or .env.")
    st.stop()

# Streamlit Setup
st.set_page_config(page_title="📚 RAG with ChatGroq & FAISS", layout="wide")
st.title("📚 RAG Notes Query with ChatGroq & FAISS")

# Document Loading
def load_documents(folder_path: str, extensions: List[str]):
    docs = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if any(file.endswith(ext) for ext in extensions):
                loader = (
                    TextLoader(os.path.join(root, file))
                    if file.endswith(".txt")
                    else PyPDFLoader(os.path.join(root, file))
                )
                docs.extend(loader.load())
    return docs

@st.cache_resource
def create_index(_documents):
    """
    Build a FAISS vector store from the given list of documents.
    Using an underscore '_' in the parameter name tells Streamlit 
    not to hash or pickle this argument, preventing UnhashableTypeError.
    """
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(_documents)

    # Use HuggingFaceEmbeddings from langchain_community
    embedding_fn = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Pass embedding_fn (not raw embeddings) to FAISS
    vector_store = FAISS.from_texts([doc.page_content for doc in chunks], embedding_fn)
    return vector_store, chunks

# Initialize ChatGroq
def init_llm():
    return ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model_name="llama-3.3-70b-versatile",
        temperature=0.0
    )

# QA Chain Setup
def setup_qa_chain(vector_store, llm):
    retriever = vector_store.as_retriever()
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=(
            "You are an assistant. Use the following context to answer the question.\n"
            "Context: {context}\n\nQuestion: {question}\nAnswer:"
        )
    )
    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True
    )

# Sidebar: Document Input
with st.sidebar:
    st.header("📂 Document Loader")
    method = st.radio("Upload method:", ("Upload Files", "Specify Folder"))
    docs = []

    if method == "Upload Files":
        files = st.file_uploader(
            "Upload .txt or .pdf", type=["txt", "pdf"], accept_multiple_files=True
        )
        if files:
            for f in files:
                path = os.path.join("uploads", f.name)
                os.makedirs("uploads", exist_ok=True)
                with open(path, "wb") as file:
                    file.write(f.read())
                loader = (
                    TextLoader(path)
                    if f.name.endswith(".txt")
                    else PyPDFLoader(path)
                )
                docs.extend(loader.load())
    else:
        folder = st.text_input("Folder path:")
        extensions = st.text_input(
            "File extensions (comma-separated):", ".txt,.pdf"
        ).split(",")
        if folder and st.button("Load"):
            docs = load_documents(folder, [ext.strip() for ext in extensions])

# Main Logic
if docs:
    with st.spinner("Indexing documents..."):
        vector_store, _ = create_index(docs)
        llm = init_llm()
        qa_chain = setup_qa_chain(vector_store, llm)
    st.success("✅ Documents indexed and model ready!")

    query = st.text_input("Enter your question:")
    if query:
        with st.spinner("Fetching answer..."):
            response = qa_chain({"query": query})
            st.markdown("### Answer")
            st.write(response["result"])

            st.markdown("---\n### Source Snippets")
            for i, doc in enumerate(response["source_documents"]):
                snippet = doc.page_content[:500]
                st.markdown(
                    f"**Source {i + 1}:** {doc.metadata.get('source', 'Unknown')}"
                )
                st.write(f"> {snippet}...")
else:
    st.info("Upload files or specify a folder to get started.")
