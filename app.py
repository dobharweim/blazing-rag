import os
import shutil
import asyncio

# Optional: comment out if it conflicts with Streamlit's event loop
# try:
#     loop = asyncio.get_event_loop()
# except RuntimeError:
#     loop = asyncio.new_event_loop()
#     asyncio.set_event_loop(loop)

import streamlit as st
from dotenv import load_dotenv

load_dotenv()  # So we can pick up GROQ_API_KEY from .env
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List

# Ensure there's a docs store in session_state
if "docs" not in st.session_state:
    st.session_state.docs = []

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    st.error("GROQ_API_KEY not set. Please set it in your environment or .env.")
    st.stop()

st.set_page_config(page_title="📚 RAG with ChatGroq & FAISS", layout="wide")
st.title("📚 RAG Notes Query with ChatGroq & FAISS")

def load_documents_and_copy(folder_path: str, extensions: List[str]):
    """Walk folder, copy matched files into 'uploads/' folder, and load them."""
    docs = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if any(file.endswith(ext) for ext in extensions):
                os.makedirs("uploads", exist_ok=True)
                src_path = os.path.join(root, file)
                dst_path = os.path.join("uploads", file)
                shutil.copyfile(src_path, dst_path)

                if file.endswith(".pdf"):
                    loader = PyPDFLoader(dst_path)
                else:
                    loader = TextLoader(dst_path)
                docs.extend(loader.load())
    return docs

@st.cache_resource
def create_index(_documents):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(_documents)
    embedding_fn = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.from_texts([doc.page_content for doc in chunks], embedding_fn)
    return vector_store, chunks

def init_llm():
    return ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model_name="llama-3.3-70b-versatile",
        temperature=0.0
    )

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

# ---------------
# Sidebar logic
# ---------------
with st.sidebar:
    st.header("📂 Document Loader")
    method = st.radio("Upload method:", ("Upload Files", "Specify Folder"))

    if method == "Upload Files":
        uploaded_files = st.file_uploader(
            "Upload .txt, .md, or .pdf",
            type=["txt", "md", "pdf"],
            accept_multiple_files=True
        )
        if uploaded_files:
            # We'll collect docs in a temporary list
            new_docs = []
            for f in uploaded_files:
                path = os.path.join("uploads", f.name)
                os.makedirs("uploads", exist_ok=True)
                with open(path, "wb") as file:
                    file.write(f.read())
                # Decide loader
                if f.name.endswith(".pdf"):
                    loader = PyPDFLoader(path)
                else:
                    loader = TextLoader(path)
                new_docs.extend(loader.load())

            if new_docs:
                # Store the newly loaded docs in session state
                st.session_state.docs = new_docs

    else:  # Specify Folder
        folder = st.text_input("Folder path:")
        exts_str = st.text_input("File extensions (comma-separated):", ".txt,.md,.pdf")
        exts = [ext.strip() for ext in exts_str.split(",")]

        if folder and st.button("Load"):
            new_docs = load_documents_and_copy(folder, exts)
            if new_docs:
                st.session_state.docs = new_docs

    # Show loaded file names
    if st.session_state.docs:
        with st.expander("Loaded Sources"):
            unique_filenames = set()
            for doc in st.session_state.docs:
                source_path = doc.metadata.get("source", "Unknown")
                unique_filenames.add(os.path.basename(source_path))
            st.write("Found the following files:")
            for filename in sorted(unique_filenames):
                st.markdown(f"- {filename}")

# ----------------
# Main logic
# ----------------
docs = st.session_state.docs  # Retrieve docs from session state
if docs:
    with st.spinner("Indexing documents..."):
        vector_store, _ = create_index(docs)
        llm = init_llm()
        qa_chain = setup_qa_chain(vector_store, llm)
    st.success("✅ Documents indexed and model ready!")

    query = st.text_input("Enter your question:")
    if query:
        with st.spinner("Fetching answer..."):
            response = qa_chain.invoke({"query": query})
            st.markdown("### Answer")
            st.write(response["result"])

            st.markdown("---\n### Source Snippets")
            for i, doc in enumerate(response["source_documents"]):
                snippet = doc.page_content[:500]
                st.markdown(
                    f"**Source {i + 1}:** {os.path.basename(doc.metadata.get('source', 'Unknown'))}"
                )
                st.write(f"> {snippet}...")
else:
    st.info("Upload files or specify a folder to get started.")
