import os
import shutil
import json

import streamlit as st
from dotenv import load_dotenv

# Load environment variables (for GROQ_API_KEY)
load_dotenv()

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import HumanMessage, AIMessage
from typing import List

# ---------------------------------------
# Paths for persistence
# ---------------------------------------
VECTOR_STORE_PATH = "vector_store"
CHAT_SESSIONS_PATH = "chat_sessions.json"

# ---------------------------------------
# Session State Setup
# ---------------------------------------
if "docs" not in st.session_state:
    st.session_state.docs = []
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "chat_chain" not in st.session_state:
    st.session_state.chat_chain = None
if "chat_sessions" not in st.session_state:
    st.session_state.chat_sessions = {}
if "current_session" not in st.session_state:
    st.session_state.current_session = "Default"
if "chat_history" not in st.session_state:
    st.session_state.chat_history = StreamlitChatMessageHistory(key="chat_messages")

# ---------------------------------------
# LLM Initialization
# ---------------------------------------
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    st.error("GROQ_API_KEY not set. Please set it in your environment or .env.")
    st.stop()

def init_llm() -> ChatGroq:
    return ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model_name="llama-3.3-70b-versatile",
        temperature=0.0
    )

# ---------------------------------------
# Document Loading Functions
# ---------------------------------------
def load_documents_and_copy(folder_path: str, extensions: List[str]) -> List:
    """Copy files to 'uploads/' and load them."""
    docs = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if any(file.endswith(ext) for ext in extensions):
                os.makedirs("uploads", exist_ok=True)
                src, dst = os.path.join(root, file), os.path.join("uploads", file)
                shutil.copyfile(src, dst)
                loader = PyPDFLoader(dst) if file.endswith(".pdf") else TextLoader(dst)
                docs.extend(loader.load())
    return docs

def load_uploaded_files(uploaded_files) -> List:
    """Save and load uploaded files."""
    docs = []
    for f in uploaded_files:
        path = os.path.join("uploads", f.name)
        os.makedirs("uploads", exist_ok=True)
        with open(path, "wb") as file:
            file.write(f.read())
        loader = PyPDFLoader(path) if f.name.endswith(".pdf") else TextLoader(path)
        docs.extend(loader.load())
    return docs

# ---------------------------------------
# Vector Store Persistence
# ---------------------------------------
def build_vector_store(docs: List):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)
    embedding_fn = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.from_texts([doc.page_content for doc in chunks], embedding_fn)
    vector_store.save_local(VECTOR_STORE_PATH)
    return vector_store

def load_vector_store():
    if os.path.exists(VECTOR_STORE_PATH):
        embedding_fn = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        # Set allow_dangerous_deserialization=True to permit loading of pickle files.
        return FAISS.load_local(VECTOR_STORE_PATH, embedding_fn, allow_dangerous_deserialization=True)
    return None

# ---------------------------------------
# Chat Sessions Persistence (Multi-Session)
# ---------------------------------------
def save_chat_sessions():
    with open(CHAT_SESSIONS_PATH, "w") as f:
        json.dump(st.session_state.chat_sessions, f, indent=2)

def load_chat_sessions():
    if os.path.exists(CHAT_SESSIONS_PATH):
        with open(CHAT_SESSIONS_PATH, "r") as f:
            st.session_state.chat_sessions = json.load(f)

def save_current_session():
    messages = [
        {"role": "user" if isinstance(msg, HumanMessage) else "assistant", "content": msg.content}
        for msg in st.session_state.chat_history.messages
    ]
    st.session_state.chat_sessions[st.session_state.current_session] = messages
    save_chat_sessions()

def load_current_session():
    st.session_state.chat_history = StreamlitChatMessageHistory(key="chat_messages")
    messages = st.session_state.chat_sessions.get(st.session_state.current_session, [])
    for msg in messages:
        if msg["role"] == "user":
            st.session_state.chat_history.add_user_message(msg["content"])
        elif msg["role"] == "assistant":
            st.session_state.chat_history.add_ai_message(msg["content"])

# Load chat sessions on app start
load_chat_sessions()

# ---------------------------------------
# Conversational Chain Setup
# ---------------------------------------
def init_chat_chain(vector_store: FAISS) -> ConversationalRetrievalChain:
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        chat_memory=st.session_state.chat_history,
        return_messages=True
    )
    return ConversationalRetrievalChain.from_llm(
        llm=init_llm(),
        retriever=vector_store.as_retriever(),
        memory=memory,
    )

st.set_page_config(page_title="📚 Multi-Session RAG", layout="wide")
st.title("📚 Multi-Session Conversational RAG")

# ---------------------------
# Sidebar: Chat Session Management
# ---------------------------
with st.sidebar:
    st.header("💬 Chat Session Management")
    session_names = list(st.session_state.chat_sessions.keys())
    session_names.insert(0, "New Chat")
    selected_session = st.selectbox("Choose a session:", session_names, index=0)
    
    if selected_session == "New Chat":
        new_session_name = st.text_input("New Session Name", "")
        if st.button("Start New Chat") and new_session_name:
            st.session_state.current_session = new_session_name
            st.session_state.chat_sessions[new_session_name] = []
            st.session_state.chat_history = StreamlitChatMessageHistory(key="chat_messages")
            if st.session_state.vector_store:
                st.session_state.chat_chain = init_chat_chain(st.session_state.vector_store)
            save_chat_sessions()
            st.rerun()
    else:
        if selected_session != st.session_state.current_session:
            st.session_state.current_session = selected_session
            load_current_session()
    
    if st.button("Clear Chat History"):
        st.session_state.chat_history = StreamlitChatMessageHistory(key="chat_messages")
        st.session_state.chat_sessions[st.session_state.current_session] = []
        save_chat_sessions()
        st.rerun()
    
    if st.button("Delete Current Chat Session"):
        if st.session_state.current_session in st.session_state.chat_sessions:
            del st.session_state.chat_sessions[st.session_state.current_session]
            save_chat_sessions()
            st.session_state.current_session = "Default"
            st.session_state.chat_history = StreamlitChatMessageHistory(key="chat_messages")
            st.rerun()

# ---------------------------
# Sidebar: Document Loader
# ---------------------------
with st.sidebar:
    st.header("📂 Document Loader")
    method = st.radio("Upload method:", ("Upload Files", "Specify Folder"))
    new_docs = []
    if method == "Upload Files":
        uploaded_files = st.file_uploader(
            "Upload .txt, .md, or .pdf",
            type=["txt", "md", "pdf"],
            accept_multiple_files=True
        )
        if uploaded_files and st.button("Process Upload"):
            new_docs = load_uploaded_files(uploaded_files)
    else:
        folder = st.text_input("Folder path:")
        exts_str = st.text_input("File extensions (comma-separated):", ".txt,.md,.pdf")
        extensions = [ext.strip() for ext in exts_str.split(",")]
        if folder and st.button("Load Folder"):
            new_docs = load_documents_and_copy(folder, extensions)
    if new_docs:
        st.session_state.docs = new_docs
        st.session_state.vector_store = build_vector_store(new_docs)
        st.session_state.chat_chain = init_chat_chain(st.session_state.vector_store)
        save_current_session()
    if st.session_state.docs:
        with st.expander("Loaded Sources"):
            unique_filenames = {os.path.basename(doc.metadata.get("source", "Unknown")) for doc in st.session_state.docs}
            st.write("Loaded files:")
            for file in sorted(unique_filenames):
                st.markdown(f"- {file}")

# ---------------------------
# Main Chat Interface
# ---------------------------
if st.session_state.vector_store is None:
    st.session_state.vector_store = load_vector_store()
if st.session_state.vector_store and st.session_state.chat_chain is None:
    st.session_state.chat_chain = init_chat_chain(st.session_state.vector_store)

if st.session_state.vector_store:
    st.success("✅ Ready to chat!")
    user_input = st.text_input("Enter your question:")
    if user_input:
        with st.spinner("Fetching answer..."):
            response = st.session_state.chat_chain.invoke({"question": user_input})
            answer = response["answer"]
            st.markdown("### Answer")
            st.write(answer)
            save_current_session()
    if st.session_state.chat_history.messages:
        st.write("### Conversation so far:")
        for msg in st.session_state.chat_history.messages:
            role = "User" if isinstance(msg, HumanMessage) else "Assistant"
            st.write(f"**{role}:** {msg.content}")
else:
    st.info("Upload files or specify a folder to start.")
