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
    """Initialize the ChatGroq LLM."""
    return ChatGroq(
        groq_api_key=GROQ_API_KEY, model_name="llama-3.3-70b-versatile", temperature=0.0
    )


# ---------------------------------------
# Document Loading Functions
# ---------------------------------------
def load_documents_and_copy(folder_path: str, extensions: List[str]) -> List:
    """
    Copy files from 'folder_path' to 'uploads/' and load them.
    We also set doc.metadata["source"] = path so we can display filenames later.
    """
    docs = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if any(file.endswith(ext) for ext in extensions):
                os.makedirs("uploads", exist_ok=True)
                src = os.path.join(root, file)
                dst = os.path.join("uploads", file)
                shutil.copyfile(src, dst)

                if file.endswith(".pdf"):
                    loader = PyPDFLoader(dst)
                else:
                    loader = TextLoader(dst)

                loaded_docs = loader.load()
                for d in loaded_docs:
                    d.metadata["source"] = dst  # Store path in metadata
                docs.extend(loaded_docs)
    return docs


def load_uploaded_files(uploaded_files) -> List:
    """
    Save uploaded files to 'uploads/' and load them.
    Also set doc.metadata["source"] = path for each doc.
    """
    docs = []
    for f in uploaded_files:
        path = os.path.join("uploads", f.name)
        os.makedirs("uploads", exist_ok=True)
        with open(path, "wb") as file:
            file.write(f.read())

        if f.name.endswith(".pdf"):
            loader = PyPDFLoader(path)
        else:
            loader = TextLoader(path)

        loaded_docs = loader.load()
        for d in loaded_docs:
            d.metadata["source"] = path
        docs.extend(loaded_docs)
    return docs


# ---------------------------------------
# Vector Store Persistence
# ---------------------------------------
def build_vector_store(docs: List):
    """
    Split docs into chunks, embed them, and build a FAISS index. Then save locally.
    We also pass chunk metadata to preserve the 'source' info.
    """
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)

    embedding_fn = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vector_store = FAISS.from_texts(
        [chunk.page_content for chunk in chunks],
        embedding_fn,
        metadatas=[chunk.metadata for chunk in chunks],
    )
    vector_store.save_local(VECTOR_STORE_PATH)
    return vector_store


def load_vector_store():
    """
    Load the FAISS index if it exists. We allow 'dangerous' deserialization
    because FAISS uses pickle under the hood.
    """
    if os.path.exists(VECTOR_STORE_PATH):
        try:
            embedding_fn = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
            return FAISS.load_local(
                VECTOR_STORE_PATH, embedding_fn, allow_dangerous_deserialization=True
            )
        except (FileNotFoundError, RuntimeError):
            # The index folder/file might be incomplete or corrupted
            return None
    return None


# ---------------------------------------
# Chat Sessions Persistence (Multi-Session)
# ---------------------------------------
def save_chat_sessions():
    """Save all chat sessions to disk as JSON."""
    with open(CHAT_SESSIONS_PATH, "w") as f:
        json.dump(st.session_state.chat_sessions, f, indent=2)


def load_chat_sessions():
    """Load chat sessions from JSON if it exists."""
    if os.path.exists(CHAT_SESSIONS_PATH):
        with open(CHAT_SESSIONS_PATH, "r") as f:
            st.session_state.chat_sessions = json.load(f)


def save_current_session():
    """
    Store the current conversation in st.session_state.chat_sessions
    under the current session name, then save to disk.
    """
    messages = []
    for msg in st.session_state.chat_history.messages:
        if isinstance(msg, HumanMessage):
            role = "user"
        elif isinstance(msg, AIMessage):
            role = "assistant"
        else:
            role = "system"
        messages.append({"role": role, "content": msg.content})

    st.session_state.chat_sessions[st.session_state.current_session] = messages
    save_chat_sessions()


def load_current_session():
    """
    Load messages for the current session from chat_sessions and
    populate st.session_state.chat_history.
    """
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
    """
    Build a ConversationalRetrievalChain with:
      - ChatGroq LLM
      - FAISS retriever
      - ConversationBufferMemory with explicit input_key/output_key
      - chain_type='stuff' (preserves source docs)
      - return_source_documents=True (so we can display them)
      - output_key='answer' (the main output for memory)
    """
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        chat_memory=st.session_state.chat_history,
        return_messages=True,
        input_key="question",
        output_key="answer",
    )
    chain = ConversationalRetrievalChain.from_llm(
        llm=init_llm(),
        retriever=vector_store.as_retriever(),
        memory=memory,
        chain_type="stuff",
        return_source_documents=True,
        output_key="answer",
    )
    return chain


# ---------------------------------------
# Streamlit App Layout
# ---------------------------------------
st.set_page_config(page_title="📚 Multi-Session RAG", layout="wide")
st.title("📚 Multi-Session Conversational RAG")

# ---------------------------------------
# Sidebar: Chat Session Management
# ---------------------------------------
with st.sidebar:
    st.header("💬 Chat Session Management")
    session_names = list(st.session_state.chat_sessions.keys())
    session_names.insert(0, "New Chat")
    selected_session = st.selectbox("Choose a session:", session_names, index=0)

    # Start a new chat session
    if selected_session == "New Chat":
        new_session_name = st.text_input("New Session Name", "")
        if st.button("Start New Chat") and new_session_name:
            st.session_state.current_session = new_session_name
            st.session_state.chat_sessions[new_session_name] = []
            st.session_state.chat_history = StreamlitChatMessageHistory(
                key="chat_messages"
            )
            if st.session_state.vector_store:
                st.session_state.chat_chain = init_chat_chain(
                    st.session_state.vector_store
                )
            save_chat_sessions()
            st.rerun()
    else:
        # Load an existing session
        if selected_session != st.session_state.current_session:
            st.session_state.current_session = selected_session
            load_current_session()

    # Clear current session's history
    if st.button("Clear Chat History"):
        st.session_state.chat_history = StreamlitChatMessageHistory(key="chat_messages")
        st.session_state.chat_sessions[st.session_state.current_session] = []
        save_chat_sessions()
        st.rerun()

    # Delete current session
    if st.button("Delete Current Chat Session"):
        if st.session_state.current_session in st.session_state.chat_sessions:
            del st.session_state.chat_sessions[st.session_state.current_session]
            save_chat_sessions()
            st.session_state.current_session = "Default"
            st.session_state.chat_history = StreamlitChatMessageHistory(
                key="chat_messages"
            )
            st.rerun()

# ---------------------------------------
# Sidebar: Document Loader
# ---------------------------------------
with st.sidebar:
    st.header("📂 Document Loader")
    method = st.radio("Upload method:", ("Upload Files", "Specify Folder"))
    new_docs = []

    if method == "Upload Files":
        uploaded_files = st.file_uploader(
            "Upload .txt, .md, or .pdf",
            type=["txt", "md", "pdf"],
            accept_multiple_files=True,
        )
        if uploaded_files and st.button("Process Upload"):
            new_docs = load_uploaded_files(uploaded_files)
    else:
        folder = st.text_input("Folder path:")
        exts_str = st.text_input("File extensions (comma-separated):", ".txt,.md,.pdf")
        extensions = [ext.strip() for ext in exts_str.split(",")]
        if folder and st.button("Load Folder"):
            new_docs = load_documents_and_copy(folder, extensions)

    # If new docs were loaded, build a new vector store and chain
    if new_docs:
        st.session_state.docs = new_docs
        st.session_state.vector_store = build_vector_store(new_docs)
        st.session_state.chat_chain = init_chat_chain(st.session_state.vector_store)
        save_current_session()

    # Show the list of loaded files
    if st.session_state.docs:
        with st.expander("Loaded Sources"):
            unique_filenames = {
                os.path.basename(doc.metadata.get("source", "Unknown"))
                for doc in st.session_state.docs
            }
            st.write("Loaded files:")
            for file in sorted(unique_filenames):
                st.markdown(f"- {file}")

# ---------------------------------------
# Main Chat Interface
# ---------------------------------------
# Only attempt to load an existing index if we haven't already
if st.session_state.vector_store is None:
    st.session_state.vector_store = load_vector_store()

# If we have a vector store, but no chain, initialize it
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

            # Save the updated conversation
            save_current_session()

            # Show source docs if they exist
            if "source_documents" in response and response["source_documents"]:
                st.markdown("### Source Snippets:")
                for i, doc in enumerate(response["source_documents"]):
                    snippet = doc.page_content[:500]
                    source_name = os.path.basename(
                        doc.metadata.get("source", "Unknown")
                    )
                    st.markdown(f"**Source {i+1} ({source_name}):**")
                    st.write(f"> {snippet}...")
else:
    st.info("No vector store yet. Upload documents or specify a folder to create one.")
