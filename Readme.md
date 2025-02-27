# Multi-Session Conversational RAG Chat App

This repository contains a Streamlit application that demonstrates a multi-session, conversational Retrieval-Augmented Generation (RAG) chat interface using LangChain, FAISS, and ChatGroq. The app lets you load documents (via file uploads or by specifying a folder), build a FAISS index for document retrieval, and have multi-turn conversations with memory across different chat sessions.

## Features

- **Multi-Session Chat Management:**  
  Create, switch, clear, or delete chat sessions. Your conversation history is persisted across app restarts.

- **Document Loader:**  
  Upload files (TXT, MD, PDF) or specify a folder. Files are copied to an `uploads/` directory, processed, and indexed for retrieval.

- **Conversational Retrieval:**  
  Uses ChatGroq (currently set to the `llama-3.3-70b-versatile` model) with memory to provide context-aware, multi-turn conversational responses.

- **Persistence:**  
  The FAISS index and chat sessions are saved to disk so that your knowledge base and conversation history persist across restarts.

## Prerequisites

- **Python 3.9**  
- [Conda](https://docs.conda.io/en/latest/) or another virtual environment manager

## Setup

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/dobharweim/blazing-rag.git
   cd your-repo
   ```

2. **Create the Environment:**

   This project uses an `environment.yml` file for dependency management. Create and update your Conda environment by running:

   ```bash
   conda env update -f environment.yml --prune
   conda activate rag_env
   ```

3. **Set Up Your Environment Variables:**

   Create a `.env` file in the root directory of the project with the following content (replace `XXXXXXXXXXXXXX` with your actual API key):

   ```bash
   GROQ_API_KEY=XXXXXXXXXXXXXX
   ```

4. **Run the Application:**

   Start the app with:

   ```bash
   streamlit run app.py --server.fileWatcherType none
   ```

   You can now view the app in your browser at the provided local URL.

## Usage

### Document Loader

- **Upload Files:**  
  Use the file uploader in the sidebar to upload `.txt`, `.md`, or `.pdf` files. The files will be saved to an `uploads/` directory and processed for document indexing.
  
- **Specify Folder:**  
  Provide a folder path and file extensions (comma-separated) to load documents directly from your local system. The app copies these files to the `uploads/` directory before processing.

### Chat Session Management

- **Start a New Chat Session:**  
  Select "New Chat" from the session dropdown in the sidebar, enter a new session name, and click "Start New Chat". This creates a fresh conversation with cleared history.

- **Clear Chat History:**  
  Use the "Clear Chat History" button to clear only the current session's conversation history.

- **Delete Current Chat Session:**  
  Use the "Delete Current Chat Session" button to completely remove the current chat session from the saved sessions.

### Chat Interface

- Enter your questions in the main chat box.
- The app uses a Conversational Retrieval Chain (with memory) to provide context-aware answers.
- Your conversation is displayed below the input box.

## Model Considerations

This app currently uses the `llama-3.3-70b-versatile` model via the ChatGroq API. This model is chosen for its versatility and high-quality responses for a range of queries. Depending on your needs (latency, cost, or domain-specific requirements), you might explore alternative models.

## Security Note

When loading the FAISS index from disk, the code enables dangerous deserialization by setting `allow_dangerous_deserialization=True`. This is safe for files you create and control. **Do not load FAISS indexes from untrusted sources.**

## Contributing

Feel free to fork this repository and submit pull requests. If you have suggestions or encounter issues, please open an issue.

## License

[MIT License](https://opensource.org/license/mit)
