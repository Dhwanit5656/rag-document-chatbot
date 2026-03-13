# 🤖 Document RAG Assistant

An interactive document-based Q&A assistant leveraging **Retrieval-Augmented Generation (RAG)**. [cite_start]This application uses **LangChain** to orchestrate the workflow, **Hugging Face** for embeddings and language models, and **FAISS** for efficient similarity search.

### 🌟 Features
* **Multi-format Support**: Upload and process **PDF**, **TXT**, and **DOCX** files.
* **Persistent Vector Storage**: Local saving of document embeddings using **FAISS** to avoid re-processing existing files.
* **Advanced Retrieval**: Utilizes **Maximal Marginal Relevance (MMR)** to ensure diverse and relevant information retrieval.
* **Streamlined UI**: Built with **Streamlit** for a clean, user-friendly chat interface.

### 🛠️ Tech Stack
* **Frontend**: Streamlit (v1.55.0) 
* **Orchestration**: LangChain (Core, Community, and Hugging Face integrations) 
* **Embeddings**: `sentence-transformers/all-MiniLM-L6-v2` 
* **LLM**: `meta-llama/Llama-3.1-8B-Instruct` (via Hugging Face Endpoint) 
* **Vector Database**: FAISS (CPU) 




