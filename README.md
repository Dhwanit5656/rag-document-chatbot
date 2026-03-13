# 🤖 Document RAG Assistant

An interactive document-based Q&A assistant leveraging **Retrieval-Augmented Generation (RAG)**. [cite_start]This application uses **LangChain** to orchestrate the workflow, **Hugging Face** for embeddings and language models, and **FAISS** for efficient similarity search.

### 🌟 Features
* [cite_start]**Multi-format Support**: Upload and process **PDF**, **TXT**, and **DOCX** files.
* [cite_start]**Persistent Vector Storage**: Local saving of document embeddings using **FAISS** to avoid re-processing existing files.
* [cite_start]**Advanced Retrieval**: Utilizes **Maximal Marginal Relevance (MMR)** to ensure diverse and relevant information retrieval.
* [cite_start]**Streamlined UI**: Built with **Streamlit** for a clean, user-friendly chat interface.

### 🛠️ Tech Stack
* [cite_start]**Frontend**: Streamlit (v1.55.0) 
* [cite_start]**Orchestration**: LangChain (Core, Community, and Hugging Face integrations) 
* [cite_start]**Embeddings**: `sentence-transformers/all-MiniLM-L6-v2` 
* [cite_start]**LLM**: `meta-llama/Llama-3.1-8B-Instruct` (via Hugging Face Endpoint) 
* [cite_start]**Vector Database**: FAISS (CPU) 

### 🚀 Getting Started

#### 1. Prerequisites
[cite_start]Ensure you have Python installed and a **Hugging Face API Token** ready[cite: 1, 2].

#### 2. Installation
Clone the repository and install the required dependencies:
```bash
pip install -r requirements.txt
