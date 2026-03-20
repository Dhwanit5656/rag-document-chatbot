# 🗣️ Multilingual-MultiDoc RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot that lets you upload documents and ask questions in **Hindi or English**. Built with LangChain, Streamlit, ChromaDB, and a Llama 3.1 backend via HuggingFace.

---

## ✨ Features

- 📄 **Multi-document support** — Upload PDF, TXT, and DOCX files
- 🌐 **Bilingual** — Ask questions and receive answers in Hindi or English (auto-detected)
- 🔍 **MMR Retrieval** — Uses Maximal Marginal Relevance for diverse, relevant context chunks
- 🧠 **LLM-powered answers** — Powered by `meta-llama/Llama-3.1-8B-Instruct` via HuggingFace
- 🗃️ **In-memory vector store** — ChromaDB for fast semantic search
- 💬 **Chat history** — Persisted within the session with source attribution per response

---

## 🗂️ Project Structure

```
├── main.py        # Streamlit UI — handles file upload, chat interface, session state
└── pipeline.py    # Backend — document loading, chunking, embedding, retrieval, LLM inference
```

---

## ⚙️ How It Works

1. **Upload** your documents via the sidebar (PDF, TXT, or DOCX)
2. **Process** — documents are chunked, language-tagged, and embedded into ChromaDB
3. **Ask** — type a question in Hindi or English; the language is auto-detected
4. **Answer** — relevant chunks are retrieved via MMR and passed to Llama 3.1 for a grounded response
5. **Sources** — each answer shows which document(s) and page(s) it was drawn from

---

## 🚀 Setup & Installation

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd <repo-folder>
```

### 2. Install dependencies

```bash
pip install streamlit langchain langchain-huggingface langchain-community \
            langchain-chroma langchain-text-splitters langdetect \
            python-dotenv huggingface_hub chromadb pypdf docx2txt
```

### 3. Set up environment variables

Create a `.env` file in the project root:

```env
HUGGINGFACEHUB_API_TOKEN=your_token_here
```

> Get your token from [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens). You need access to `meta-llama/Llama-3.1-8B-Instruct` — request it [here](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct).

### 4. Run the app

```bash
streamlit run main.py
```

---

## 🧩 Tech Stack

| Component | Library / Model |
|---|---|
| UI | Streamlit |
| LLM | `meta-llama/Llama-3.1-8B-Instruct` (HuggingFace) |
| Embeddings | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` |
| Vector Store | ChromaDB |
| RAG Framework | LangChain |
| Language Detection | `langdetect` |
| Document Loaders | PyPDF, Docx2txt, TextLoader |

---

## 📋 Usage Notes

- The vector store is **in-memory** — it resets each time you click **Reset** or restart the app
- Language detection requires at least ~20 characters to work reliably; shorter inputs default to English
- The LLM is instructed to answer **only from the provided context** — it will say so if the answer isn't found in the documents
- Temporary files uploaded through the UI are deleted automatically after processing

---

## 🛠️ Configuration

Key parameters you can tune in `pipeline.py`:

| Parameter | Location | Default | Description |
|---|---|---|---|
| `chunk_size` | `load_document_chunk` | `500` | Characters per chunk |
| `chunk_overlap` | `load_document_chunk` | `50` | Overlap between chunks |
| `k` | `search_query` | `6` | Number of chunks returned |
| `fetch_k` | `search_query` | `15` | Candidate pool for MMR |
| `lambda_mult` | `search_query` | `0.5` | MMR diversity vs. relevance balance |
| `max_new_tokens` | LLM init | `2048` | Max tokens in LLM response |

---

## ⚠️ Known Limitations

- The vector store does not persist across app restarts (in-memory only)
- Very short queries (<20 chars) are assumed to be English for language detection
- Requires a HuggingFace account with approved access to Llama 3.1

---

## 📄 License

MIT — feel free to use and adapt.
