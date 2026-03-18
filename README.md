# 📄 Multi-Document RAG Assistant

A conversational AI app that lets you upload multiple documents and chat with all of them at once — powered by LangChain, HuggingFace, and Streamlit.

---

## ✨ Features

- 📁 **Multi-document support** — Upload PDFs, TXTs, and DOCX files simultaneously
- 🧠 **RAG (Retrieval-Augmented Generation)** — Answers are grounded in your documents
- 💬 **Persistent chat history** — Maintains conversation context across multiple turns
- 🔍 **MMR-based retrieval** — Uses Maximal Marginal Relevance for diverse, relevant context chunks
- ⚡ **Powered by Llama 3.1 8B** — Fast, capable open-source LLM via HuggingFace Inference API
- 🖥️ **Clean Streamlit UI** — Simple sidebar upload + chat interface

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| Frontend | [Streamlit](https://streamlit.io/) |
| LLM | `meta-llama/Llama-3.1-8B-Instruct` via HuggingFace |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` |
| Vector Store | FAISS (in-memory) |
| RAG Framework | LangChain |
| Document Loaders | PyPDFLoader, TextLoader, Docx2txtLoader |

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/your-username/multi-doc-rag-assistant.git
cd multi-doc-rag-assistant
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Set up environment variables

Create a `.env` file in the root directory:

```env
HUGGINGFACEHUB_API_TOKEN=your_huggingface_token_here
```

> Get your free API token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)

### 4. Run the app

```bash
streamlit run app.py
```

---

## 📦 Requirements

```
streamlit
langchain
langchain-huggingface
langchain-community
langchain-text-splitters
faiss-cpu
python-dotenv
pypdf
docx2txt
sentence-transformers
```

---

## 🖼️ Usage

1. **Upload documents** using the sidebar — supports `.pdf`, `.txt`, and `.docx`
2. Click **"Process Documents"** to index them into the vector store
3. **Ask questions** in the chat input at the bottom
4. The assistant retrieves the most relevant chunks and generates a grounded answer

---

## 📁 Project Structure

```
multi-doc-rag-assistant/
│
├── app.py              # Main Streamlit application
├── .env                # API keys (not committed)
├── .gitignore
├── requirements.txt
└── README.md
```

---

## ⚙️ Configuration

You can tune the following parameters in `app.py` to adjust retrieval behaviour:

| Parameter | Default | Description |
|---|---|---|
| `chunk_size` | `500` | Size of each document chunk |
| `chunk_overlap` | `50` | Overlap between consecutive chunks |
| `k` | `6` | Number of chunks retrieved per query |
| `search_type` | `mmr` | Retrieval strategy (MMR for diversity) |
| `lambda_mult` | `0.5` | MMR diversity vs. relevance trade-off |
| `max_new_tokens` | `2098` | Max tokens in LLM response |

---

## 📝 Notes

- The vector store is **session-based** — it resets when the app is reloaded
- Re-upload and re-process documents after each refresh
- For large documents, increasing `chunk_size` may improve coherence

---

## 🙌 Acknowledgements

- [LangChain](https://github.com/langchain-ai/langchain) for the RAG framework
- [HuggingFace](https://huggingface.co/) for model hosting and embeddings
- [Streamlit](https://streamlit.io/) for the UI framework
- [Meta AI](https://ai.meta.com/) for the Llama 3.1 model
