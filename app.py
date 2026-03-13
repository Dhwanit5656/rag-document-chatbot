import streamlit as st
import os
from dotenv import load_dotenv

from langchain_huggingface import ChatHuggingFace, HuggingFaceEmbeddings, HuggingFaceEndpoint
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.messages import HumanMessage, AIMessage

load_dotenv()

st.set_page_config(page_title="Multi-Doc RAG Assistant", page_icon="🤖")

# Initialize Session States
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "vector_db" not in st.session_state:
    st.session_state.vector_db = None

st.title("📄 Multi-Document GPT")
st.write("Upload multiple files and chat with all of them at once.")

# -----------------------------
# Document Processing
# -----------------------------

def process_docs(uploaded_files):
    all_chunks = []
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    
    for uploaded_file in uploaded_files:
        filepath = os.path.join("temp", uploaded_file.name)
        os.makedirs("temp", exist_ok=True)
        with open(filepath, "wb") as f:
            f.write(uploaded_file.getbuffer())

        ext = os.path.splitext(filepath)[1]
        if ext == ".pdf": 
            loader = PyPDFLoader(filepath)
        elif ext == ".txt": 
            # ADD encoding="utf-8" HERE
            loader = TextLoader(filepath, encoding="utf-8")
        elif ext == ".docx": 
            loader = Docx2txtLoader(filepath)
        else: 
            continue

        docs = loader.load()
        all_chunks.extend(splitter.split_documents(docs))
    
    return all_chunks

def get_vectorstore(chunks):
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    # In-memory FAISS for session-based multi-doc support
    return FAISS.from_documents(chunks, embedding_model)

# -----------------------------
# LLM & Chain Setup
# -----------------------------

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",
    max_new_tokens=1024
)
model = ChatHuggingFace(llm=llm)

# GPT-style prompt with Memory placeholder
contextual_prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer based on context. If unknown, say 'Not found'.\n\nContext:\n{context}"),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question}"),
])

# -----------------------------
# Sidebar Upload
# -----------------------------

with st.sidebar:
    st.header("Upload Center")
    uploaded_files = st.file_uploader(
        "Upload Documents", 
        type=["pdf", "txt", "docx"], 
        accept_multiple_files=True
    )
    
    if st.button("Process Documents"):
        with st.spinner("Indexing documents..."):
            chunks = process_docs(uploaded_files)
            st.session_state.vector_db = get_vectorstore(chunks)
            st.success("Indexing complete!")

# -----------------------------
# Chat Interface
# -----------------------------

# Display chat history
for message in st.session_state.chat_history:
    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.markdown(message.content)
    else:
        with st.chat_message("assistant"):
            st.markdown(message.content)

# Handle Input
if question := st.chat_input("Ask about your documents..."):
    if not st.session_state.vector_db:
        st.error("Please upload and process documents first!")
    else:
        # 1. Add user message to history
        st.session_state.chat_history.append(HumanMessage(content=question))
        
        # 2. FIX: Extract history to a local variable to avoid the AttributeError
        # We pass this local 'history_to_pass' into the chain
        history_to_pass = st.session_state.chat_history[:-1]

        with st.chat_message("user"):
            st.markdown(question)

        # 3. RAG Chain
        retriever = st.session_state.vector_db.as_retriever(search_kwargs={"k": 3})
        
        chain = (
            RunnableParallel({
                "context": retriever | (lambda docs: "\n\n".join(d.page_content for d in docs)),
                "question": RunnablePassthrough(),
                "chat_history": lambda x: history_to_pass  # Use the local variable here
            })
            | contextual_prompt 
            | model 
            | StrOutputParser()
        )

        # 4. Generate Response
        with st.chat_message("assistant"):
            with st.spinner("Searching..."):
                response = chain.invoke(question)
                st.markdown(response)
        

        st.session_state.chat_history.append(AIMessage(content=response))

