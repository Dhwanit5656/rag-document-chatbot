import streamlit as st
import os
from dotenv import load_dotenv

from langchain_huggingface import ChatHuggingFace, HuggingFaceEmbeddings, HuggingFaceEndpoint
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnableLambda, RunnablePassthrough

load_dotenv()

st.set_page_config(page_title="Document RAG Assistant", page_icon="🤖")

st.title("📄 Chat with your Document")
st.write("Upload a PDF, TXT, or DOCX and ask questions.")

# -----------------------------
# Document Loader
# -----------------------------

def load_doc(filepath):

    ext = os.path.splitext(filepath)[1]

    if ext == ".pdf":
        loader = PyPDFLoader(filepath)

    elif ext == ".txt":
        loader = TextLoader(filepath)

    elif ext == ".docx":
        loader = Docx2txtLoader(filepath)

    else:
        st.error("Unsupported file type")
        return None

    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    chunks = splitter.split_documents(docs)

    return chunks


# -----------------------------
# Vector Store
# -----------------------------

def vector_store(chunks, filename):

    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    db_path = f"vector_dbs/{filename}"

    if os.path.exists(db_path):
        vector_db = FAISS.load_local(
            db_path,
            embedding_model,
            allow_dangerous_deserialization=True
        )
    else:
        vector_db = FAISS.from_documents(chunks, embedding_model)
        vector_db.save_local(db_path)

    return vector_db

# -----------------------------
# LLM
# -----------------------------

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",
    max_new_tokens=2048
)

model = ChatHuggingFace(llm=llm)

prompt = PromptTemplate(
    template="""
            You are a helpful assistant.

            Answer the question based only on the provided context.
            If the answer is not present in the context say "Not found in the document".
            Nicely wrap the Answer according to the information.

            Context:
            {context}

            Question:
            {question}
""",
    input_variables=["context", "question"]
)

parser = StrOutputParser()


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


# -----------------------------
# Upload Section
# -----------------------------

uploaded_file = st.file_uploader(
    "Upload Document",
    type=["pdf", "txt", "docx"]
)

if uploaded_file:

    filepath = os.path.join("temp", uploaded_file.name)

    os.makedirs("temp", exist_ok=True)

    with open(filepath, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.success("Document uploaded successfully!")

    chunks = load_doc(filepath)

    filename = uploaded_file.name.replace(" ", "_")

    vector_db = vector_store(chunks,filename)

    retriever = vector_db.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 3, "lambda_mult": 0.5}
    )

    parallel_chain = RunnableParallel({
        "context": retriever | RunnableLambda(format_docs),
        "question": RunnablePassthrough()
    })

    chains = parallel_chain | prompt | model | parser

    # -----------------------------
    # Chat Section
    # -----------------------------

    question = st.chat_input("Ask something about the document")

    if question:

        with st.spinner("Thinking..."):

            response = chains.invoke(question)

        st.chat_message("user").write(question)
        st.chat_message("assistant").write(response)