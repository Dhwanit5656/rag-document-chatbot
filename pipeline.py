import os
import shutil
from langchain_huggingface import ChatHuggingFace,HuggingFaceEmbeddings,HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate,ChatPromptTemplate,MessagesPlaceholder
from langchain_community.document_loaders import PyPDFLoader,Docx2txtLoader,TextLoader
from langchain_chroma import Chroma
from langchain_core.runnables import RunnableParallel,RunnablePassthrough,RunnableLambda
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from langdetect import detect
from langdetect import DetectorFactory
from dotenv import load_dotenv
load_dotenv()
from huggingface_hub import login
login(token=os.getenv("HUGGINGFACEHUB_API_TOKEN"))
DetectorFactory.seed=0

def detect_language(text: str) -> str:
    if len(text.strip())<20:
        return 'en'
    try:
        lang = detect(text)
        if lang=='hi':
            return 'hi'
        elif lang=='en':
            return 'en'
        else:
            return 'en'
    except:
            return 'en'

def load_document_chunk(filepaths: list) -> list:

    print('Loading the documents....')
    all_documents = []
    for filepath in filepaths:

        ext = os.path.splitext(filepath)[1].lower()
        if ext =='.pdf':
            loader = PyPDFLoader(filepath)
        elif ext=='.txt':
            loader = TextLoader(filepath, encoding='utf-8')
        elif ext == '.docx':
            loader = Docx2txtLoader(filepath)
        else:
            print("Give a supported file")
            continue
        try:
            docs = loader.load()
            all_documents.extend(docs)
            print("Documents loaded successfully")
        except Exception as e:
            print(f"❌ Failed to load {filepath}: {e}")
            continue
    print(f"\n📄 Total documents loaded: {len(all_documents)}")
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size = 500,
        chunk_overlap = 50
    )

    chunks = splitter.split_documents(all_documents)
    chunks = [
    chunk for chunk in chunks
    if chunk.page_content and len(chunk.page_content.strip()) > 20
    ]

    print(f"📦 Valid chunks after filtering: {len(chunks)}")

    for chunk in chunks:
        full_path = chunk.metadata.get('source','unknown')
        filename = os.path.basename(full_path)
        chunk.metadata['source'] = filename
        lang = detect_language(chunk.page_content)
        chunk.metadata['language'] = lang

    return chunks

embedding = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )
db = None
def vector_store(chunks: list):
    
    global db
    db = Chroma.from_documents(
            embedding=embedding,
            collection_name='multilingual-multidoc-rag',
            documents=chunks
        )
   
    print(f'stored {len(chunks)} in the ChromaDB')

def search_query(query: str, k: int=6, fetch_k: int=15) ->tuple:

    global db
    
    retriver = db.as_retriever(
        search_type = 'mmr',
        search_kwargs = {
            'k':k, 'fetch_k':fetch_k, 'lambda_mult':0.5
        }
    )

    docs = retriver.invoke(query)

    context_text = [doc.page_content for doc in docs]
    metadatas = [doc.metadata for doc in docs]

    return context_text,metadatas

print("🔄 Connecting to meta-Llama-3.1-8B")

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",
    max_new_tokens=2048
)

model = ChatHuggingFace(llm=llm)

def get_answer(question: str, context_text: list, language: str) -> str:

    #joining the contexts together
    context_str = '\n\n'.join(context_text)

    if language=='hi':
        lang_instruction = "कृपया हिंदी में उत्तर दें।"
    else:
        lang_instruction = "Please respond in English."
    
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are a helpful multilingual assistant.\n"
            "Answer ONLY based on the context provided.\n"
            "If answer not found say: "
            "'I could not find this in the documents.'\n\n"
            f"{lang_instruction}\n\n"
            "Context:\n{context}"
        ),
        (
            'human',
            "{question}"
        )
    ])
    
    parser = StrOutputParser()
    parallel_Chain = RunnableParallel(
        {
            'context': RunnablePassthrough() | (lambda x: context_str),
            'question': RunnablePassthrough()
        }
    )
    chain = parallel_Chain | prompt | model | parser

    return chain.invoke(question)

