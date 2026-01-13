import streamlit as st
import os
import tempfile
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# ────────────────────────────────────────────────
# CONFIG
# ────────────────────────────────────────────────

DEFAULT_PDF = "Small_Business_Template.pdf"
CHROMA_PATH = "./chroma_db"  # we'll use a session-specific collection
MODEL_NAME = "llama3.1:8b-instruct-q4_0"

st.set_page_config(page_title="Business Policy Chatbot", layout="wide")

# ────────────────────────────────────────────────
# SIDEBAR – Upload + Info + Clear
# ────────────────────────────────────────────────

with st.sidebar:
    st.header("Business Policy Chatbot")

    # File uploader
    uploaded_file = st.file_uploader(
        "Upload your own company policy PDF or anything else really",
        type=["pdf"],
        help="Upload a policy/handbook PDF to ask questions about it. Max ~20 MB recommended."
    )

    st.markdown("""
    **How it works**  
    • Upload a PDF → it becomes the knowledge base  
    • No upload → uses default sample document  
    • All answers come **only** from the loaded document

    **Tech stack**  
    • LangChain + Chroma (RAG)  
    • Ollama (Llama 3.1 8B Instruct)  
    • Sentence Transformers embeddings  
    • Streamlit UI
    """)

    st.divider()

    if st.button("Clear Chat History", type="primary", use_container_width=True):
        st.session_state.messages = []
        if "vectorstore" in st.session_state:
            del st.session_state.vectorstore  # force reload on next question
        st.success("Chat history cleared!")
        st.rerun()

# ────────────────────────────────────────────────
# VECTOR STORE (per-session, cached)
# ────────────────────────────────────────────────

@st.cache_resource
def get_vectorstore(pdf_path):
    if not os.path.exists(CHROMA_PATH):
        os.makedirs(CHROMA_PATH)

    with st.spinner("Indexing document... (first time only)"):
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800, chunk_overlap=120
        )
        chunks = text_splitter.split_documents(documents)

        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=CHROMA_PATH,
            collection_name="session_collection"  # unique per session if needed
        )

    return vectorstore.as_retriever(search_kwargs={"k": 4})

# Decide which PDF to use
if uploaded_file is not None:
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name
    st.session_state.pdf_path = tmp_file_path
    st.sidebar.success("Custom PDF uploaded! Using it as knowledge base.")
else:
    st.session_state.pdf_path = DEFAULT_PDF
    st.sidebar.info("Using default sample policy document.")

retriever = get_vectorstore(st.session_state.pdf_path)

# ────────────────────────────────────────────────
# LLM & RAG Chain
# ────────────────────────────────────────────────

llm = OllamaLLM(
    model=MODEL_NAME,
    temperature=0.35,
    num_ctx=8192
)

template = """You are a professional HR and business policy assistant.
Answer ONLY using the provided context from the uploaded company documents.
Be concise, accurate, and use bullet points or numbered lists when helpful.
If the information is not in the context, say exactly:  
"I don't have sufficient information from the available documents."

Context:
{context}

Question: {question}

Answer:"""

prompt = ChatPromptTemplate.from_template(template)

def format_docs(docs):
    return "\n\n".join(doc.page_content.strip() for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# ────────────────────────────────────────────────
# MAIN CHAT UI
# ────────────────────────────────────────────────

st.title("Business Policy Chatbot")
st.markdown("Upload your own policy PDF or ask about the default sample document.")

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input
if prompt := st.chat_input("Ask about the company policy..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = rag_chain.invoke(prompt)
        st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})

# Cleanup temp file on session end (optional - Streamlit doesn't have direct hook, but good practice)
if "pdf_path" in st.session_state and uploaded_file is None:
    try:
        os.unlink(st.session_state.pdf_path)
    except:
        pass
