# Business Policy Chatbot

An interactive **RAG (Retrieval-Augmented Generation)** chatbot that lets users upload company policy/handbook PDFs and ask natural-language questions about leave, benefits, deductions, eligibility, and more.

All answers are generated **strictly from the uploaded document** — no hallucinations outside the source content.

**Live Demo** (Render):  
https://business-policy-chatbot.onrender.com

## Features

- Upload your own PDF policy document (or use the default sample)
- Ask questions in natural language (e.g. "What is the paid holiday entitlement?")
- Semantic search over document chunks for accurate retrieval
- Answers powered by local LLM (Ollama + Llama 3.1 8B Instruct)
- Clean, responsive chat interface with history & clear button
- Built for privacy — runs fully local or on your server

## Tech Stack

- **Frontend/UI**: Streamlit
- **RAG Pipeline**: LangChain
- **Vector Store**: ChromaDB (local embeddings)
- **Embeddings**: sentence-transformers (all-MiniLM-L6-v2)
- **LLM**: Ollama (Llama 3.1 8B Instruct or 3B for lighter use)
- **PDF Parsing**: PyPDFLoader
- **Deployment**: Render.com (Docker)

## Quick Start (Local Development)

1. Clone the repo:
   ```bash
   git clone https://github.com/tekena-manuel/business-policy-chatbot.git
   cd business-policy-chatbot

2. create/activate virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate   # Mac/Linux
   # or on Windows: venv\Scripts\activate
3. install dependencies:
   ```bash
   pip install -r requirements.txt
4. install Ollama
- download and run in background
- pull model: llama pull llama3.1:8b-instruct (or llama3.2:3b-instruct
5. Run the app(after building of course)
  ```bash
  streamlit run app.py
  
