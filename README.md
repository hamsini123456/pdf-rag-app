# PDF RAG Chatbot 📄🤖

An AI-powered chatbot that answers questions from any PDF using RAG (Retrieval Augmented Generation).

## 🚀 Live Demo
👉 https://pdf-rag-app-5ecgqxdj3ew3phmtjxngng.streamlit.app/

## What it does
- Upload any PDF document
- Ask questions about it in natural language  
- Get accurate answers powered by LLaMA 3.3 70B via Groq

## Tech Stack
- Python · LangChain · FAISS · HuggingFace Embeddings
- Groq API (LLaMA 3.3 70B) · Streamlit

## How RAG works
1. PDF is loaded and split into chunks
2. Chunks are converted to embeddings using HuggingFace
3. FAISS stores and searches relevant chunks
4. Relevant context is sent to LLaMA 3.3 via Groq
5. LLM generates accurate answer from your document

## How to run locally
1. Clone the repo
2. Install: `pip install -r requirements.txt`
3. Add your Groq API key in app.py
4. Run: `streamlit run app.py`
