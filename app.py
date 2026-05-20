import os
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

GROQ_API_KEY = st.secrets["GROQ_API_KEY"]

st.title("Ask my PDF 📄")

uploaded = st.file_uploader("Upload a PDF", type="pdf")

if uploaded:
    with open("temp.pdf", "wb") as f:
        f.write(uploaded.read())

    loader = PyPDFLoader("temp.pdf")
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2")
    db = FAISS.from_documents(chunks, embeddings)

    llm = ChatGroq(api_key=GROQ_API_KEY,
        model_name="llama-3.3-70b-versatile")

    question = st.text_input("Ask a question about your PDF:")
    if question:
        with st.spinner("Thinking..."):
            docs = db.similarity_search(question, k=3)
            context = "\n".join([d.page_content for d in docs])
            prompt = ChatPromptTemplate.from_template(
                "Answer using this context:\n{context}\n\nQuestion: {question}")
            chain = prompt | llm | StrOutputParser()
            answer = chain.invoke({"context": context, "question": question})
        st.success(answer)
