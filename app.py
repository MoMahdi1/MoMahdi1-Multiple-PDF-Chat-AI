import streamlit as st
from dotenv import load_dotenv
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFaceEndpoint

import os

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY_1"))

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        separators=["\n\n", "\n"]
    )
    return text_splitter.split_text(text)

def get_vectorstore(text_chunks):
    
   # embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer

    Context:\n {context}\n
    Question: \n{question}\n
    Answer:
    """
    model = ChatGoogleGenerativeAI(model='gemini-flash-latest', temperature=0.3)
    
    
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = prompt | model | StrOutputParser()
    return chain

def user_input(user_question):
    
    #embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    #embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    context = "\n\n".join([doc.page_content for doc in docs])
    chain = get_conversational_chain()
    response = chain.invoke({"context": context, "question": user_question})
    st.write("Reply: ", response)

def main():
    
    
    st.set_page_config(page_title="Multiple PDF Chat", page_icon=":books:")
    st.header("Multiple PDF Chat :books:")
    st.write("Upload multiple PDF files and chat with them!")

    user_question = st.text_input("Ask a question from the PDF Files:")
    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.write("🤖")
        st.write("---")
        st.title("Upload PDF")
        pdf_docs = st.file_uploader(
            "Upload your PDF files",
            accept_multiple_files=True,
            type=["pdf", "PDF"]
        )

        if st.button("Process"):
            with st.spinner("Reading PDF files..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vectorstore(text_chunks)
                st.success("PDF files processed successfully!")

       

if __name__ == "__main__":
    main()
    
     