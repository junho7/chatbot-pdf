import streamlit as st
from langchain_google_genai import GoogleGenerativeAI
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
import os
import tempfile
from dotenv import load_dotenv

load_dotenv()


# Set your Google API key
os.environ['GOOGLE_API_KEY'] = os.getenv("GEMINI_API_KEY")

# Initialize Gemini model
llm = GoogleGenerativeAI(model="gemini-pro")

# Hardcode the path to your PDF file
PDF_FILE_PATH = "./GenAI_Handbook.pdf"

# Function to process the PDF
def process_pdf(file_path):
    # Load the PDF
    loader = PyPDFLoader(file_path)
    documents = loader.load()

    # Split the documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)

    # Create embeddings
    embeddings = HuggingFaceEmbeddings()

    # Create a vectorstore from the chunks
    db = Chroma.from_documents(texts, embeddings)

    # Create a retriever interface from the vectorstore
    retriever = db.as_retriever(search_kwargs={'k': 3})

    # Create a chain to answer questions
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type='stuff', retriever=retriever)

    return qa

# Streamlit app
st.title('PDF Chatbot with Google Gemini')

# Process the PDF when the app starts
@st.cache_resource
def load_qa_system():
    return process_pdf(PDF_FILE_PATH)

qa = load_qa_system()

st.success(f'PDF processed successfully: {PDF_FILE_PATH}')

# Chat interface
query = st.text_input('Ask a question about the PDF:')
if query:
    response = qa.run(query)
    st.write('Answer:', response)