from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from PyPDF2 import PdfReader
import os
from dotenv import load_dotenv

load_dotenv()

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=os.getenv("GEMINI_API_KEY")
)

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.getenv("GEMINI_API_KEY")
)

def process_pdf(pdf):

    text = ""

    pdf_reader = PdfReader(pdf)

    for page in pdf_reader.pages:
        text += page.extract_text()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_text(text)

    vector_store = FAISS.from_texts(chunks, embedding=embeddings)

    return vector_store

def ask_pdf_question(vector_store, question):

    docs = vector_store.similarity_search(question)

    chain = load_qa_chain(llm, chain_type="stuff")

    response = chain.run(
        input_documents=docs,
        question=question
    )

    return response