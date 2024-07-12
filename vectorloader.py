import os
import re
import time
#import pdfplumber
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Initialize OpenAI
os.environ["OPENAI_API_KEY"] = ""
os.environ["PINECONE_API_KEY"] = ""

index_name = "hs-courses"
embeddings = OpenAIEmbeddings()

loader = PyPDFLoader("MathCourses.pdf")
documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

vectorstore_from_docs = PineconeVectorStore.from_documents(docs, index_name=index_name, embedding=embeddings)

# Define a function to preprocess text
# def preprocess_text(text):
#     # Replace consecutive spaces, newlines and tabs
#     text = re.sub(r'\s+', ' ', text)
#     return text

# def process_pdf(file_path):
#     # create a loader
#     loader = PyPDFLoader(file_path)
#     # load your data
#     data = loader.load()
#     # Split your data up into smaller documents with Chunks
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
#     documents = text_splitter.split_documents(data)
#     # Convert Document objects into strings
#     texts = [str(doc) for doc in documents]
#     return texts

# # Define a function to create embeddings
# def create_embeddings(texts):
#     embeddings_list = []
#     for text in texts:
#         time.sleep(10)
#         res = client.embeddings.create(input=[text], model=MODEL)
#         print(res)
#         embeddings_list.append(res.data[0].embedding)
#     return embeddings_list

# # Define a function to upsert embeddings to Pinecone
# def upsert_embeddings_to_pinecone(index, embeddings, ids):
#     index.upsert(vectors=[(id, embedding) for id, embedding in zip(ids, embeddings)])

# # Process a PDF and create embeddings
# file_path = "MathCourses.pdf"  # Replace with your actual file path
# texts = process_pdf(file_path)
# embeddings = create_embeddings(texts)

# # Upsert the embeddings to Pinecone
# upsert_embeddings_to_pinecone(index, embeddings, [file_path])