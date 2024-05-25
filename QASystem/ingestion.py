from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import BedrockEmbeddings
from langchain_community.llms import Bedrock

import json
import os
import sys
import boto3

# Create Bedrock client with the correct service name and region
bedrock = boto3.client(service_name="bedrock-runtime", region_name="us-east-1")
bedrock_embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v1", client=bedrock)

def data_ingestion():
    try:
        loader = PyPDFDirectoryLoader("./data")
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        docs = text_splitter.split_documents(documents)
        return docs
    except Exception as e:
        print(f"Error during data ingestion: {e}")
        sys.exit(1)

def get_vector_store(docs):
    try:
        vector_store_faiss = FAISS.from_documents(docs, bedrock_embeddings)
        vector_store_faiss.save_local("faiss_index")
        return vector_store_faiss
    except Exception as e:
        print(f"Error creating FAISS vector store: {e}")
        sys.exit(1)

if __name__ == '__main__':
    docs = data_ingestion()
    get_vector_store(docs)
