import sys
import os
import torch
import textwrap
from langchain.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQAWithSourcesChain
import tenacity
import nltk
from langchain.vectorstores import FAISS
import streamlit as st
import getpass
import os
import httpx

from langchain.memory import ConversationBufferMemory

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

URLs=['https://www.shoecity.co.za/',
      'https://www.shoecity.co.za/collections/mens-footwear-view-all',
      'https://www.shoecity.co.za/collections/womens-footwear-view-all'
      ]

loader = UnstructuredURLLoader(urls=URLs)
data = loader.load()

chunks = CharacterTextSplitter(separator='\n', chunk_size=1000, chunk_overlap=200).split_documents(data)

#Embeddings Model
from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
if embeddings is None:
  raise ValueError("Embeddings model not found. Make sure it is set in your .env file.")
else:
  st.write("Loaded Embeddings model found!")

vector_store = FAISS.from_documents(chunks,embeddings)

#Components - LLM

# Configure retry behavior
retry_policy = tenacity.retry(
    wait=tenacity.wait_exponential(multiplier=1, min=4, max=10),  # Exponential backoff
    stop=tenacity.stop_after_attempt(5),  # Maximum 5 retries
    retry=tenacity.retry_if_exception_type(httpx.HTTPStatusError)  # Retry on HTTPStatusError
)

# Wrap your LLM calls with the retry decorator
@retry_policy
def run_llm(llm, **kwargs):
    return llm(**kwargs)

if not os.environ.get("MISTRAL_API_KEY"):
  os.environ["MISTRAL_API_KEY"] = getpass.getpass("Enter API key for Mistral AI:")

from langchain_mistralai import ChatMistralAI

llm = ChatMistralAI(model="mistral-large-latest", temperature=0) #The maximum token limit for the mistral-large-latest model is 131k tokens

chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vector_store.as_retriever())

memory = ConversationBufferMemory()
user_input = "what is my name?"

result = chain({"question": user_input})

memory.save_context({"input": user_input}, {"output": result["answer"]})

st.write(result["answer"])

st.write("Hello, World")
