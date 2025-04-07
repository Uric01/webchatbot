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


nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

URLs=['https://www.shoecity.co.za/',
      'https://www.shoecity.co.za/collections/mens-footwear-view-all',
      'https://www.shoecity.co.za/collections/womens-footwear-view-all'
      ]

@st.cache_data()
def load_data(url):
      loader = UnstructuredURLLoader(urls=url)
      return loader


loader = load_data(URLs)
data = loader.load()

chunks = CharacterTextSplitter(separator='\n', chunk_size=1000, chunk_overlap=200).split_documents(data)

#Embeddings Model
from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
if embeddings is None:
  raise ValueError("Embeddings model not found. Make sure it is set in your .env file.")
else:
  print("Loaded Embeddings model found!")
