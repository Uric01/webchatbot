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

@st.cache_data(max_entries=5, ttl=3600)
loader = UnstructuredURLLoader(urls=URLs)
data = loader.load()

chunks = CharacterTextSplitter(separator='\n', chunk_size=1000, chunk_overlap=200).split_documents(data)


st.write("Hello, World")
