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


st.write("Hello, World")
