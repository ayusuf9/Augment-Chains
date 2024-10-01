# File: qa_functions.py
# This file contains all your existing functions from the notebook

import os
import warnings
import logging
import hashlib
from pathlib import Path
import numpy as np
import time
import asyncio
import tiktoken
import json
import logging
from typing import List, Dict, Union
from functools import lru_cache

from tenacity import retry, stop_after_attempt, wait_random_exponential

from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA, ConversationalRetrievalChain, LLMChain
from langchain.chat_models import AzureChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainFilter
from langchain.schema.vectorstore import VectorStore
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema import BaseOutputParser
from langchain.callbacks import get_openai_callback
from langchain.retrievers.document_compressors import EmbeddingsFilter
from quanthub.util import llm

warnings.filterwarnings('ignore')

openai = llm.get_llm_client() 
OUTPUT_PRICE_PER_1K_TOKENS = 0.015

# # Place all your existing functions here
# def get_file_hash(file_path):
#     # ... [Your existing function code] ...

# def get_embedding(text):
#     # ... [Your existing function code] ...

# # ... [All other functions from your notebook] ...

# def ask_question(question, chain):
#     # ... [Your existing function code] ...

# File: app.py
# This is your Streamlit app file

import streamlit as st
from langchain.callbacks import StreamlitCallbackHandler
from qa_functions import *  # Import all functions from qa_functions.py

# Streamlit app
st.set_page_config(page_title="PDF Q&A Chatbot", page_icon="📚")
st.title("PDF Q&A Chatbot")

# PDF selection
pdf_options = {
    "PDF 1": '/users/CFII_DataScience/USERs/SPTADM/Improved_LLM_Chat/pwk_second.pdf',
    "PDF 2": '/users/CFII_DataScience/USERs/SPTADM/Improved_LLM_Chat/pwk_second.pdf'
}
selected_pdf = st.selectbox("Select a PDF to query:", list(pdf_options.keys()))

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None

# Load vectorstore and initialize qa_chain
if st.session_state.vectorstore is None or st.session_state.qa_chain is None:
    with st.spinner("Loading PDF and initializing Q&A system..."):
        st.session_state.vectorstore = process_pdf_with_metadata(pdf_options[selected_pdf])
        st.session_state.qa_chain = get_qa_chain(st.session_state.vectorstore)

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask a question about the PDF"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        stream_handler = StreamlitCallbackHandler(st.empty())
        response = st.session_state.qa_chain.run(prompt, callbacks=[stream_handler])
        st.session_state.messages.append({"role": "assistant", "content": response})

# Display cost (optional)
if st.checkbox("Show cost"):
    total_cost = sum(message.get("cost", 0) for message in st.session_state.messages)
    st.write(f"Total cost: ${total_cost:.4f}")