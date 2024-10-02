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
from typing import List, Dict, Union
from functools import lru_cache

import streamlit as st  # New import for Streamlit

from tenacity import retry, stop_after_attempt, wait_random_exponential

from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA, LLMChain
from langchain.chat_models import AzureChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import EmbeddingsFilter
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema import BaseOutputParser
from langchain.callbacks import get_openai_callback
from quanthub.util import llm

warnings.filterwarnings('ignore')

openai = llm.get_llm_client()
OUTPUT_PRICE_PER_1K_TOKENS = 0.015

def get_file_hash(file_path):
    with open(file_path, "rb") as file:
        file_hash = hashlib.md5(file.read()).hexdigest()
    return file_hash

def get_embedding(text):
    embedding = openai.Embedding.create(
        input=text,
        deployment_id="text-embedding-ada-002",
        model="text-embedding-ada-002",
        api_key=openai.api_key
    )
    return embedding.data[0].embedding

def process_chunk(chunk):
    embedding = get_embedding(chunk.page_content)
    return (chunk.page_content, embedding, chunk.metadata)

def get_dynamic_text_splitter(text):
    avg_word_length = sum(len(word) for word in text.split()) / len(text.split())
    target_chunk_size = int(3000 / avg_word_length) * 10  # aiming for ~2000 words

    # ensuring chunk size is within reasonable bounds
    min_chunk_size = 2000
    max_chunk_size = 5000
    chunk_size = max(min_chunk_size, min(target_chunk_size, max_chunk_size))

    return RecursiveCharacterTextSplitter(
      chunk_size=chunk_size,
      chunk_overlap=chunk_size // 10,
      length_function=len,
      separators=["\n\n", "\n", " ", ""]  # splitting on paragraph, then line, then word
    )

def process_pdf_with_metadata(file_path):
    file_hash = get_file_hash(file_path)
    cache_dir = Path(f"./VectorStore/{file_hash}")
    faiss_index_path = cache_dir / "index.faiss"

    if faiss_index_path.exists():
        #print(f"Loading files from cache..: {cache_dir}")
        try:
            embedding_function = OpenAIEmbeddings(
                deployment_id="text-embedding-ada-002",
                model="text-embedding-ada-002",
                api_key=openai.api_key
            )
            vectorstore = FAISS.load_local(str(cache_dir), 
                                           embeddings=embedding_function,
                                           allow_dangerous_pickling=True)
            return vectorstore
        except Exception as e:
            print(f"Error loading cached index: {str(e)}. Reprocessing PDF.")
    
    print("Cache not found. Processing new PDF.")

    try:
        loader = PyPDFLoader(file_path)
        pages = loader.load()
        
        for i, page in enumerate(pages):
            page.metadata["page_number"] = i + 1
        
        full_text = ' '.join([page.page_content for page in pages])
        text_splitter = get_dynamic_text_splitter(full_text)
        chunks = text_splitter.split_documents(pages)
        
        # chunking right here.. 
        batch_size = 10 
        embeddings = []
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i+batch_size]
            batch_results = [process_chunk(chunk) for chunk in batch]
            embeddings.extend(batch_results)
            print(f"Processed {i+len(batch)}/{len(chunks)} chunks")
            time.sleep(5)
        
        vectorstore = FAISS.from_embeddings(
            text_embeddings=[(e[0], e[1]) for e in embeddings],
            embedding=get_embedding, 
            metadatas= [e[2] for e in embeddings] 
        )

        cache_dir.mkdir(parents=True, exist_ok=True)
        vectorstore.save_local(str(cache_dir))

        print(f"Vectorstore saved to {cache_dir}")
        
        return vectorstore
    except Exception as e:
        print(f"An error occurred while processing the PDF: {str(e)}")
        return None

class LineListOutputParser(BaseOutputParser[List[str]]):
    def parse(self, text: str) -> List[str]:
        lines = text.strip().split("\n")
        return list(filter(None, lines))  # Remove empty lines

def setup_logging():
    logger = logging.getLogger("langchain.retrievers.multi_query")
    logger.setLevel(logging.INFO)
    
    file_handler = logging.FileHandler("v1_multi_query_retriever.log")
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger

def get_llm_instance(deployment_name, temperature=0.0, max_tokens=800):
    return AzureChatOpenAI(
        deployment_name=deployment_name,
        openai_api_version="2023-07-01-preview",
        openai_api_key=openai.api_key,
        openai_api_base=openai.api_base,
        openai_api_type=openai.api_type,
        temperature=temperature,
        max_tokens=max_tokens
    )

def generate_multiple_queries(question: Union[str, dict]) -> List[str]:
    # Handle both string and dict inputs
    if isinstance(question, dict):
        question = question.get('query', '')
    
    llm_instance = get_llm_instance(llm.GPT_4_OMNI_MODEL, temperature=0.0, max_tokens=200)
    output_parser = LineListOutputParser()
    
    QUERY_PROMPT = PromptTemplate(
        input_variables=["question"],
        template="""
        As an AI assistant, your task is to generate 3 related versions of the given question. 
        These alternative questions should help retrieve all relevant information from a database, considering different aspects and phrasings.

        Ensure each alternative question is distinct and adds value to the information retrieval process. Provide these 3 alternative questions separated by newlines.

        Original question: {question}

        Alternative questions:
        """,
    )

    llm_chain = LLMChain(llm=llm_instance, prompt=QUERY_PROMPT)
    result = llm_chain.run(question)
    return output_parser.parse(result)

def get_improved_retriever(vectorstore: VectorStore):
    base_retriever = vectorstore.as_retriever(search_kwargs={"k": 20})  # Adjust k as needed
    llm_instance = get_llm_instance(llm.GPT_4_OMNI_MODEL)

    multi_query_retriever = MultiQueryRetriever.from_llm(
        retriever=base_retriever,
        llm=llm_instance,
        parser_key="lines"
    )

    logger = setup_logging()
    
    embeddings = OpenAIEmbeddings(
        deployment_id="text-embedding-ada-002",
        model="text-embedding-ada-002",
        api_key=openai.api_key
    )
    
    embeddings_filter = EmbeddingsFilter(embeddings=embeddings, similarity_threshold=0.70, k=20)

    compression_retriever = ContextualCompressionRetriever(
        base_compressor=embeddings_filter,
        base_retriever=multi_query_retriever
    )

    return compression_retriever

def get_qa_chain(vectorstore: VectorStore):
    retriever = get_improved_retriever(vectorstore)
    llm_gpt = get_llm_instance(llm.GPT_4_OMNI_MODEL, temperature=0.0, max_tokens=500)
    
    cot_prompt = PromptTemplate(
        template="""
        Question: {question}
        
        Context (This is the data/information you should use in answering the question):
          {context}

          You are an AI assistant specializing in analyzing municipal bond prospectuses. 
          You provide accurate and insightful answers to questions, just like a financial professional would.               
          Your response should be structured as follows:
          
          When presented with a prospectus and a question, focus on the following:

            1. **Accuracy is key**: Provide precise and correct information based on the prospectus content.
            2. **Calculations matter**: When asked to calculate, ensure your calculations are accurate and reflect a professional standard.
            3. **Context is crucial**: Frame your answers within the context of the specific bond issue and the overall municipal bond market.
            
            For example, if asked:
            
            (a). "What is the total debt?" Accurately calculate the answer based on the provided financial statements. Note that total debt is the sum of all Series principal amounts (e.g., Series A, Series B, etc.) listed in the document. Do not include any interest rates or interest amounts in this calculation. Only consider the principal amounts of the debt series when computing the total debt.
            (b). "What is the purpose of this bond issuance?" provide a clear and concise answer directly from the prospectus.
            (c). "What are the risks associated with this bond?" analyze and explain the risk factors outlined in the prospectus.
            Remember, your role is to help users understand the intricacies of municipal bonds through accurate information, precise calculations, and clear explanations.

          Your response should be structured as follows:
          Answer: [Provide a clear, precise, and concise answer to the question, including only the most relevant information and numerical results.]

        """,
        input_variables=["question", "context"]
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm_gpt,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": cot_prompt},
        return_source_documents=True
    )
    return qa_chain

def ask_question(question, chain):
    from langchain.callbacks import get_openai_callback
    with get_openai_callback() as cb:
        response = chain({"query": question})
    output_cost = (cb.completion_tokens / 1000) * OUTPUT_PRICE_PER_1K_TOKENS
    return response["result"], output_cost

def main():
    st.title("PDF Question Answering Chatbot")

    # Dropdown to select PDF
    pdf_option = st.selectbox(
        'Select a PDF to query:',
        ('First PDF', 'Second PDF')
    )

    if pdf_option == 'First PDF':
        pdf_path = '/users/CFII_DataScience/USERs/SPTADM/Improved_LLM_Chat/pwk_first.pdf'
    else:
        pdf_path = '/users/CFII_DataScience/USERs/SPTADM/Improved_LLM_Chat/pwk_second.pdf'  # Update with the actual second PDF path

    # Initialize vectorstore and QA chain when the app first runs or when pdf_option changes
    if 'qa_chain' not in st.session_state or st.session_state['pdf_path'] != pdf_path:
        with st.spinner('Processing PDF and setting up QA chain...'):
            vectorstore = process_pdf_with_metadata(pdf_path)
            qa_chain = get_qa_chain(vectorstore)
            st.session_state['qa_chain'] = qa_chain
            st.session_state['pdf_path'] = pdf_path
            st.session_state['messages'] = []

    # Chat interface using Streamlit's chat functions
    if 'messages' not in st.session_state:
        st.session_state['messages'] = []

    # Display past messages
    for message in st.session_state['messages']:
        if message["role"] == "user":
            with st.chat_message("user"):
                st.write(message["content"])
        else:
            with st.chat_message("assistant"):
                st.write(message["content"])

    # User input
    if prompt := st.chat_input("Ask a question about the PDF"):
        # Display user message
        with st.chat_message("user"):
            st.write(prompt)
        st.session_state['messages'].append({"role": "user", "content": prompt})

        # Get response from QA chain
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                result, cost = ask_question(prompt, st.session_state['qa_chain'])
                st.write(result)
                st.session_state['messages'].append({"role": "assistant", "content": result})

if __name__ == "__main__":
    main()
