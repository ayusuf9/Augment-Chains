
import os
import json
import streamlit as st
import time
import pandas as pd
import openai
import faiss

from tqdm import tqdm
from tenacity import retry, stop_after_attempt, wait_random_exponential, retry_if_exception_type
from streamlit_navigation_bar import st_navbar
from streamlit_extras.switch_page_button import switch_page
from datetime import datetime

from langchain.document_loaders import PyMuPDFLoader, PyPDFLoader, PDFMinerLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores.faiss import FAISS
from langchain.chat_models import AzureChatOpenAI
from langchain.chains import ConversationalRetrievalChain, RetrievalQA
from langchain.prompts import PromptTemplate
from langchain import LLMChain

from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import List
from quanthub.util import llm

st.set_page_config(
    layout="wide",
    page_icon="📑",
    page_title="Tables",
    initial_sidebar_state="collapsed"
)


styles = {
    "nav": {
        "background-color": "rgb(54, 69, 79)",
        "height": "5rem",
    },
    "div": {
        "max-width": "24rem",
    },
    "span": {
        "border-radius": "0.5rem",
        "color": "rgb(255, 255, 255)",
        "margin": "0.125rem",
        "padding": "0.4375rem 0.625rem",
        "font-family": "'Arial', sans-serif",
        "font-size": "20px",
        "font-weight": "bold"
    },
    "active": {
        "background-color": "rgba(173, 216, 230, 0.25)",
        "font-weight": "bold",
        "padding": "14px",
    },
    "hover": {
        "background-color": "rgba(173, 216, 230, 0.35)",
    },
}

class LineOfInquiry(BaseModel):
    questions: List[str] = Field(description="Related questions to find relevant context")

def initialize_session_state():
    session_vars = {
        'custom_questions': [],
        'qa_results': pd.DataFrame(),
        'processed_documents': set(),
        'vector_stores': {},
        'embeddings': None,
        'llm': None
    }
    
    for var, default in session_vars.items():
        if var not in st.session_state:
            st.session_state[var] = default
    
    if st.session_state.embeddings is None:
        st.session_state.embeddings = load_embeddings_model()
    if st.session_state.llm is None:
        st.session_state.llm = load_llm_model()

def load_embeddings_model():
    openai_api_client = llm.get_azure_openai_client()
    try:
        embeddings = OpenAIEmbeddings(
            deployment="text-embedding-ada-002",
            model="text-embedding-ada-002",
            openai_api_key=openai_api_client.api_key,
            openai_api_base=openai_api_client.api_base,
            openai_api_type=openai_api_client.api_type,
            chunk_size=1
        )
        return embeddings
    except Exception as e:
        st.error("Failed to load embedding model, please refresh.")
        st.error(f"Error details: {str(e)}")
        return None

def load_llm_model():
    openai_api_client = llm.get_azure_openai_client()
    try:
        llm_model = AzureChatOpenAI(
            deployment_name='gpt-4o',
            model_name='gpt-4o',
            openai_api_version="2024-02-01",
            openai_api_key=openai_api_client.api_key,
            openai_api_base=openai_api_client.api_base,
            openai_api_type="azure_ad",
            temperature=0.0,
            streaming=True
        )
        return llm_model
    except Exception as e:
        st.error("Failed to load LLM model, please refresh.")
        st.error(f"Error details: {str(e)}")
        return None

def get_available_indexes(base_path):
    try:
        index_folders = [d for d in os.listdir(base_path) if d.endswith('_faiss_index')]
        identifiers = [folder.replace('_faiss_index', '') for folder in index_folders]
        return sorted(identifiers)
    except Exception as e:
        st.error(f"Error reading indexes: {str(e)}")
        return []

def load_vector_store(identifier, base_path):
    try:
        if identifier in st.session_state.vector_stores:
            return st.session_state.vector_stores[identifier]
        
        index_folder = os.path.join(base_path, f"{identifier}_faiss_index")
        
        if not os.path.exists(index_folder):
            st.error(f"Index folder not found for {identifier}")
            return None
        
        vector_store = FAISS.load_local(
            index_folder,
            embeddings=st.session_state.embeddings,
            allow_dangerous_deserialization=True
        )
        st.session_state.vector_stores[identifier] = vector_store
        return vector_store
    except Exception as e:
        st.error(f"Error loading index for {identifier}: {str(e)}")
        return None

def get_multi_query_retriever(vector_store):
    if not st.session_state.llm:
        return None

    question_generator_prompt = PromptTemplate(
        input_variables=["question"],
        template="""You are an expert in municipal finance, particularly in bond issuance structures, debt hierarchy, and related documentation.
      Break down the question into simpler, related questions to improve document retrieval accuracy within municipal bond contexts.
      
      Question: {question}
      
      Now, please generate similar alternative questions for the following question."""
  )
  

    try:
        retriever = MultiQueryRetriever.from_llm(
            llm=st.session_state.llm,
            retriever=vector_store.as_retriever(search_kwargs={"k": 10}),
            parser_key="questions",
            prompt=question_generator_prompt
        )
        return retriever
    except Exception as e:
        st.error(f"Error creating multi-query retriever: {str(e)}")
        return None

def process_new_queries(vector_stores, queries, existing_results=None):
    if not st.session_state.llm:
        st.error("LLM not initialized properly")
        return pd.DataFrame()

    chat_prompt = PromptTemplate(
        
        template="""
        **Question:**

        {question}

        **Instructions for the AI Assistant:**

        You are an expert investment analyst specializing in analyzing municipal bond issuance documents, particularly those from MuniOs.com. Your role is to provide accurate, insightful, and professional answers to questions about municipal bonds, just like a seasoned financial professional in a leading financial company.

        **Context:**

        {context}

        **Answer:**

        [Provide your detailed analysis and final answer here.]
        """,
            input_variables=["question", "context"]
    )
        
    
    simplify_prompt = PromptTemplate(
        input_variables=["answer"],
        template="""
          Given the following answer, extract and return only the key point. The key point should be a concise summary that directly conveys the main information, such as:
  
          **Answer:**
          {answer}
  
          **Key point:**
                  """
          )
    
    simplify_chain = LLMChain(llm=st.session_state.llm, prompt=simplify_prompt)
    
    results = pd.DataFrame(columns=list(vector_stores.keys()), index=queries) if existing_results is None else existing_results.copy()
    
    new_questions = [q for q in queries if q not in results.index]
    
    if new_questions:
        new_rows = pd.DataFrame(index=new_questions, columns=results.columns)
        results = pd.concat([results, new_rows])
    
    total_operations = len(queries) * len(vector_stores)
    if total_operations > 0:
        with st.spinner("Processing queries..."):
            progress_bar = st.progress(0)
            current_operation = 0
            
            for identifier, vector_store in vector_stores.items():
                multi_retriever = get_multi_query_retriever(vector_store)
                if not multi_retriever:
                    continue
                
                qa_chain = RetrievalQA.from_chain_type(
                    llm=st.session_state.llm,
                    chain_type="stuff",
                    retriever=multi_retriever,
                    chain_type_kwargs={"prompt": chat_prompt}
                )
                
                for query in queries:
                    try:
                        result = qa_chain.run(query)
                        simplified_result = simplify_chain.run(answer=result)
                        results.at[query, identifier] = simplified_result.strip()
                    except Exception as e:
                        results.at[query, identifier] = f"Error: {str(e)}"
                    
                    current_operation += 1
                    progress_bar.progress(current_operation / total_operations)
            
            progress_bar.empty()
    
    return results

def style_dataframe(df):
    return df.style.set_properties(**{
        'text-align': 'left',
        'font-size': '16px',
        'border-color': 'lightgrey',
        'border-style': 'solid',
        'border-width': '1px',
        'padding': '8px'
    }).set_table_styles([
        {'selector': 'th', 'props': [
            ('background-color', '#f0f2f6'),
            ('color', 'black'),
            ('font-weight', 'bold'),
            ('text-align', 'left'),
            ('padding', '8px'),
            ('font-size', '18px')
        ]},
        {'selector': 'tr:nth-of-type(even)', 'props': [
            ('background-color', '#f9f9f9')
        ]},
        {'selector': 'td:first-child', 'props': [
            ('font-weight', 'bold'),
            ('background-color', '#f0f2f6'),
            ('font-size', '17px')
        ]},
        {'selector': 'table', 'props': [
            ('margin-left', 'auto'),
            ('margin-right', 'auto')
        ]}
    ])

def main():
    initialize_session_state()
    
    page = st_navbar(["Home", "Chat", "Tables"], selected="Tables", styles=styles)
    
    if page == "Home":
        switch_page("streamlit app")
    if page == "Chat":
        switch_page("chat")

    st.warning("Please note that this is a Prototype and not the full-fledged application.")
    st.warning("The more robust version is underway..")

    base_path = '/app/pdfs_qa'
    available_identifiers = get_available_indexes(base_path)
    
    st.markdown(
        """
    <style>
    span[data-baseweb="tag"] {
      background-color: rgb(54, 69, 79) !important;
    }
    </styl
    """,
        unsafe_allow_html=True,
    )
    selected_identifiers = st.multiselect(
        "Select documents to analyze:",
        available_identifiers
    )
    
    default_questions = [
        "where is this property located?",
        "what year was the property built",
        "what is the total number of units?",
        "What is the Cap Rate?",
        "What is the Total Senior Debt?",
        "what is the Total Mezzanine Debt?",
        "what is the Total Debt?",
    ]

    new_question = st.text_input("Add a question:")
    if st.button("Add Question") and new_question:
        if new_question not in st.session_state.custom_questions:
            st.session_state.custom_questions.append(new_question)
    
    all_questions = default_questions + st.session_state.custom_questions

    if selected_identifiers:
        vector_stores = {}
        for identifier in selected_identifiers:
            vector_store = load_vector_store(identifier, base_path)
            if vector_store:
                vector_stores[identifier] = vector_store
        
        if vector_stores:
            results_df = process_new_queries(
                vector_stores,
                all_questions,
                st.session_state.qa_results
            )
            
            st.session_state.qa_results = results_df
            
            display_df = results_df[selected_identifiers]
            styled_df = style_dataframe(display_df)
            st.write(styled_df.to_html(escape=False), unsafe_allow_html=True)
            
            if st.button("Export Results"):
                csv = display_df.to_csv().encode('utf-8')
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name="comp_tables.csv",
                    mime="text/csv"
                )

    elif st.session_state.qa_results is not None:
        st.session_state.qa_results = pd.DataFrame()

if __name__ == "__main__":
    main()