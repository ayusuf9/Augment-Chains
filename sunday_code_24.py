import os
import json
import streamlit as st
import time
import pandas as pd
import openai
import faiss

from tqdm import tqdm
from tenacity import retry, stop_after_attempt, wait_random_exponential, retry_if_exception_type
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

CACHE_FILE = "qa_preprocessed_cache.json"

st.set_page_config(
    layout="wide",
    page_icon="📑",
    page_title="QA Processor",
    initial_sidebar_state="collapsed"
)

class LineOfInquiry(BaseModel):
    questions: List[str] = Field(description="Related questions to find relevant context")

def initialize_session_state():
    """Initialize session state variables"""
    session_vars = {
        'qa_results': pd.DataFrame(),
        'processed_documents': set(),
        'vector_stores': {},
        'embeddings': None,
        'llm': None,
        'processing_status': {}
    }
    
    for var, default in session_vars.items():
        if var not in st.session_state:
            st.session_state[var] = default
    
    if st.session_state.embeddings is None:
        st.session_state.embeddings = load_embeddings_model()
    if st.session_state.llm is None:
        st.session_state.llm = load_llm_model()

def load_embeddings_model():
    """Initialize OpenAI embeddings model"""
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
        st.error(f"Failed to load embedding model: {str(e)}")
        return None

def load_llm_model():
    """Initialize Azure OpenAI LLM model"""
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
        st.error(f"Failed to load LLM model: {str(e)}")
        return None

def get_available_indexes(base_path):
    """Get list of available document indexes"""
    try:
        index_folders = [d for d in os.listdir(base_path) if d.endswith('_faiss_index')]
        identifiers = [folder.replace('_faiss_index', '') for folder in index_folders]
        return sorted(identifiers)
    except Exception as e:
        st.error(f"Error reading indexes: {str(e)}")
        return []

def load_vector_store(identifier, base_path):
    """Load FAISS vector store for a given identifier"""
    try:
        index_folder = os.path.join(base_path, f"{identifier}_faiss_index")
        
        if not os.path.exists(index_folder):
            st.error(f"Index folder not found for {identifier}")
            return None
        
        vector_store = FAISS.load_local(
            index_folder,
            embeddings=st.session_state.embeddings,
            allow_dangerous_deserialization=True
        )
        return vector_store
    except Exception as e:
        st.error(f"Error loading index for {identifier}: {str(e)}")
        return None

def get_multi_query_retriever(vector_store):
    """Initialize multi-query retriever"""
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

def save_to_cache(results_df):
    """Save processed results to cache file"""
    try:
        cache_data = {
            'timestamp': datetime.now().isoformat(),
            'results': results_df.to_dict()
        }
        with open(CACHE_FILE, 'w') as f:
            json.dump(cache_data, f)
        st.success(f"Successfully saved results to {CACHE_FILE}")
    except Exception as e:
        st.error(f"Failed to save cache: {str(e)}")

def process_queries(vector_stores, queries, progress_bar=None):
    """Process queries against vector stores"""
    if not st.session_state.llm:
        st.error("LLM not initialized properly")
        return pd.DataFrame()

    chat_prompt = PromptTemplate(
        template="""
        **Question:**
        {question}
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
        Given the following answer, extract and return only the key point. The key point should be a concise summary that directly conveys the main information.
        
        **Answer:**
        {answer}
        
        **Key point:**
        """
    )
    
    simplify_chain = LLMChain(llm=st.session_state.llm, prompt=simplify_prompt)
    
    results = pd.DataFrame(columns=list(vector_stores.keys()), index=queries)
    total_operations = len(queries) * len(vector_stores)
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
            if progress_bar is not None:
                progress_bar.progress(current_operation / total_operations)
    
    return results

def style_dataframe(df):
    """Apply styling to the DataFrame"""
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
        ]}
    ])

def main():
    st.title("QA Processor - Admin Version")
    st.warning("This is the processing version - run once to generate cache")
    
    initialize_session_state()
    
    base_path = '/app/pdfs_qa'
    available_identifiers = get_available_indexes(base_path)
    
    # Document selection
    selected_identifiers = st.multiselect(
        "Select documents to process:",
        available_identifiers
    )
    
    # Default questions
    default_questions = [
        "where is this property located?",
        "what year was the property built",
        "what is the total number of units?",
        "What is the Cap Rate?",
        "What is the Total Senior Debt?",
        "what is the Total Mezzanine Debt?",
        "what is the Total Debt?",
    ]

    st.write("### Default Questions to Process:")
    for q in default_questions:
        st.write(f"- {q}")

    # Processing section
    if st.button("Process All Questions"):
        if not selected_identifiers:
            st.error("Please select at least one document to process.")
            return

        with st.spinner("Processing... This may take several minutes"):
            progress_bar = st.progress(0)
            
            # Load vector stores
            vector_stores = {}
            for identifier in selected_identifiers:
                vector_store = load_vector_store(identifier, base_path)
                if vector_store:
                    vector_stores[identifier] = vector_store
            
            if vector_stores:
                # Process queries
                results_df = process_queries(
                    vector_stores,
                    default_questions,
                    progress_bar
                )
                
                # Save results
                save_to_cache(results_df)
                
                # Display results
                st.write("### Processed Results:")
                styled_df = style_dataframe(results_df)
                st.write(styled_df.to_html(escape=False), unsafe_allow_html=True)
                
                # Export option
                csv = results_df.to_csv().encode('utf-8')
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name="processed_results.csv",
                    mime="text/csv"
                )
            else:
                st.error("Failed to load vector stores for selected documents.")

if __name__ == "__main__":
    main()