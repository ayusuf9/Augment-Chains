import os
import json
import streamlit as st
import pandas as pd
from datetime import datetime
from streamlit_navigation_bar import st_navbar
from streamlit_extras.switch_page_button import switch_page
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores.faiss import FAISS
from langchain.chat_models import AzureChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain import LLMChain
from langchain.retrievers.multi_query import MultiQueryRetriever
from quanthub.util import llm

# Configuration
CACHE_FILE = "qa_preprocessed_cache.json"

# Page setup
st.set_page_config(layout="wide", page_icon="📑", page_title="Property Analysis")

def initialize_session_state():
    if 'results_df' not in st.session_state:
        st.session_state.results_df = load_cached_results()
    if 'processing' not in st.session_state:
        st.session_state.processing = False
    if 'llm' not in st.session_state:
        st.session_state.llm = load_llm_model()
    if 'embeddings' not in st.session_state:
        st.session_state.embeddings = load_embeddings_model()

def load_cached_results():
    """Load pre-processed results from cache"""
    try:
        if os.path.exists(CACHE_FILE):
            with open(CACHE_FILE, 'r') as f:
                cache_data = json.load(f)
                return pd.DataFrame.from_dict(cache_data.get('results', {}))
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error loading cache: {str(e)}")
        return pd.DataFrame()

def load_embeddings_model():
    """Initialize embeddings model"""
    openai_api_client = llm.get_azure_openai_client()
    try:
        return OpenAIEmbeddings(
            deployment="text-embedding-ada-002",
            model="text-embedding-ada-002",
            openai_api_key=openai_api_client.api_key,
            openai_api_base=openai_api_client.api_base,
            openai_api_type=openai_api_client.api_type,
            chunk_size=1
        )
    except Exception as e:
        st.error(f"Failed to load embeddings: {str(e)}")
        return None

def load_llm_model():
    """Initialize LLM model"""
    openai_api_client = llm.get_azure_openai_client()
    try:
        return AzureChatOpenAI(
            deployment_name='gpt-4o',
            model_name='gpt-4o',
            openai_api_version="2024-02-01",
            openai_api_key=openai_api_client.api_key,
            openai_api_base=openai_api_client.api_base,
            openai_api_type="azure_ad",
            temperature=0.0,
            streaming=True
        )
    except Exception as e:
        st.error(f"Failed to load LLM: {str(e)}")
        return None

def get_multi_query_retriever(vector_store):
    """Setup retriever"""
    question_prompt = PromptTemplate(
        input_variables=["question"],
        template="""You are an expert in municipal finance. Break down the question:
        Question: {question}
        Generate similar alternative questions."""
    )
    
    try:
        return MultiQueryRetriever.from_llm(
            llm=st.session_state.llm,
            retriever=vector_store.as_retriever(search_kwargs={"k": 10}),
            parser_key="questions",
            prompt=question_prompt
        )
    except Exception as e:
        st.error(f"Retriever error: {str(e)}")
        return None

def process_new_question(question, vector_stores):
    """Process a single new question"""
    results = {}
    
    chat_prompt = PromptTemplate(
        template="""
        Question: {question}
        Context: {context}
        Answer: Provide a detailed analysis.""",
        input_variables=["question", "context"]
    )
    
    for identifier, vector_store in vector_stores.items():
        try:
            retriever = get_multi_query_retriever(vector_store)
            if not retriever:
                continue
                
            qa_chain = RetrievalQA.from_chain_type(
                llm=st.session_state.llm,
                chain_type="stuff",
                retriever=retriever,
                chain_type_kwargs={"prompt": chat_prompt}
            )
            
            result = qa_chain.run(question)
            results[identifier] = result.strip()
            
        except Exception as e:
            results[identifier] = f"Error: {str(e)}"
    
    return results

def load_vector_stores(selected_ids):
    """Load vector stores for selected documents"""
    vector_stores = {}
    base_path = '/app/pdfs_qa'
    
    for identifier in selected_ids:
        try:
            index_folder = os.path.join(base_path, f"{identifier}_faiss_index")
            if os.path.exists(index_folder):
                vector_store = FAISS.load_local(
                    index_folder,
                    embeddings=st.session_state.embeddings,
                    allow_dangerous_deserialization=True
                )
                vector_stores[identifier] = vector_store
        except Exception as e:
            st.error(f"Error loading index for {identifier}: {str(e)}")
    
    return vector_stores

def style_dataframe(df):
    """Style the results table"""
    return df.style.set_properties(**{
        'text-align': 'left',
        'font-size': '16px',
        'padding': '8px',
        'border': '1px solid lightgrey'
    }).set_table_styles([
        {'selector': 'th', 'props': [
            ('background-color', '#f0f2f6'),
            ('font-weight', 'bold'),
            ('font-size', '18px')
        ]},
        {'selector': 'tr:nth-of-type(even)', 'props': [
            ('background-color', '#f9f9f9')
        ]}
    ])

def main():
    initialize_session_state()
    
    # Navigation
    page = st_navbar(["Home", "Chat", "Tables"], selected="Tables")
    if page == "Home": switch_page("streamlit app")
    if page == "Chat": switch_page("chat")
    
    st.title("Property Analysis Dashboard")
    
    # Get available documents
    available_docs = st.session_state.results_df.columns.tolist()
    selected_docs = st.multiselect("Select documents:", available_docs)
    
    if selected_docs:
        # Display existing results
        display_df = st.session_state.results_df[selected_docs]
        
        # Question input
        col1, col2 = st.columns([3, 1])
        with col1:
            new_question = st.text_input("Add a new question:")
        with col2:
            if st.button("Process Question"):
                if new_question.strip():
                    with st.spinner("Processing new question..."):
                        # Load vector stores
                        vector_stores = load_vector_stores(selected_docs)
                        
                        # Process question
                        results = process_new_question(new_question, vector_stores)
                        
                        # Add to results DataFrame
                        st.session_state.results_df.loc[new_question] = pd.Series(results)
                        
                        # Update display
                        display_df = st.session_state.results_df[selected_docs]
                else:
                    st.warning("Please enter a question")
        
        # Display results table
        st.markdown("### Analysis Results")
        styled_df = style_dataframe(display_df)
        st.write(styled_df.to_html(escape=False), unsafe_allow_html=True)
        
        # Export option
        if st.button("Export Results"):
            csv = display_df.to_csv().encode('utf-8')
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"analysis_results_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
    else:
        st.info("Please select documents to view analysis")

if __name__ == "__main__":
    main()