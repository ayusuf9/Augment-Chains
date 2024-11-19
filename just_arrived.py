import os
import openai
import json
import streamlit as st
import pandas as pd
from datetime import datetime
from streamlit_navigation_bar import st_navbar
from streamlit_extras.switch_page_button import switch_page
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores.faiss import FAISS
from langchain.chat_models import AzureChatOpenAI
from langchain.chains import RetrievalQA, LLMChain
from langchain.prompts import PromptTemplate
from langchain.retrievers.multi_query import MultiQueryRetriever
from quanthub.util import llm2

# Constants
CACHE_FILE = "cached_data.json"
BASE_PATH = '/Users/sptadm/Desktop/point_local_2'

# Page Configuration
st.set_page_config(
    layout="wide",
    page_icon="📑",
    page_title="Property Analysis",
    initial_sidebar_state="collapsed"
)

# Navigation Styles
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

def get_available_indexes(base_path):
    """Get list of available FAISS indexes from the base path"""
    try:
        indexes = []
        for item in os.listdir(base_path):
            if item.endswith('_faiss_index'):
                identifier = item.replace('_faiss_index', '')
                indexes.append(identifier)
        return indexes
    except Exception as e:
        st.error(f"Error scanning indexes: {str(e)}")
        return []

def load_cached_results():
    """Load cached results from file or initialize from available indexes"""
    try:
        # First try to load from cache
        if os.path.exists(CACHE_FILE):
            with open(CACHE_FILE, 'r') as f:
                cache_data = json.load(f)
                df = pd.DataFrame.from_dict(cache_data.get('results', {}))
                if not df.empty:
                    return df
        
        # If no cache exists or it's empty, initialize from available indexes
        available_indexes = get_available_indexes(BASE_PATH)
        if available_indexes:
            # Initialize an empty DataFrame with columns for each index
            df = pd.DataFrame(columns=available_indexes)
            
            # Save this initial structure to cache
            with open(CACHE_FILE, 'w') as f:
                json.dump({'results': df.to_dict()}, f)
            
            return df
        else:
            st.warning("No FAISS indexes found in the base path")
            return pd.DataFrame()
            
    except Exception as e:
        st.error(f"Error loading cache: {str(e)}")
        return pd.DataFrame()

def load_embeddings_model():
    """Initialize OpenAI embeddings model"""
    try:
        openai_api_client = llm2.get_llm_client()
        embeddings = OpenAIEmbeddings(
            deployment="text-embedding-3-large",
            model="text-embedding-3-large",
            openai_api_key=openai_api_client.api_key,
            openai_api_base=openai_api_client.api_base,
            openai_api_type=openai_api_client.api_type,
            chunk_size=100
        )
        return embeddings
    except Exception as e:
        st.error(f"Failed to load embeddings: {str(e)}")
        return None

def load_llm_model():
    """Initialize Azure OpenAI model"""
    try:
        openai_api_client = llm2.get_llm_client()
        llm = AzureChatOpenAI(
            deployment_name=llm2.GPT_4_O1_MODEL,
            model_name='o1-preview',
            openai_api_version="2023-07-01-preview",
            openai_api_key=openai_api_client.api_key,
            openai_api_base=openai_api_client.api_base,
            openai_api_type="azure_ad",
            temperature=1,
        )
        return llm
    except Exception as e:
        st.error(f"Failed to load LLM: {str(e)}")
        return None

def initialize_session_state():
    """Initialize all session state variables"""
    # Always ensure base path exists
    if not os.path.exists(BASE_PATH):
        st.error(f"Base path not found: {BASE_PATH}")
        return

    # Load results first
    if 'results_df' not in st.session_state:
        st.session_state.results_df = load_cached_results()
    
    session_vars = {
        'custom_questions': [],
        'qa_results': pd.DataFrame(),
        'processed_documents': set(),
        'vector_stores': {},
        'processing': False
    }
    
    for var, default in session_vars.items():
        if var not in st.session_state:
            st.session_state[var] = default
    
    # Initialize models with special handling
    if 'llm' not in st.session_state or st.session_state.llm is None:
        st.session_state.llm = load_llm_model()
    if 'embeddings' not in st.session_state or st.session_state.embeddings is None:
        st.session_state.embeddings = load_embeddings_model()

def get_multi_query_retriever(vector_store):
    """Setup multi-query retriever with error handling"""
    if not vector_store:
        return None

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

def load_vector_stores(identifier, base_path):
    """Load FAISS vector stores with proper error handling"""
    try:
        # Check if already loaded
        if identifier in st.session_state.vector_stores:
            return st.session_state.vector_stores[identifier]
        
        # Validate path and requirements
        index_folder = os.path.join(base_path, f"{identifier}_faiss_index")
        if not os.path.exists(index_folder):
            st.error(f"Index folder not found for {identifier}")
            return None
        
        if st.session_state.embeddings is None:
            st.error("Embeddings model not initialized")
            return None
        
        # Load vector store
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

def process_queries(vector_stores, queries, progress_bar):
    """Process queries against vector stores with comprehensive error handling"""
    if not st.session_state.llm:
        st.error("LLM not initialized properly")
        return pd.DataFrame()

    chat_prompt = PromptTemplate(
        template="""
        **Question:**
        {question}

        **Context:**
        {context}

        **Instructions:**
        You are an expert investment analyst specializing in analyzing municipal bond issuance documents. 
        Provide accurate, insightful, and professional answers based solely on the provided context.
        
        **Answer:**
        [Provide your detailed analysis here.]
        """,
        input_variables=["question", "context"]
    )

    simplify_prompt = PromptTemplate(
        input_variables=["answer"],
        template="""
        Given the following answer, extract and return only the key point:
        {answer}
        
        Return only the essential information without explanation or context.
        """
    )

    simplify_chain = LLMChain(llm=st.session_state.llm, prompt=simplify_prompt)
    results = pd.DataFrame(columns=list(vector_stores.keys()), index=queries)
    total_operations = len(queries) * len(vector_stores)
    current_operation = 0

    try:
        for identifier, vector_store in vector_stores.items():
            if vector_store is None:
                st.warning(f"Skipping {identifier} - vector store not loaded")
                continue

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
                    with st.spinner(f"Processing query: {query[:50]}..."):
                        result = qa_chain.run(query)
                        simplified_result = simplify_chain.run(answer=result)
                        results.at[query, identifier] = simplified_result.strip()
                except Exception as e:
                    results.at[query, identifier] = f"Error: {str(e)}"
                    st.warning(f"Error processing query '{query}' for {identifier}: {str(e)}")

                current_operation += 1
                if progress_bar is not None:
                    progress_bar.progress(current_operation / total_operations)

        return results
    except Exception as e:
        st.error(f"Fatal error in process_queries: {str(e)}")
        return pd.DataFrame()

def style_dataframe(df):
    """Apply consistent styling to the results dataframe"""
    return df.style.set_properties(**{
        'text-align': 'left',
        'font-size': '16px',
        'padding': '8px',
        'border': '1px solid lightgrey'
    }).set_table_styles([
        {'selector': 'th', 'props': [
            ('background-color', '#f0f2f6'),
            ('font-size', '15px')
        ]},
        {'selector': 'tr:nth-of-type(even)', 'props': [
            ('background-color', '#f9f9f9')
        ]}
    ])

def main():
    """Main application function with error handling"""
    try:
        initialize_session_state()

        if not st.session_state.llm or not st.session_state.embeddings:
            st.error("Failed to initialize LLM or embeddings model")
            return

        # Navigation setup
        options = {'show_menu': False}
        page = st_navbar(["Home", "Chat", "Tables"], selected="Tables", styles=styles, options=options)
        
        if page == "Home": 
            switch_page("streamlit app")
        if page == "Chat": 
            switch_page("chat")

        st.title("Comp Tables")

        # Debug information at the top
        if st.checkbox("Show Debug Info"):
            st.write("Base Path:", BASE_PATH)
            st.write("Available Indexes:", get_available_indexes(BASE_PATH))
            st.write("Results DataFrame Columns:", st.session_state.results_df.columns.tolist())
            st.write("Session State Keys:", list(st.session_state.keys()))

        # Document selection with actual indexes
        available_docs = get_available_indexes(BASE_PATH)
        if not available_docs:
            st.error("No FAISS indexes found in the base directory. Please check your data.")
            return
            
        selected_docs = st.multiselect(
            "Select documents:", 
            available_docs, 
            default=available_docs[:min(5, len(available_docs))]
        )

        if selected_docs:
            # Display current results if they exist
            if not st.session_state.results_df.empty:
                display_df = st.session_state.results_df[selected_docs]
                styled_df = style_dataframe(display_df)
                st.write(styled_df.to_html(escape=False), unsafe_allow_html=True)

            st.markdown("### Analysis Results")

            # Define questions
            default_questions = [
                "where is this property located?",
                "what year was the property built",
                "what is the total number of units?",
                "What is the Cap Rate?",
                "What is the Total Senior Debt?",
                "what is the Total Mezzanine Debt?",
                "what is the Total Debt?",
                "How much is in the coverage reserve fund",
                "How much is in the senior capitalized interest fund",
                "How much is in the mezzanine capitalized interest",
                "what is the Rental Revenues in 2024",
                "Net Operating Income in 2024?"
            ]

            # Question input
            new_question = st.text_input("Add a new question:", key="new_question_input")
            if st.button("Add Question") and new_question.strip():
                if new_question not in st.session_state.custom_questions:
                    st.session_state.custom_questions.append(new_question)
                    st.success(f"Added new question: {new_question}")
                else:
                    st.warning("This question already exists!")

            # Process queries
            all_questions = default_questions + st.session_state.custom_questions
            
            with st.spinner("Loading vector stores..."):
                vector_stores = {}
                for identifier in selected_docs:
                    vector_store = load_vector_stores(identifier, BASE_PATH)
                    if vector_store:
                        vector_stores[identifier] = vector_store

            if vector_stores:
                progress_bar = st.progress(0)
                
                try:
                    with st.spinner("Processing questions..."):
                        results_df = process_queries(
                            vector_stores,
                            all_questions,
                            progress_bar
                        )
                        
                        st.session_state.qa_results = results_df
                        styled_df = style_dataframe(results_df)
                        st.write(styled_df.to_html(escape=False), unsafe_allow_html=True)

                        # Export functionality
                        if st.button("Export Results"):
                            csv = results_df.to_csv().encode('utf-8')
                            st.download_button(
                                label="Download CSV",
                                data=csv,
                                file_name=f"analysis_results_{datetime.now().strftime('%Y%m%d')}.csv",
                                mime="text/csv"
                            )

                except Exception as e:
                    st.error(f"Error processing questions: {str(e)}")
                    st.exception(e)
            else:
                st.warning("No vector stores could be loaded. Please check your data sources.")
        else:
            st.info("Please select documents to view analysis")

    except Exception as e:
        st.error(f"Application error: {str(e)}")
        st.exception(e)

if __name__ == "__main__":
    main()