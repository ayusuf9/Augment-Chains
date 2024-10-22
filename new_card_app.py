import streamlit as st
import pandas as pd
import os
import time
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores.faiss import FAISS
from langchain.chat_models import AzureChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain import LLMChain
from quanthub.util import llm

# Initialize session state
if 'processed_queries' not in st.session_state:
    st.session_state.processed_queries = set()
if 'results_df' not in st.session_state:
    st.session_state.results_df = None
if 'show_instructions' not in st.session_state:
    st.session_state.show_instructions = False

# Custom CSS
st.markdown("""
    <style>
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .main-header {
        color: #1E3D59;
        padding: 1rem 0;
        border-bottom: 2px solid #E8EEF2;
        margin-bottom: 2rem;
    }
    .subheader {
        color: #2E5077;
        margin: 1.5rem 0;
    }
    .card {
        background-color: #FFFFFF;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }
    .instruction-text {
        font-size: 0.9rem;
        color: #666666;
        line-height: 1.5;
    }
    .metric-card {
        background-color: #F7F9FC;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
    }
    .status-message {
        padding: 0.75rem;
        border-radius: 6px;
        margin: 1rem 0;
    }
    .success-message {
        background-color: #D4EDDA;
        color: #155724;
    }
    .warning-message {
        background-color: #FFF3CD;
        color: #856404;
    }
    </style>
""", unsafe_allow_html=True)

# Configuration and Setup
st.set_page_config(
    page_title="PDF Analysis System",
    page_icon="📑",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize OpenAI client and models (same as before)
@st.cache_resource
def initialize_openai():
    openai_api_client = llm.get_azure_openai_client()
    embedding_model = OpenAIEmbeddings(
        deployment_id="text-embedding-ada-002",
        model="text-embedding-ada-002",
        api_key=openai_api_client.api_key
    )
    llm_instance = AzureChatOpenAI(
        deployment_name=llm.GPT_4_OMNI_MODEL,
        openai_api_version="2023-07-01-preview",
        openai_api_key=openai_api_client.api_key,
        openai_api_base=openai_api_client.api_base,
        openai_api_type=openai_api_client.api_type,
        temperature=0.0,
    )
    return openai_api_client, embedding_model, llm_instance

# Load FAISS indexes (same as before)
@st.cache_resource
def load_vector_stores(pdf_directory, embedding_model):
    faiss_index_file = os.path.join(pdf_directory, "faiss_index")
    pdf_files = [f for f in os.listdir(pdf_directory) if f.endswith('.pdf')]
    vector_stores = {}
    
    for pdf_file in pdf_files:
        index_file = f"{faiss_index_file}_{pdf_file[:-4]}"
        if os.path.exists(f"{index_file}.faiss"):
            vector_stores[pdf_file] = FAISS.load_local(index_file, embeddings=embedding_model)
    
    return vector_stores, pdf_files

# Load existing results (same as before)
@st.cache_data
def load_existing_results(pdf_directory):
    results_path = os.path.join(pdf_directory, "results.csv")
    if os.path.exists(results_path):
        df = pd.read_csv(results_path, index_col=0)
        return df
    return pd.DataFrame()

# Initialize prompts (same as before)
def initialize_prompts():
    chat_prompt = PromptTemplate(
        template="""
        Question: {question}
        
        Context (This is the data/information you should use in answering the question):
        {context}

        You are an AI assistant specializing in analyzing municipal bond prospectuses. 
        You provide accurate and insightful answers to questions, just like a financial professional working in a big financial company would.               
        Your response should be structured as follows:
        
        When presented with extracted information from the municipal bond document and a question, focus on the following:

            1. **Accuracy is key**: Provide precise and correct information based on the prospectus content.
            2. **Calculations matter**: When asked to calculate, ensure your calculations are accurate and reflect a professional standard.
            3. **Context is crucial**: Frame your answers within the context of the specific bond issue and the overall municipal bond market.
            
        Your response should be structured as follows:
        Answer: [Provide a clear, precise, and concise answer to the question, including only the most relevant information and numerical results.]
        """,
        input_variables=["question", "context"]
    )

    simplify_prompt = PromptTemplate(
        input_variables=["answer"],
        template="""
        Given the following answer, extract and return just the key point, which should be a concise summary such as a number, percentage, name, or brief phrase that directly conveys the main information. Do not include any additional explanation or context.

        Answer:
        {answer}

        Key point:"""
    )
    
    return chat_prompt, simplify_prompt

# Process new query (same as before)
def process_query(query, vector_stores, llm_instance, chat_prompt, simplify_prompt):
    results = {}
    simplify_chain = LLMChain(llm=llm_instance, prompt=simplify_prompt)
    
    progress_bar = st.progress(0)
    total_files = len(vector_stores)
    
    for idx, (pdf_file, vector_store) in enumerate(vector_stores.items()):
        retriever = vector_store.as_retriever(search_kwargs={"k": 20})
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm_instance,
            chain_type="stuff",
            retriever=retriever,
            chain_type_kwargs={"prompt": chat_prompt},
            verbose=False,
        )
        
        try:
            with st.spinner(f'Processing {pdf_file}...'):
                result = qa_chain.run(query)
                simplified_result = simplify_chain.run(answer=result)
                results[pdf_file[:-4]] = simplified_result.strip()
        except Exception as e:
            results[pdf_file[:-4]] = f"Error: {str(e)}"
        
        progress_bar.progress((idx + 1) / total_files)
    
    progress_bar.empty()
    return results

def display_metrics(df):
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
            <div class="metric-card">
                <h3>Total Queries</h3>
                <p style="font-size: 24px; font-weight: bold;">{}</p>
            </div>
        """.format(len(df)), unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
            <div class="metric-card">
                <h3>Documents Analyzed</h3>
                <p style="font-size: 24px; font-weight: bold;">{}</p>
            </div>
        """.format(len(df.columns)), unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
            <div class="metric-card">
                <h3>Last Updated</h3>
                <p style="font-size: 24px; font-weight: bold;">{}</p>
            </div>
        """.format(time.strftime("%Y-%m-%d %H:%M")), unsafe_allow_html=True)

def main():
    # Initialize components
    pdf_directory = '/users/CFII_DataScience/USERs/SPTADM/Agents_Experiment/pdf_files/'
    openai_api_client, embedding_model, llm_instance = initialize_openai()
    vector_stores, pdf_files = load_vector_stores(pdf_directory, embedding_model)
    chat_prompt, simplify_prompt = initialize_prompts()
    
    # Main header
    st.markdown('<h1 class="main-header">📑 PDF Document Analysis System</h1>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown('<h2 class="subheader">Settings & Help</h2>', unsafe_allow_html=True)
        
        if st.button("Show/Hide Instructions"):
            st.session_state.show_instructions = not st.session_state.show_instructions
        
        if st.session_state.show_instructions:
            st.markdown("""
                <div class="instruction-text">
                <h4>How to use this system:</h4>
                1. Enter your query in the text box
                2. Click 'Process Query' to analyze
                3. Results will appear in the main table
                4. Previously processed queries are saved
                
                <h4>Query Tips:</h4>
                • Be specific and clear
                • One question at a time
                • Check existing results first
                </div>
            """, unsafe_allow_html=True)
    
    # Load existing results
    if st.session_state.results_df is None:
        st.session_state.results_df = load_existing_results(pdf_directory)
        if not st.session_state.results_df.empty:
            st.session_state.processed_queries.update(st.session_state.results_df.index)
    
    # Display metrics
    if not st.session_state.results_df.empty:
        display_metrics(st.session_state.results_df)
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown('<h2 class="subheader">Analysis Results</h2>', unsafe_allow_html=True)
        if not st.session_state.results_df.empty:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.dataframe(
                st.session_state.results_df,
                use_container_width=True,
                height=400
            )
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.info("No processed queries yet. Start by entering a query on the right.")
    
    with col2:
        st.markdown('<h2 class="subheader">Query Interface</h2>', unsafe_allow_html=True)
        st.markdown('<div class="card">', unsafe_allow_html=True)
        
        query = st.text_input("Enter your query:", placeholder="Type your question here...")
        
        if st.button("Process Query", use_container_width=True):
            if not query:
                st.markdown('<div class="status-message warning-message">Please enter a query.</div>', unsafe_allow_html=True)
            elif query in st.session_state.processed_queries:
                st.markdown('<div class="status-message warning-message">This query has already been processed.</div>', unsafe_allow_html=True)
            else:
                results = process_query(query, vector_stores, llm_instance, chat_prompt, simplify_prompt)
                
                # Update DataFrame
                new_row = pd.DataFrame([results], index=[query])
                st.session_state.results_df = pd.concat([st.session_state.results_df, new_row])
                st.session_state.processed_queries.add(query)
                
                # Save updated results
                output_file = os.path.join(pdf_directory, "results.csv")
                st.session_state.results_df.to_csv(output_file)
                
                st.markdown('<div class="status-message success-message">Query processed and results saved!</div>', unsafe_allow_html=True)
                st.experimental_rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()