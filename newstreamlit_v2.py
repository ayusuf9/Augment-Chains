import os
import warnings
import time
import pandas as pd
import streamlit as st
from tqdm import tqdm
import faiss
from tenacity import retry, stop_after_attempt, wait_random_exponential, retry_if_exception_type

from langchain.document_loaders import DirectoryLoader, PyMuPDFLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores.faiss import FAISS
from langchain.chat_models import AzureChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain import LLMChain

# Initialize session state
if 'custom_questions' not in st.session_state:
    st.session_state.custom_questions = []
if 'qa_results' not in st.session_state:
    st.session_state.qa_results = pd.DataFrame()
if 'vector_stores' not in st.session_state:
    st.session_state.vector_stores = {}

# Cache the model initialization
@st.cache_resource
def initialize_models():
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
    
    return embedding_model, llm_instance

# Cache the FAISS index loading
@st.cache_resource
def load_faiss_index(_pdf_path, _index_path, embedding_model):
    return FAISS.load_local(_index_path, embeddings=embedding_model)

# Cache the document processing and index creation
@st.cache_resource
def create_faiss_index(_pdf_path, embedding_model):
    loader = PyMuPDFLoader(_pdf_path)
    documents = loader.load()
    
    batch_size = 10
    all_texts = []
    all_embeddings = []
    
    for i in range(0, len(documents), batch_size):
        batch_docs = documents[i:i + batch_size]
        texts = [doc.page_content for doc in batch_docs]
        embeddings = embed_texts(texts, embedding_model)
        all_texts.extend(batch_docs)
        all_embeddings.extend(embeddings)
        time.sleep(5)
    
    pure_texts = [page.page_content for page in all_texts]
    textual_embeddings = zip(pure_texts, all_embeddings)
    return FAISS.from_embeddings(textual_embeddings, embedding_model)

def load_vector_stores(selected_pdfs, embedding_model):
    vector_stores = {}
    faiss_index_file = os.path.join(pdf_directory, "faiss_index")
    
    for pdf_file in selected_pdfs:
        pdf_path = os.path.join(pdf_directory, pdf_file)
        index_file = f"{faiss_index_file}_{pdf_file[:-4]}"
        
        try:
            if os.path.exists(f"{index_file}.faiss"):
                if pdf_file not in st.session_state.vector_stores:
                    st.session_state.vector_stores[pdf_file] = load_faiss_index(
                        pdf_path,
                        index_file,
                        embedding_model
                    )
                vector_stores[pdf_file] = st.session_state.vector_stores[pdf_file]
                st.success(f"Loaded existing FAISS index for {pdf_file}")
            else:
                if pdf_file not in st.session_state.vector_stores:
                    with st.spinner(f"Creating new index for {pdf_file}..."):
                        vector_store = create_faiss_index(pdf_path, embedding_model)
                        vector_store.save_local(index_file)
                        st.session_state.vector_stores[pdf_file] = vector_store
                vector_stores[pdf_file] = st.session_state.vector_stores[pdf_file]
                st.success(f"Created and saved FAISS index for {pdf_file}")
        except Exception as e:
            st.error(f"Error loading/creating index for {pdf_file}: {str(e)}")
            continue
    
    return vector_stores

# Cache the prompt templates
@st.cache_resource
def get_prompts():
    chat_prompt = PromptTemplate(
        template="""
        Question: {question}
        
        Context (This is the data/information you should use in answering the question):
        {context}

        You are an AI assistant specializing in analyzing municipal bond prospectuses. 
        Your response should be structured as follows:
        Answer: [Provide a clear, precise, and concise answer to the question]
        """,
        input_variables=["question", "context"]
    )
    
    simplify_prompt = PromptTemplate(
        input_variables=["answer"],
        template="""
        Given the following answer, extract and return only the key point:
        {answer}
        Key point:
        """
    )
    
    return chat_prompt, simplify_prompt

# Cache query processing for each PDF
@st.cache_data
def process_single_query(pdf_file, query, vector_store, llm_instance, chat_prompt, simplify_prompt):
    try:
        retriever = vector_store.as_retriever(search_kwargs={"k": 20})
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm_instance,
            chain_type="stuff",
            retriever=retriever,
            chain_type_kwargs={"prompt": chat_prompt},
            verbose=False,
        )
        
        simplify_chain = LLMChain(llm=llm_instance, prompt=simplify_prompt)
        
        result = qa_chain.run(query)
        simplified_result = simplify_chain.run(answer=result)
        return simplified_result.strip()
    except Exception as e:
        return f"Error: {str(e)}"

def process_queries(vector_stores, llm_instance, queries):
    chat_prompt, simplify_prompt = get_prompts()
    results = pd.DataFrame(index=queries, columns=[pdf[:-4] for pdf in vector_stores.keys()])
    
    progress_bar = st.progress(0)
    progress_text = st.empty()
    
    total_iterations = len(queries) * len(vector_stores)
    current_iteration = 0
    
    for pdf_file, vector_store in vector_stores.items():
        for query in queries:
            progress_text.text(f"Processing: {pdf_file} - {query}")
            
            result = process_single_query(
                pdf_file,
                query,
                vector_store,
                llm_instance,
                chat_prompt,
                simplify_prompt
            )
            results.at[query, pdf_file[:-4]] = result
            
            current_iteration += 1
            progress_bar.progress(current_iteration / total_iterations)
    
    progress_bar.empty()
    progress_text.empty()
    return results

# Streamlit app layout
st.title("Municipal Bond Document Analysis")

# Sidebar for configuration and PDF selection
with st.sidebar:
    st.header("Configuration")
    
    # PDF file selection
    pdf_directory = st.text_input("PDF Directory Path", value='/users/CFII_DataScience/USERs/SPTADM/Agents_Experiment/pdf_files/')
    pdf_files = [f for f in os.listdir(pdf_directory) if f.endswith('.pdf')]
    selected_pdfs = st.multiselect("Select PDF files", pdf_files)

    # Question management
    st.header("Add Custom Question")
    new_question = st.text_input("Enter your question:")
    if st.button("Add Question"):
        if new_question and new_question not in st.session_state.custom_questions:
            st.session_state.custom_questions.append(new_question)
            st.success("Question added successfully!")

    # Display and manage custom questions
    st.header("Custom Questions")
    for i, question in enumerate(st.session_state.custom_questions):
        col1, col2 = st.columns([3, 1])
        with col1:
            st.write(f"{i+1}. {question}")
        with col2:
            if st.button("Remove", key=f"remove_{i}"):
                st.session_state.custom_questions.pop(i)
                st.experimental_rerun()

# Main app logic
if selected_pdfs:
    st.header("Analysis Results")
    
    # Initialize models
    embedding_model, llm_instance = initialize_models()
    
    # Load vector stores for selected PDFs
    vector_stores = load_vector_stores(selected_pdfs, embedding_model)
    
    # Combine default and custom questions
    all_questions = list_of_queries + st.session_state.custom_questions
    
    # Process queries and update results
    results_df = process_queries(vector_stores, llm_instance, all_questions)
    st.session_state.qa_results = results_df
    
    # Display results
    st.dataframe(results_df)
    
    # Export functionality
    if st.button("Export Results to CSV"):
        csv = results_df.to_csv().encode('utf-8')
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name="analysis_results.csv",
            mime="text/csv"
        )
else:
    st.info("Please select one or more PDF files from the sidebar to begin analysis.")

# Add CSS to improve the layout
st.markdown("""
    <style>
    .stButton>button {
        width: 100%;
    }
    .stDataFrame {
        width: 100%;
    }
    </style>
    """, unsafe_allow_html=True)