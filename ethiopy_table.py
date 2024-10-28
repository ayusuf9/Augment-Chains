import os
import json
import streamlit as st
import time
import pandas as pd
from tqdm import tqdm
import openai
import faiss
from tenacity import retry, stop_after_attempt, wait_random_exponential, retry_if_exception_type
from streamlit_navigation_bar import st_navbar
from streamlit_extras.switch_page_button import switch_page

from langchain.document_loaders import PyMuPDFLoader, PyPDFLoader, PDFMinerLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores.faiss import FAISS
from langchain.chat_models import AzureChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain import LLMChain
from langchain.chains import RetrievalQA
from datetime import datetime
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.chains import LLMChain
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import List

st.set_page_config(layout="wide", 
                   page_icon="🗂️",
                   page_title="Tables",
                   initial_sidebar_state="collapsed")

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

page = st_navbar(["Home", "Chat", "Tables"], selected="Tables", styles=styles)

if page == "Home":
    switch_page("app")
if page == "Chat":
    switch_page("chat")

st.markdown(
    """
<style>
span[data-baseweb="tag"] {
  background-color: rgb(54, 69, 79) !important;
}
</style>
""",
    unsafe_allow_html=True,
)

def initialize_session_state():
    if 'custom_questions' not in st.session_state:
        st.session_state.custom_questions = []
    if 'qa_results' not in st.session_state:
        st.session_state.qa_results = pd.DataFrame()
    if 'processed_documents' not in st.session_state:
        st.session_state.processed_documents = set()
    if 'vector_stores' not in st.session_state:
        st.session_state.vector_stores = {}

def get_available_indexes(base_path):
    index_folders = [d for d in os.listdir(base_path) if d.endswith('_faiss_index')]
    identifiers = [folder.replace('_faiss_index', '') for folder in index_folders]
    return sorted(identifiers)

def initialize_models():
    llm_instance = ChatOpenAI(
            model_name="gpt-4o",
            openai_api_key=OPENAI_API_KEY,
            temperature=0.0,
            streaming=True
        )

    embedding_model = OpenAIEmbeddings(
            model="text-embedding-3-small",
            openai_api_key=OPENAI_API_KEY,
            openai_api_base=os.getenv("OPENAI_API_BASE"),
            openai_api_type=os.getenv("OPENAI_API_TYPE"),
            chunk_size=100
        )
    
    return embedding_model, llm_instance

def get_multi_query_retriever(vector_store, llm_instance):
    """Create a multi-query retriever for more comprehensive document retrieval."""

    question_generator_prompt = PromptTemplate(
        input_variables=["question"],
        template="""You are an expert in municipal finance, particularly in bond issuance structures, debt hierarchy, and related documentation.
        Break down the question into simpler, related questions to improve document retrieval accuracy within municipal bond contexts.
        
        Question: {question}
        
        Generate 3 alternative questions that incorporate key aspects of municipal bond financing, such as types of bonds (e.g., senior, mezzanine), debt aggregation, and issuance details. 
        Ensure that the alternative questions approach the query from diverse angles but remain closely related to the original question, using municipal finance-specific terminology where applicable.
        
        Example:
        Original question: "What is the total debt?"
        
        To get the total debt, the amounts of two types of bonds issued needs to be added.
        
        For example: 
        The total debt for this issuance consists of:
        - $88,970,000 in Essential Housing Senior Lien Revenue Bonds, Series 2021A
        - $40,635,000 in Essential Housing Mezzanine Lien Revenue Bonds, Series 2021B
        
        This results in a combined total debt of $129,605,000.
        
        Alternative questions:
        1. "What are the individual amounts of senior and mezzanine lien bonds in this issuance, and how do they combine to form the total debt?"
        2. "How is the total debt calculated by aggregating different bond series, such as senior lien and mezzanine lien bonds, in this issuance?"
        3. "What is the breakdown of bond types that contribute to the overall debt total for this municipal issuance?"
        
        Now, please generate similar alternative questions for the following question."""
    )
    
    # question_generator_prompt = PromptTemplate(
    #     input_variables=["question"],
    #     template="""You are an expert at breaking down complex questions into simpler, related questions.
    #     Generate multiple different ways to ask the following question to get comprehensive context:
        
    #     Question: {question}
        
    #     Provide 3 alternative questions that would help gather relevant information.
    #     Make sure the questions are diverse in their approach but related to the original question."""
    # )

    retriever = MultiQueryRetriever.from_llm(
        llm=llm_instance,
        retriever=vector_store.as_retriever(search_kwargs={"k": 10}),
        parser_key="questions",
        prompt=question_generator_prompt,
        #verbose=True
    )
    
    return retriever

def load_vector_store(identifier, base_path, embedding_model):
    """Load a specific FAISS index if not already loaded."""
    if identifier in st.session_state.vector_stores:
        return st.session_state.vector_stores[identifier]
    
    index_folder = os.path.join(base_path, f"{identifier}_faiss_index")
    
    if not os.path.exists(index_folder):
        st.error(f"Index folder not found for {identifier}")
        return None
    
    try:
        vector_store = FAISS.load_local(
            index_folder,
            embeddings=embedding_model,
            allow_dangerous_deserialization=True
        )
        st.session_state.vector_stores[identifier] = vector_store
        return vector_store
    except Exception as e:
        st.error(f"Error loading index for {identifier}: {str(e)}")
        return None

def process_new_queries(vector_stores, llm_instance, queries, existing_results=None):
    """Process queries using multi-query retriever for better context gathering."""
    chat_prompt = PromptTemplate(
        template="""
        **Question:**
        {question}
        ---
        **Context:**
        {context}
        
        **Answer:**
        [Provide a clear, precise, and concise answer to the question based on all the gathered context.]
        """,
        input_variables=["question", "context"]
    )
    
    simplify_prompt = PromptTemplate(
        input_variables=["answer"],
        template="""
        Extract the key point from this answer:
        {answer}
        Key point:
        """
    )
    
    simplify_chain = LLMChain(llm=llm_instance, prompt=simplify_prompt)
    
    if existing_results is not None and not existing_results.empty:
        results = existing_results.copy()
    else:
        results = pd.DataFrame(columns=list(vector_stores.keys()), index=queries)
    
    new_questions = [q for q in queries if q not in results.index]

    if new_questions:
        new_rows = pd.DataFrame(index=new_questions, columns=results.columns)
        results = pd.concat([results, new_rows])
    
    documents_to_process = vector_stores.keys()
    questions_to_process = {}
    
    for doc in documents_to_process:
        if doc not in st.session_state.processed_documents:
            questions_to_process[doc] = queries
        else:
            questions_to_process[doc] = new_questions
        
        if doc not in results.columns:
            results[doc] = None
    
    total_operations = sum(len(qs) for qs in questions_to_process.values())
    if total_operations > 0:
        with st.spinner(f"Processing queries with enhanced retrieval..."):
            progress_bar = st.progress(0)
            current_iteration = 0
            
            for identifier, questions in questions_to_process.items():
                if not questions:  
                    continue
                    
                vector_store = vector_stores[identifier]
                multi_retriever = get_multi_query_retriever(vector_store, llm_instance)
                
                qa_chain = RetrievalQA.from_chain_type(
                    llm=llm_instance,
                    chain_type="stuff",
                    retriever=multi_retriever,
                    chain_type_kwargs={"prompt": chat_prompt},
                    verbose=False,
                )
                
                for query in questions:
                    try:
                        result = qa_chain.run(query)
                        simplified_result = simplify_chain.run(answer=result)
                        results.at[query, identifier] = simplified_result.strip()
                    except Exception as e:
                        results.at[query, identifier] = f"Error: {str(e)}"
                    
                    current_iteration += 1
                    progress_bar.progress(current_iteration / total_operations)

                progress_bar.empty()
                st.session_state.processed_documents.add(identifier)
    
    return results

def style_dataframe(df):
    """Apply consistent styling to the results dataframe."""
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
    st.success("Municipal Bond Document Analysis")
    initialize_session_state()
    
    base_path = '/Users/ayusuf/chatmee'
    
    available_identifiers = get_available_indexes(base_path)
    
    selected_identifiers = st.multiselect(
        "Select documents to analyze:",
        available_identifiers
    )
    
    default_questions = [
        "where is this property located?",
        "what is the total number of units?",
        "What is the occupancy rate most recently?",
        "What is the change in total income the last three months?",
        "What is the total income?"
    ]
    
    new_question = st.text_input("Add a custom question:")
    if st.button("Add Question") and new_question:
        if new_question not in st.session_state.custom_questions:
            st.session_state.custom_questions.append(new_question)
    
    all_questions = default_questions + st.session_state.custom_questions
    
    if selected_identifiers:
        embedding_model, llm_instance = initialize_models()
        
        vector_stores = {}
        for identifier in selected_identifiers:
            vector_store = load_vector_store(identifier, base_path, embedding_model)
            if vector_store:
                vector_stores[identifier] = vector_store
        
        if vector_stores:
            results_df = process_new_queries(
                vector_stores,
                llm_instance,
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