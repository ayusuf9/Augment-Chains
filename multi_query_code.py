import os
from pathlib import Path
import json
import streamlit as st
import time
import openai
from tenacity import retry, stop_after_attempt, wait_random_exponential, retry_if_exception_type
from streamlit_navigation_bar import st_navbar
from streamlit_extras.switch_page_button import switch_page
from langchain.document_loaders import PyMuPDFLoader, PyPDFLoader, PDFMinerLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores.faiss import FAISS
from langchain.chat_models import AzureChatOpenAI
from langchain.chains import ConversationalRetrievalChain, LLMChain
from langchain.prompts import PromptTemplate
from langchain.retrievers.multi_query import MultiQueryRetriever
from datetime import datetime
from tqdm import tqdm
from quanthub.util import llm

st.set_page_config(
    page_icon="💬",
    page_title="Muni Chat",
    initial_sidebar_state="collapsed"
)

styles = {
    "nav": {
        "background-color": "rgb(54, 69, 79)",
        "height": "5rem",
    },
    "div": {
        "max-width": "20rem",
    },
    "span": {
        "border-radius": "0.5rem",
        "color": "rgb(255, 255, 255)",
        "margin": "0 0.125rem",
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

def initialize_session_state():
    session_vars = {
        'current_identifier': None,
        'vector_store': None,
        'chat_history': [],
        'embeddings': None,
        'llm': None,
        'multi_query_retriever': None
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

def get_avatar(role):
    return ":material/for_you:" if role == "user" else ":material/robot_2:" if role == "assistant" else None

def save_chat_history(identifier: str, chat_history: list):
    save_dir = "chat_histories"
    os.makedirs(save_dir, exist_ok=True)
    filename = os.path.join(save_dir, "all_chat_histories.json")
    
    new_chat_session = {
        "document_identifier": identifier,
        "session_timestamp": datetime.now().isoformat(),
        "conversations": []
    }
    
    for i in range(0, len(chat_history), 2):
        if i + 1 < len(chat_history):
            qa_pair = {
                "question": chat_history[i]["content"],
                "answer": chat_history[i + 1]["content"],
                "timestamp": datetime.now().isoformat()
            }
            new_chat_session["conversations"].append(qa_pair)
            
    try:
        if os.path.exists(filename):
            with open(filename, 'r', encoding='utf-8') as f:
                all_chats = json.load(f)
        else:
            all_chats = {"chat_sessions": []}
        
        today = datetime.now().date()
        session_updated = False
        
        for session in all_chats["chat_sessions"]:
            session_date = datetime.fromisoformat(session["session_timestamp"]).date()
            if session["document_identifier"] == identifier and session_date == today:
                session["conversations"] = new_chat_session["conversations"]
                session_updated = True
                break
        
        if not session_updated:
            all_chats["chat_sessions"].append(new_chat_session)
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(all_chats, f, indent=2, ensure_ascii=False)
            
    except Exception as e:
        st.error(f"Error saving chat history: {str(e)}")
        return None
    
    return filename

@st.cache_resource
def load_index_from_identifier(identifier: str, base_directory: str):
    """Load FAISS index using cached embeddings model"""
    index_dir = os.path.join(base_directory, f"{identifier}_faiss_index")
    
    try:
        if os.path.isdir(index_dir):
            vector_store = FAISS.load_local(
                index_dir, 
                embeddings=st.session_state.embeddings,
                allow_dangerous_deserialization=True
            )
            return vector_store
        else:
            st.error(f"Index directory not found for identifier: {identifier}")
            return None
    except Exception as e:
        st.error(f"Error loading index: {str(e)}")
        return None

def get_available_indices(base_directory: str) -> list:
    """Get list of available indices from directory"""
    try:
        os.makedirs(base_directory, exist_ok=True)
        indices = [
            d.replace('_faiss_index', '') 
            for d in os.listdir(base_directory) 
            if d.endswith('_faiss_index') and os.path.isdir(os.path.join(base_directory, d))
        ]
        return sorted(indices)
    except Exception as e:
        st.error(f"Error reading indices: {str(e)}")
        return []

def create_multi_query_retriever(vector_store):
    """Create a Multi-Query Retriever using the vector store"""
    if not vector_store or not st.session_state.llm:
        return None

    prompt_template = """You are an AI language model assistant specializing in municipal bonds. 
    Your task is to generate multiple different versions of the given question to help retrieve 
    relevant information from municipal bond documents. Each version should focus on different 
    aspects of the original question while maintaining its core intent.

    Generate {num_queries} different versions of the question that:
    1. Rephrase the question using different financial terminology
    2. Break down complex questions into simpler components
    3. Include relevant municipal bond-specific context

    Original question: {question}

    Generate {num_queries} different versions of the question, numbered 1-{num_queries}:"""

    question_generator_prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["question", "num_queries"]
    )

    llm_chain = LLMChain(llm=st.session_state.llm, prompt=question_generator_prompt)

    try:
        retriever = MultiQueryRetriever(
            retriever=vector_store.as_retriever(search_kwargs={"k": 10}),
            llm_chain=llm_chain,
            parser_key="numbered_lines",  # Parse numbered lines from LLM response
            num_queries=3  # Generate 3 different versions of each question
        )
        return retriever
    except Exception as e:
        st.error(f"Error creating Multi-Query Retriever: {str(e)}")
        return None

def setup_qa_chain():
    """Setup QA chain using cached LLM and vector store with Multi-Query Retriever"""
    if not st.session_state.vector_store or not st.session_state.llm:
        return None

    # Create Multi-Query Retriever
    multi_query_retriever = create_multi_query_retriever(st.session_state.vector_store)
    if not multi_query_retriever:
        st.error("Failed to create Multi-Query Retriever")
        return None

    st.session_state.multi_query_retriever = multi_query_retriever

    chat_prompt = PromptTemplate(
        template="""
        **Question:**

        {question}
        ---

        **Rules for the AI Assistant:**

        You are an AI assistant specializing in analyzing municipal bond documents from MuniOs.com. Your role is to provide accurate, insightful, and professional answers to questions about municipal bonds, just like a financial professional in a leading financial company would.

        When generating your response, please adhere to the following guidelines:

        1. **Use Only the Provided Context**: Base your answer solely on the context provided. Do not include information that is not present in the context. Avoid introducing any outside knowledge or assumptions.

        2. **Accuracy and Precision**: Ensure all information you provide is precise and correct. Pay special attention to numerical data and financial terms.

        3. **Calculations**: Perform any necessary calculations accurately, following professional financial standards. If calculations are involved, briefly show the calculation steps and provide the final result.

        4. **Contextual Understanding**: Frame your answer within the context of the specific bond issue and consider the broader municipal bond market as appropriate.

        5. **Clarity and Professionalism**: Write in clear, concise language appropriate for a professional audience. Use proper financial terminology and maintain a professional tone.

        **Things to note** 
        - note that the total debt is the sum of the principal without interest. 

        **Context (Based on the rules, provide an answer to the information to the following information):**

        {context}

        **Your response should be structured as follows:**

        **Answer:**

        [Provide a clear, precise, and concise answer to the question, including only the most relevant information and numerical results.]
        """,
        input_variables=["question", "context"]
    )

    return ConversationalRetrievalChain.from_llm(
        llm=st.session_state.llm,
        retriever=multi_query_retriever,
        combine_docs_chain_kwargs={"prompt": chat_prompt},
        return_source_documents=True,
    )

def handle_user_input(qa_chain, prompt):
    """Handle user input and generate response with improved retrieval"""
    try:
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar=":material/for_you:"):
            st.markdown(prompt)

        with st.chat_message("assistant", avatar=":material/robot_2:"):
            with st.spinner("Thinking... (Generating multiple queries for better retrieval)"):
                response = qa_chain({
                    "question": prompt,
                    "chat_history": [(msg["role"], msg["content"]) 
                                   for msg in st.session_state.chat_history]
                })
                st.markdown(response['answer'])
                st.session_state.chat_history.append(
                    {"role": "assistant", "content": response['answer']}
                )
                save_chat_history(
                    st.session_state.current_identifier,
                    st.session_state.chat_history
                )

    except Exception as e:
        st.error(f"Error generating response: {str(e)}")

def main():
    # Initialize session state
    initialize_session_state()
    
    page = st_navbar(["Home", "Chat", "Tables"], selected="Chat", styles=styles)
    
    if page == "Home":
        switch_page("streamlit app")
    if page == "Tables":
        switch_page("tables")
        
    st.warning("Please note that this is a Prototype and not the full-fledged application.")

    base_directory = '/app/pdfs_qa'
    available_indices = get_available_indices(base_directory)
    
    if not available_indices:
        st.error("No index files found in the specified directory.")
        return
    
    options = ['Select a Muni Doc'] + available_indices

    selected_identifier = st.selectbox(
        "Select a Muni Doc",
        options,
        index=0
    )

    if selected_identifier != 'Select a Muni Doc':
        st.success(f"Successfully loaded {selected_identifier}.pdf, now you can proceed with your questions.")
        
        if st.session_state.current_identifier != selected_identifier:
            vector_store = load_index_from_identifier(selected_identifier, base_directory)
            if vector_store is not None:
                st.session_state.vector_store = vector_store
                st.session_state.current_identifier = selected_identifier
                st.session_state.chat_history = []

        qa_chain = setup_qa_chain()
        if qa_chain is None:
            st.error("Failed to setup QA chain. Please refresh the page.")
            return

        for message in st.session_state.chat_history:
            with st.chat_message(message["role"], avatar=get_avatar(message["role"])):
                st.markdown(message["content"])
        
        if prompt := st.chat_input("What is your question?"):
            handle_user_input(qa_chain, prompt)
    else:
        st.info("Please select a Muni Doc to proceed.")

if __name__ == "__main__":
    main()