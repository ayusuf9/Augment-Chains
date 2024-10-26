import os
import json
import streamlit as st
import time
from tqdm import tqdm
import openai
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
from datetime import datetime

st.set_page_config(page_icon="🏢",
                   page_title="Muni",
                   initial_sidebar_state="collapsed")

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

page = st_navbar(["Home", "Chat", "Tables"], selected="Chat", styles=styles)

if page == "Home":
    switch_page("app")
if page == "Tables":
    switch_page("tables")

# openai_api_client = llm2.get_llm_client()  TODO this is for CG 

st.markdown("""
    <style>
    /* Import Google Font */
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400&display=swap');

    /* Apply Google Font to all text elements */
    html, body, [class*="css"]  {
        font-family: 'Karla', sans-serif;
    }
    </style>
    """, unsafe_allow_html=True)

st.warning("Please note that this is a Prototype and not the full-fledged application.")

# Initialize the embedding model
@st.cache_resource
def get_embedding_model():
    return OpenAIEmbeddings(
            model="text-embedding-3-small",
            openai_api_key=OPENAI_API_KEY,
            openai_api_base=os.getenv("OPENAI_API_BASE"),
            openai_api_type=os.getenv("OPENAI_API_TYPE"),
            chunk_size=100
        )
    # return OpenAIEmbeddings(
    #     deployment="text-embedding-ada-002",
    #     model="text-embedding-ada-002",
    #     openai_api_key=openai_api_client.api_key,
    #     openai_api_base=openai_api_client.api_base,
    #     openai_api_type=openai_api_client.api_type,
    #     chunk_size=1
    # )

def get_avatar(role):
    if role == "user":
        return ":material/account_box:"
    elif role == "assistant":
        return ":material/psychology_alt:"
    else:
        return None

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
    index_dir = os.path.join(base_directory, f"{identifier}_faiss_index")
    
    try:
        if os.path.isdir(index_dir):
            vector_store = FAISS.load_local(
                index_dir, 
                embeddings=get_embedding_model(),
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
    """
    I am getting the list of available indices from the base directory I had setup within this function.
    The goal is to return the clean identifiers (without '_faiss_index') -- to be used in the select box.
    """
    try:
        indices = [d.replace('_faiss_index', '') 
                  for d in os.listdir(base_directory) 
                  if d.endswith('_faiss_index') and os.path.isdir(os.path.join(base_directory, d))]
        return sorted(indices) 
    except Exception as e:
        st.error(f"Error reading indices: {str(e)}")
        return []

def main():
    if 'current_identifier' not in st.session_state:
        st.session_state.current_identifier = None
    if 'vector_store' not in st.session_state:
        st.session_state.vector_store = None
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    base_directory = '/Users/ayusuf/chatmee'

    available_indices = get_available_indices(base_directory)
    
    if not available_indices:
        st.error("No index files found in the specified directory.")
        return

    selected_identifier = st.selectbox(
        "Select a Muni Doc",
        available_indices,
        format_func=lambda x: x
    )
    st.success(f"Successfully Loaded {selected_identifier}.pdf, now you can proceed with your questions.")

    if selected_identifier:
        if st.session_state.current_identifier != selected_identifier:
            vector_store = load_index_from_identifier(selected_identifier, base_directory)
            if vector_store is not None:
                st.session_state.vector_store = vector_store
                st.session_state.current_identifier = selected_identifier
                st.session_state.chat_history = [] 

        if st.session_state.vector_store is not None:
            llm_instance = ChatOpenAI(
            model_name="gpt-4o",
            openai_api_key=OPENAI_API_KEY,
            temperature=0.0,
            streaming=True
        )
            # llm_instance = AzureChatOpenAI(
            #     deployment_name='gpt-4o',
            #     model_name='gpt-4o',
            #     openai_api_version="2023-07-01-preview",
            #     openai_api_key=openai_api_client.api_key,
            #     openai_api_base=openai_api_client.api_base,
            #     openai_api_type="azure_ad",
            #     temperature=0.0,
            #     streaming=True
            # )

            retriever = st.session_state.vector_store.as_retriever(search_kwargs={"k": 30})

            chat_prompt = PromptTemplate(
                template="""
                **Question:**
                {question}
                ---
                **Rules for the AI Assistant:**
                You are an AI assistant specializing in analyzing municipal bond documents from MuniOs.com.
                Your task is to provide accurate, relevant information based on the provided context.
                
                **Context:**
                {context}
                
                **Instructions:**
                1. Answer based only on the provided context
                2. If you're unsure, say so
                3. Keep responses concise and focused
                4. Use bullet points for multiple items
                5. Format numbers and data clearly
                
                **Answer:**
                """,
                input_variables=["question", "context"]
            )

            qa_chain = ConversationalRetrievalChain.from_llm(
                llm=llm_instance,
                retriever=retriever,
                combine_docs_chain_kwargs={"prompt": chat_prompt},
                return_source_documents=True,
            )

            for message in st.session_state.chat_history:
                with st.chat_message(message["role"], avatar=get_avatar(message["role"])):
                    st.markdown(message["content"])

            if prompt := st.chat_input("What is your question?"):
                st.session_state.chat_history.append({"role": "user", "content": prompt})
                with st.chat_message("user", avatar=":material/account_box:"):
                    st.markdown(prompt)

                with st.chat_message("assistant", avatar=":material/psychology_alt:"):
                    with st.spinner("Thinking..."):
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

if __name__ == "__main__":
    main()