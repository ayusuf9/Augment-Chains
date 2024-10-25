import os
import streamlit as st
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores.faiss import FAISS
from langchain.chat_models import AzureChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from streamlit_navigation_bar import st_navbar
from streamlit_extras.switch_page_button import switch_page
from quanthub.util import llm2

# Set page config
st.set_page_config(page_icon="🏢",
                   page_title="Muni",
                   initial_sidebar_state="collapsed")

# Navigation bar styles (unchanged)
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

openai_api_client = llm2.get_llm_client()

st.warning("Please note that this is a Prototype and not the full-fledged application.")

# Initialize the embedding model
@st.cache_resource
def get_embedding_model():
    return OpenAIEmbeddings(
        deployment="text-embedding-ada-002",
        model="text-embedding-ada-002",
        openai_api_key=openai_api_client.api_key,
        openai_api_base=openai_api_client.api_base,
        openai_api_type=openai_api_client.api_type,
        chunk_size=1
    )

def get_avatar(role):
    if role == "user":
        return ":material/account_box:"
    elif role == "assistant":
        return ":material/psychology_alt:"
    else:
        return None

@st.cache_resource
def load_index_from_identifier(identifier: str, base_directory: str):
    """
    Load a FAISS index directly from the stored directory structure.
    """
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
    Get list of available indices from the base directory.
    Returns the clean identifiers (without '_faiss_index').
    """
    try:
        # List all directories that end with '_faiss_index'
        indices = [d.replace('_faiss_index', '') 
                  for d in os.listdir(base_directory) 
                  if d.endswith('_faiss_index') and os.path.isdir(os.path.join(base_directory, d))]
        return sorted(indices)  # Sort alphabetically for consistent display
    except Exception as e:
        st.error(f"Error reading indices: {str(e)}")
        return []

def main():
    # Initialize session states
    if 'current_identifier' not in st.session_state:
        st.session_state.current_identifier = None
    if 'vector_store' not in st.session_state:
        st.session_state.vector_store = None
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    # Base directory where all index folders are stored
    base_directory = '/Users/sptadm/Desktop/point_local/pdfs_qa'
    
    # Get available indices
    available_indices = get_available_indices(base_directory)
    
    if not available_indices:
        st.error("No index files found in the specified directory.")
        return

    # Create dropdown with clean identifiers
    selected_identifier = st.selectbox(
        "Select a Muni Doc",
        available_indices,
        format_func=lambda x: x  # Display the clean identifier
    )

    if selected_identifier:
        # Only load new index if selection changes
        if st.session_state.current_identifier != selected_identifier:
            vector_store = load_index_from_identifier(selected_identifier, base_directory)
            if vector_store is not None:
                st.session_state.vector_store = vector_store
                st.session_state.current_identifier = selected_identifier
                st.session_state.chat_history = []  # Clear chat history for new selection

        # Initialize LLM and QA chain only if we have a valid vector store
        if st.session_state.vector_store is not None:
            llm_instance = AzureChatOpenAI(
                deployment_name='gpt-4o',
                model_name='gpt-4o',
                openai_api_version="2023-07-01-preview",
                openai_api_key=openai_api_client.api_key,
                openai_api_base=openai_api_client.api_base,
                openai_api_type="azure_ad",
                temperature=0.0,
                streaming=True
            )

            retriever = st.session_state.vector_store.as_retriever(search_kwargs={"k": 30})

            chat_prompt = PromptTemplate(
                template="""
                **Question:**
                {question}
                ---
                **Rules for the AI Assistant:**
                You are an AI assistant specializing in analyzing municipal bond documents from MuniOs.com...
                [Rest of the prompt template remains unchanged]
                """,
                input_variables=["question", "context"]
            )

            qa_chain = ConversationalRetrievalChain.from_llm(
                llm=llm_instance,
                retriever=retriever,
                combine_docs_chain_kwargs={"prompt": chat_prompt},
                return_source_documents=True,
            )

            # Display chat history
            for message in st.session_state.chat_history:
                with st.chat_message(message["role"], avatar=get_avatar(message["role"])):
                    st.markdown(message["content"])

            # Handle new user input
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

if __name__ == "__main__":
    main()