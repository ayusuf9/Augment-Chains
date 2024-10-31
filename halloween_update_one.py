import os
from pathlib import Path
import json
import streamlit as st
import time
import openai
from typing import List, Dict, Any, Optional, Tuple

from tenacity import retry, stop_after_attempt, wait_random_exponential, retry_if_exception_type
from streamlit_navigation_bar import st_navbar
from streamlit_extras.switch_page_button import switch_page
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores.faiss import FAISS
from langchain.chat_models import AzureChatOpenAI
from langchain.chains import ConversationalRetrievalChain, LLMChain
from langchain.prompts import PromptTemplate
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.schema import Document
from datetime import datetime
from tqdm import tqdm
from quanthub.util import llm2

# Streamlit configuration
st.set_page_config(
    page_icon="💬",
    page_title="Muni Chat",
    initial_sidebar_state="collapsed"
)

# Styling
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

def initialize_session_state() -> None:
    """Initialize all session state variables."""
    session_vars = {
        'vector_store': None,
        'chat_history': [],
        'embeddings': None,
        'llm': None,
        'retriever': None,
        'uploaded_file': None,
        'document_chunks': [],
        'last_error': None
    }
    
    for var, default in session_vars.items():
        if var not in st.session_state:
            st.session_state[var] = default
    
    if st.session_state.embeddings is None:
        st.session_state.embeddings = load_embeddings_model()
    if st.session_state.llm is None:
        st.session_state.llm = load_llm_model()

def load_llm_model() -> Optional[AzureChatOpenAI]:
    """Load and configure the Azure OpenAI model."""
    try:
        openai_api_client = llm2.get_azure_openai_client()
        llm_model = AzureChatOpenAI(
            deployment_name=llm2.GPT_4_OMNI_MODEL,
            model_name='gpt-4o',
            openai_api_version="2024-02-01",
            openai_api_key=openai_api_client.api_key,
            openai_api_base=openai_api_client.api_base,
            openai_api_type="azure_ad",
            temperature=0,
            request_timeout=120,
            max_retries=3
        )
        return llm_model
    except Exception as e:
        st.error(f"Failed to load LLM model: {str(e)}")
        return None

def load_embeddings_model() -> Optional[OpenAIEmbeddings]:
    """Load and configure the embeddings model."""
    try:
        openai_api_client = llm2.get_azure_openai_client()
        embeddings = OpenAIEmbeddings(
            deployment='text-embedding-3-small',
            model='text-embedding-3-small',
            openai_api_key=openai_api_client.api_key,
            openai_api_base=openai_api_client.api_base,
            openai_api_type=openai_api_client.api_type,
            chunk_size=16
        )
        return embeddings
    except Exception as e:
        st.error(f"Failed to load embedding model: {str(e)}")
        return None

@retry(
    reraise=True,
    stop=stop_after_attempt(3),
    wait=wait_random_exponential(min=1, max=20),
    retry=retry_if_exception_type((openai.error.APIError, openai.error.ServiceUnavailableError))
)
def embed_texts(texts: List[str], embedding_model: OpenAIEmbeddings) -> List[List[float]]:
    """Embed texts with retry logic."""
    return embedding_model.embed_documents(texts)

def chunk_documents(documents: List[Document]) -> List[Document]:
    """
    Split documents into optimally sized chunks with metadata.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        length_function=len,
        separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
        is_separator_regex=False
    )
    
    chunks = text_splitter.split_documents(documents)
    
    # Enhance chunks with metadata
    for i, chunk in enumerate(chunks):
        chunk.metadata.update({
            'chunk_id': i,
            'total_chunks': len(chunks),
            'source_page': chunk.metadata.get('page', 0),
            'chunk_size': len(chunk.page_content),
            'timestamp': datetime.now().isoformat()
        })
    
    return chunks

def create_enhanced_retriever(vector_store: FAISS) -> Optional[ContextualCompressionRetriever]:
    """
    Create an enhanced retriever with contextual compression.
    """
    if not vector_store or not st.session_state.llm:
        return None

    # Base retriever configuration
    base_retriever = vector_store.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={
            "k": 15,
            "score_threshold": 0.5,
            "fetch_k": 20
        }
    )

    # Configure document compressor
    compressor_prompt = PromptTemplate(
        template="""Given the following question and context, identify and extract only the most relevant information for answering the question.
        Focus on financial and municipal bond-related details that directly address the question.
        
        Question: {question}
        Context: {context}
        
        Relevant Information:""",
        input_variables=["question", "context"]
    )

    compressor = LLMChainExtractor.from_llm(
        llm=st.session_state.llm,
        prompt=compressor_prompt
    )

    # Create and return the compression retriever
    return ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=base_retriever
    )

def process_uploaded_pdf(pdf_file) -> Optional[FAISS]:
    """
    Process uploaded PDF file and create vector store.
    """
    try:
        with st.spinner("Processing your file"):
            # Setup directories
            temp_dir = Path("temp_pdfs")
            temp_dir.mkdir(exist_ok=True)
            temp_pdf_path = temp_dir / pdf_file.name
            
            # Save uploaded file
            temp_pdf_path.write_bytes(pdf_file.getvalue())
            
            # Load document
            loader = PyPDFLoader(str(temp_pdf_path))
            documents = loader.load()
            
            # Process documents
            chunks = chunk_documents(documents)
            st.session_state.document_chunks = chunks
            
            # Create embeddings with progress tracking
            with st.spinner(text="Creating embeddings..."):
                batch_size = 5
                progress_bar = st.progress(0)
                
                all_embeddings = []
                for i in range(0, len(chunks), batch_size):
                    batch = chunks[i:i + batch_size]
                    texts = [doc.page_content for doc in batch]
                    
                    try:
                        embeddings = embed_texts(texts, st.session_state.embeddings)
                        all_embeddings.extend(embeddings)
                    except Exception as e:
                        st.error(f"Error in batch {i//batch_size + 1}: {str(e)}")
                        continue
                    
                    progress = min((i + batch_size) / len(chunks), 1.0)
                    progress_bar.progress(progress)
                
                # Create vector store
                vector_store = FAISS.from_documents(
                    chunks,
                    st.session_state.embeddings,
                )
                
                # Clean up
                temp_pdf_path.unlink()
                
                return vector_store

    except Exception as e:
        st.error(f"Error processing PDF: {str(e)}")
        st.session_state.last_error = str(e)
        return None

def setup_qa_chain() -> Optional[ConversationalRetrievalChain]:
    """
    Setup the QA chain with enhanced retrieval and prompting.
    """
    if not st.session_state.vector_store or not st.session_state.llm:
        return None

    retriever = create_enhanced_retriever(st.session_state.vector_store)
    if not retriever:
        st.error("Failed to create retriever")
        return None

    st.session_state.retriever = retriever

    chat_prompt = PromptTemplate(
        template="""You are a municipal bond expert assistant. Use the following context and conversation history to provide accurate, detailed answers about municipal bonds and related financial information.

Context: {context}

Question: {question}

Chat History: {chat_history}

Instructions:
1. Answer only based on the provided context
2. If information is unclear or missing, acknowledge this explicitly
3. Include specific quotes when relevant
4. Use proper financial terminology
5. Format numbers and percentages consistently
6. Maintain professional tone

Response:""",
        input_variables=["context", "question", "chat_history"]
    )

    return ConversationalRetrievalChain.from_llm(
        llm=st.session_state.llm,
        retriever=retriever,
        combine_docs_chain_kwargs={"prompt": chat_prompt},
        return_source_documents=True,
        verbose=True
    )

def get_avatar(role: str) -> str:
    """Get avatar icon based on role."""
    return ":material/for_you:" if role == "user" else ":material/robot_2:"

def handle_user_input(qa_chain: ConversationalRetrievalChain, prompt: str) -> None:
    """
    Handle user input and generate response.
    """
    try:
        # Add user message to history
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user", avatar=get_avatar("user")):
            st.markdown(prompt)

        # Generate and display response
        with st.chat_message("assistant", avatar=get_avatar("assistant")):
            with st.spinner("Analyzing documents and generating response..."):
                # Format chat history
                formatted_history = [(msg["role"], msg["content"]) 
                                   for msg in st.session_state.chat_history[:-1]]
                
                # Get response
                response = qa_chain({
                    "question": prompt,
                    "chat_history": formatted_history
                })
                
                # Display main response
                st.markdown(response['answer'])
                
                # Show source documents in expandable section
                with st.expander("View Source Documents"):
                    for i, doc in enumerate(response['source_documents']):
                        st.markdown(f"""
                        **Source {i+1}**
                        - Page: {doc.metadata.get('page', 'unknown')}
                        - Chunk ID: {doc.metadata.get('chunk_id', 'unknown')}
                        - Content:
                        ```
                        {doc.page_content}
                        ```
                        """)
                
                # Add assistant response to history
                st.session_state.chat_history.append(
                    {"role": "assistant", "content": response['answer']}
                )

    except Exception as e:
        st.error(f"Error generating response: {str(e)}")
        st.session_state.last_error = str(e)

def main():
    """
    Main application function.
    """
    initialize_session_state()
    
    st.warning("Please note that this is a Prototype and not the full-fledged application.")

    # File upload
    uploaded_file = st.file_uploader(
        "Upload a PDF document",
        type="pdf",
        help="Upload a PDF file to start chatting"
    )
    
    # Process new file
    if uploaded_file is not None and uploaded_file != st.session_state.uploaded_file:
        st.session_state.uploaded_file = uploaded_file
        vector_store = process_uploaded_pdf(uploaded_file)
        
        if vector_store:
            st.session_state.vector_store = vector_store
            st.session_state.chat_history = []
            st.success(f"Successfully processed {uploaded_file.name}")
    
    # Chat interface
    if st.session_state.vector_store is not None:
        st.divider()
        
        qa_chain = setup_qa_chain()
        if qa_chain is None:
            st.error("Failed to setup QA chain. Please refresh the page.")
            return

        # Display chat history
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"], avatar=get_avatar(message["role"])):
                st.markdown(message["content"])
        
        # Handle new input
        if prompt := st.chat_input("What is your question?"):
            handle_user_input(qa_chain, prompt)
    else:
        st.info("Please upload a PDF document to start chatting.")

if __name__ == "__main__":
    main()