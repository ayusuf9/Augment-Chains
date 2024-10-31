import os
from pathlib import Path
import json
import streamlit as st
import time
import openai
from tenacity import retry, stop_after_attempt, wait_random_exponential, retry_if_exception_type
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import AzureChatOpenAI
from langchain.chains import ConversationalRetrievalChain, LLMChain
from langchain.prompts import PromptTemplate
from langchain.retrievers.multi_query import MultiQueryRetriever
from datetime import datetime
from tqdm import tqdm
from quanthub.util import llm
from pymilvus import (
    connections,
    utility,
    FieldSchema, CollectionSchema, DataType,
    Collection,
)
import uuid
import numpy as np
from langchain.schema import Document
from typing import List, Dict, Any

st.set_page_config(
    page_icon="💬",
    page_title="Muni Chat",
    initial_sidebar_state="collapsed",
    layout="wide"
)

# Milvus connection parameters
MILVUS_HOST = "localhost"  # Change this to your Milvus host
MILVUS_PORT = "19530"
COLLECTION_NAME = "muni_documents"
EMBEDDING_DIM = 1536  # Dimension for OpenAI embeddings

def connect_to_milvus():
    """Establish connection to Milvus"""
    try:
        connections.connect("default", host=MILVUS_HOST, port=MILVUS_PORT)
        st.success("Connected to Milvus successfully!")
    except Exception as e:
        st.error(f"Failed to connect to Milvus: {str(e)}")
        raise

def create_milvus_collection():
    """Create Milvus collection if it doesn't exist"""
    try:
        if utility.exists_collection(COLLECTION_NAME):
            collection = Collection(COLLECTION_NAME)
            collection.load()
            return collection
        
        fields = [
            FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, auto_id=False, max_length=100),
            FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="metadata", dtype=DataType.JSON),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=EMBEDDING_DIM)
        ]
        
        schema = CollectionSchema(fields, "Muni Documents Schema", enable_dynamic_field=True)
        collection = Collection(COLLECTION_NAME, schema, consistency_level="Strong")
        
        # Create index for vector field
        index_params = {
            "index_type": "IVF_FLAT",
            "metric_type": "IP",
            "params": {"nlist": 1536},
        }
        collection.create_index("embedding", index_params)
        collection.load()
        return collection
    except Exception as e:
        st.error(f"Failed to create Milvus collection: {str(e)}")
        raise

class MilvusRetriever:
    """Custom retriever class for Milvus"""
    def __init__(self, collection: Collection, embeddings: OpenAIEmbeddings):
        self.collection = collection
        self.embeddings = embeddings
    
    def get_relevant_documents(self, query: str) -> List[Document]:
        try:
            query_embedding = self.embeddings.embed_query(query)
            search_params = {
                "metric_type": "IP",
                "params": {"nprobe": 10},
            }
            
            results = self.collection.search(
                [query_embedding],
                "embedding",
                search_params,
                limit=5,
                output_fields=["content", "metadata"]
            )
            
            documents = []
            for hits in results:
                for hit in hits:
                    doc = Document(
                        page_content=hit.entity.get("content"),
                        metadata=hit.entity.get("metadata", {})
                    )
                    documents.append(doc)
            
            return documents
        except Exception as e:
            st.error(f"Error retrieving documents: {str(e)}")
            return []

def load_embeddings_model():
    """Load OpenAI embeddings model"""
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
    """Load Azure OpenAI LLM model"""
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

def initialize_session_state():
    """Initialize Streamlit session state variables"""
    session_vars = {
        'collection': None,
        'chat_history': [],
        'embeddings': None,
        'llm': None,
        'multi_query_retriever': None,
        'uploaded_file': None
    }
    
    for var, default in session_vars.items():
        if var not in st.session_state:
            st.session_state[var] = default
    
    if st.session_state.embeddings is None:
        st.session_state.embeddings = load_embeddings_model()
    if st.session_state.llm is None:
        st.session_state.llm = load_llm_model()
    if st.session_state.collection is None:
        connect_to_milvus()
        st.session_state.collection = create_milvus_collection()

def process_uploaded_pdf(pdf_file):
    """Process uploaded PDF file and store in Milvus"""
    temp_pdf_path = None
    try:
        # Create temporary file to save uploaded PDF
        temp_dir = "temp_pdfs"
        os.makedirs(temp_dir, exist_ok=True)
        temp_pdf_path = os.path.join(temp_dir, pdf_file.name)
        
        # Save uploaded file temporarily
        with open(temp_pdf_path, "wb") as f:
            f.write(pdf_file.getvalue())
        
        # Load and split the PDF
        loader = PyPDFLoader(temp_pdf_path)
        documents = loader.load()
        
        # Create text splitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        # Split documents
        splits = text_splitter.split_documents(documents)
        
        # Prepare data for Milvus
        data = []
        for doc in splits:
            embedding = st.session_state.embeddings.embed_documents([doc.page_content])[0]
            data.append({
                "id": str(uuid.uuid4()),
                "content": doc.page_content,
                "metadata": json.dumps(doc.metadata),  # Convert metadata to JSON string
                "embedding": embedding
            })
        
        # Insert into Milvus
        st.session_state.collection.insert(data)
        st.session_state.collection.flush()
        
        retriever = MilvusRetriever(st.session_state.collection, st.session_state.embeddings)
        return retriever
        
    except Exception as e:
        st.error(f"Error processing PDF: {str(e)}")
        return None
    finally:
        if temp_pdf_path and os.path.exists(temp_pdf_path):
            os.remove(temp_pdf_path)

def create_multi_query_retriever(milvus_retriever):
    """Create a Multi-Query Retriever using the Milvus retriever"""
    if not milvus_retriever or not st.session_state.llm:
        return None

    prompt_template = """You are an AI language model assistant specializing in municipal bonds. 
    Your task is to help generate alternative versions of the given question to improve document retrieval.
    Each version should focus on different aspects while maintaining the original intent.

    Original question: {question}

    Instructions:
    1. Rephrase using different financial terminology
    2. Break down complex questions into simpler components
    3. Include relevant municipal bond-specific context

    Please write 3 numbered alternative questions:"""

    question_generator_prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["question"]
    )

    llm_chain = LLMChain(llm=st.session_state.llm, prompt=question_generator_prompt)

    try:
        retriever = MultiQueryRetriever(
            retriever=milvus_retriever,
            llm_chain=llm_chain,
            parser_key="numbered_lines",
            num_queries=3
        )
        return retriever
    except Exception as e:
        st.error(f"Error creating Multi-Query Retriever: {str(e)}")
        return None

def get_avatar(role):
    """Get avatar emoji based on role"""
    return "🧑" if role == "user" else "🤖" if role == "assistant" else None

def setup_qa_chain(retriever):
    """Setup QA chain using cached LLM and retriever"""
    if not retriever or not st.session_state.llm:
        return None

    chat_prompt = PromptTemplate(
        template="""
        Question: {question}

        Context: {context}

        Instructions for providing an answer:
        1. Use only the information provided in the context
        2. Be precise with numerical data and financial terms
        3. Show calculation steps if needed
        4. Use professional financial terminology
        5. Be clear and concise
        6. If the information is not in the context, say so

        Answer the question based on the context provided:""",
        input_variables=["question", "context"]
    )

    return ConversationalRetrievalChain.from_llm(
        llm=st.session_state.llm,
        retriever=retriever,
        combine_docs_chain_kwargs={"prompt": chat_prompt},
        return_source_documents=True,
    )

def handle_user_input(qa_chain, prompt):
    """Handle user input and generate response"""
    try:
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar=get_avatar("user")):
            st.markdown(prompt)

        with st.chat_message("assistant", avatar=get_avatar("assistant")):
            response_placeholder = st.empty()
            with st.spinner("Generating response..."):
                response = qa_chain({
                    "question": prompt,
                    "chat_history": [(msg["role"], msg["content"]) 
                                   for msg in st.session_state.chat_history]
                })
                response_placeholder.markdown(response['answer'])
                st.session_state.chat_history.append(
                    {"role": "assistant", "content": response['answer']}
                )

    except Exception as e:
        st.error(f"Error generating response: {str(e)}")
        st.error("Please try rephrasing your question or upload a new document.")

def main():
    """Main application function"""
    initialize_session_state()
    
    st.title("📚 Municipal Bond Document Analysis")
    
    # Create two columns for layout
    col1, col2 = st.columns([2, 3])
    
    with col1:
        st.info("📤 Upload a PDF document to begin analysis")
        uploaded_file = st.file_uploader(
            "Choose a PDF file",
            type="pdf",
            help="Upload a municipal bond document (PDF format)",
            key="pdf_uploader"
        )
        
        if uploaded_file:
            st.success(f"📄 Selected file: {uploaded_file.name}")
    
    with col2:
        if uploaded_file is not None and uploaded_file != st.session_state.uploaded_file:
            st.session_state.uploaded_file = uploaded_file
            
            with st.spinner("🔄 Processing document..."):
                retriever = process_uploaded_pdf(uploaded_file)
                if retriever:
                    st.session_state.multi_query_retriever = create_multi_query_retriever(retriever)
                    st.session_state.chat_history = []
                    st.success("✅ Document processed successfully!")
        
        if st.session_state.multi_query_retriever is not None:
            st.divider()
            st.subheader("💬 Chat with your document")
            
            qa_chain = setup_qa_chain(st.session_state.multi_query_retriever)
            if qa_chain is None:
                st.error("⚠️ Error setting up the chat system. Please try again.")
                return

            # Display chat history
            for message in st.session_state.chat_history:
                with st.chat_message(message["role"], avatar=get_avatar(message["role"])):
                    st.markdown(message["content"])
            
            # Chat input
            if prompt := st.chat_input("Ask a question about the document..."):
                handle_user_input(qa_chain, prompt)
        else:
            st.info("👆 Please upload a document to start the analysis")

    # Add footer with version info
    st.markdown("---")
    st.caption("Municipal Bond Document Analysis System v1.0")

if __name__ == "__main__":
    main()