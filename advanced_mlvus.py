# app/
# ├── config/
# │   ├── __init__.py
# │   └── settings.py
# ├── core/
# │   ├── __init__.py
# │   ├── document_store.py
# │   ├── question_manager.py
# │   └── vector_store.py
# ├── scripts/
# │   ├── __init__.py
# │   └── process_pdfs.py
# ├── ui/
# │   ├── __init__.py
# │   ├── components.py
# │   └── pages/
# │       └── main.py
# ├── requirements.txt
# └── main.py

# config/settings.py
from typing import Dict, Any
from pydantic import BaseSettings
from pathlib import Path

class Settings(BaseSettings):
    # OpenAI Configuration
    OPENAI_API_KEY: str
    EMBEDDING_MODEL: str = "text-embedding-3-small"
    LLM_MODEL: str = "gpt-4-turbo-preview"
    
    # Milvus Configuration
    MILVUS_HOST: str = "localhost"
    MILVUS_PORT: int = 19530
    COLLECTION_NAME: str = "pdf_embeddings"
    
    # Document Processing
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    
    # PDF Storage
    PDF_DIR: Path = Path("data/pdfs")
    PDF_METADATA_FILE: Path = Path("data/metadata.json")
    
    class Config:
        env_file = ".env"

settings = Settings()

# core/document_store.py
import json
from pathlib import Path
from typing import List, Dict
from datetime import datetime

class DocumentStore:
    def __init__(self, settings):
        self.settings = settings
        self.metadata_file = settings.PDF_METADATA_FILE
        self.pdf_dir = settings.PDF_DIR
        self._ensure_directories()
        self.metadata = self._load_metadata()
    
    def _ensure_directories(self):
        self.pdf_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_file.parent.mkdir(parents=True, exist_ok=True)
        if not self.metadata_file.exists():
            self._save_metadata({})
    
    def _load_metadata(self) -> Dict:
        if self.metadata_file.exists():
            return json.loads(self.metadata_file.read_text())
        return {}
    
    def _save_metadata(self, metadata: Dict):
        self.metadata_file.write_text(json.dumps(metadata, indent=2))
    
    def get_available_documents(self) -> List[Dict]:
        return [
            {
                "id": doc_id,
                "title": info["title"],
                "date_added": info["date_added"],
                "category": info["category"]
            }
            for doc_id, info in self.metadata.items()
        ]

# scripts/process_pdfs.py
from pathlib import Path
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from datetime import datetime
import shutil
import json

def process_new_pdf(pdf_path: Path, settings, document_store, vector_store):
    """
    Backend script to process new PDFs and store their vectors.
    This would be run separately from the main application.
    """
    # Generate unique ID and metadata
    doc_id = f"doc_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Load and process document
    loader = PyPDFLoader(str(pdf_path))
    documents = loader.load()
    
    # Split into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP
    )
    chunks = text_splitter.split_documents(documents)
    
    # Create embeddings and store in Milvus
    embeddings = OpenAIEmbeddings(model=settings.EMBEDDING_MODEL)
    vector_store.add_documents(chunks, doc_id=doc_id)
    
    # Store PDF and update metadata
    target_path = settings.PDF_DIR / f"{doc_id}.pdf"
    shutil.copy2(pdf_path, target_path)
    
    metadata = {
        "title": pdf_path.stem,
        "date_added": datetime.now().isoformat(),
        "category": "default",  # You can modify this based on your needs
        "path": str(target_path)
    }
    
    document_store.metadata[doc_id] = metadata
    document_store._save_metadata(document_store.metadata)
    
    return doc_id

# core/vector_store.py
from pymilvus import connections, Collection, FieldSchema, CollectionSchema
from langchain.vectorstores import Milvus

class MilvusStore:
    def __init__(self, settings):
        self.settings = settings
        self.setup_connection()
        
    def setup_connection(self):
        connections.connect(
            host=self.settings.MILVUS_HOST,
            port=self.settings.MILVUS_PORT
        )
    
    def search_documents(self, query: str, doc_ids: List[str], k: int = 4):
        vector_store = Milvus(
            collection_name=self.settings.COLLECTION_NAME,
            embedding_function=self.embeddings,
            connection_args={
                "host": self.settings.MILVUS_HOST,
                "port": self.settings.MILVUS_PORT
            }
        )
        return vector_store.similarity_search(
            query,
            k=k,
            filter={"doc_id": {"$in": doc_ids}}
        )

# ui/components.py
import streamlit as st
from typing import List, Dict

class DocumentSelector:
    @staticmethod
    def render(available_docs: List[Dict]):
        st.subheader("Select Documents for Comparison")
        
        # Group documents by category
        docs_by_category = {}
        for doc in available_docs:
            category = doc["category"]
            if category not in docs_by_category:
                docs_by_category[category] = []
            docs_by_category[category].append(doc)
        
        # Create multiselect for each category
        selected_docs = []
        for category, docs in docs_by_category.items():
            st.write(f"### {category.title()}")
            options = {doc["title"]: doc["id"] for doc in docs}
            selected = st.multiselect(
                "Select documents",
                options=list(options.keys()),
                key=f"select_{category}"
            )
            selected_docs.extend([options[title] for title in selected])
        
        return selected_docs

class QuestionManager:
    @staticmethod
    def render():
        st.subheader("Questions")
        
        # Initialize session state for questions
        if "questions" not in st.session_state:
            st.session_state.questions = []
        
        # Add new question
        col1, col2 = st.columns([3, 1])
        with col1:
            new_question = st.text_input("Enter new question")
        with col2:
            if st.button("Add") and new_question:
                st.session_state.questions.append(new_question)
        
        # Display and manage existing questions
        for i, question in enumerate(st.session_state.questions):
            col1, col2 = st.columns([4, 1])
            with col1:
                st.text(question)
            with col2:
                if st.button("Remove", key=f"remove_{i}"):
                    st.session_state.questions.pop(i)
                    st.rerun()
        
        return st.session_state.questions

# main.py
import streamlit as st
from config.settings import settings
from core.document_store import DocumentStore
from core.vector_store import MilvusStore
from core.question_manager import QuestionManager
from ui.components import DocumentSelector, QuestionManager

def main():
    st.title("PDF Comparative Analysis")
    
    # Initialize stores
    doc_store = DocumentStore(settings)
    vector_store = MilvusStore(settings)
    
    # Document selection
    available_docs = doc_store.get_available_documents()
    selected_docs = DocumentSelector.render(available_docs)
    
    # Question management
    questions = QuestionManager.render()
    
    # Generate comparison
    if st.button("Generate Comparison") and selected_docs and questions:
        with st.spinner("Generating comparison..."):
            question_manager = QuestionManager(settings, vector_store)
            answers = question_manager.process_questions(
                questions=questions,
                doc_ids=selected_docs
            )
            
            # Display results in a table
            st.subheader("Comparison Results")
            
            # Create headers
            headers = ["Question"] + [
                doc_store.metadata[doc_id]["title"] 
                for doc_id in selected_docs
            ]
            
            # Create rows
            rows = []
            for question in questions:
                row = [question]
                for doc_id in selected_docs:
                    row.append(answers.get((question, doc_id), "N/A"))
                rows.append(row)
            
            # Display table
            st.table([headers] + rows)

if __name__ == "__main__":
    main()