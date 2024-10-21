import os
import streamlit as st
import time
from tqdm import tqdm
from tenacity import retry, stop_after_attempt, wait_random_exponential, retry_if_exception_type

from langchain.document_loaders import PyMuPDFLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores.faiss import FAISS
from langchain.chat_models import AzureChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate

from quanthub.util import llm

# Set page config
st.set_page_config(layout="wide", page_title="PDF Chat App")

# Apply custom CSS for light blue theme and white background
st.markdown("""
    <style>
    .stApp {
        background-color: white;
    }
    .stButton>button {
        background-color: #E6F3FF;
        color: #2C3E50;
    }
    .stTextInput>div>div>input {
        background-color: #E6F3FF;
    }
    </style>
    """, unsafe_allow_html=True)

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

openai_api_client = llm.get_azure_openai_client()

@retry(
    reraise=True,
    stop=stop_after_attempt(6),
    wait=wait_random_exponential(min=1, max=60),
    retry=retry_if_exception_type(Exception),
)
def embed_texts(texts, embedding_model):
    return embedding_model.embed_documents(texts)

# Initialize the embedding model
embedding_model = OpenAIEmbeddings(
    deployment="text-embedding-ada-002",
    model="text-embedding-ada-002",
    openai_api_key=openai_api_client.api_key,
    openai_api_base=openai_api_client.api_base,
    openai_api_type="azure",
    chunk_size=1
)

def process_pdf(pdf_path):
    faiss_index_file = f"{pdf_path}_faiss_index"
    
    if os.path.exists(f"{faiss_index_file}.faiss"):
        vector_store = FAISS.load_local(faiss_index_file, embeddings=embedding_model)
        st.success("Loaded existing FAISS index.")
    else:
        loader = PyMuPDFLoader(pdf_path)
        documents = loader.load()
        st.info(f"Loaded {len(documents)} pages from the PDF.")
        
        with st.spinner("Processing PDF..."):
            batch_size = 10
            all_texts = []
            all_embeddings = []

            progress_bar = st.progress(0)
            for i in range(0, len(documents), batch_size):
                batch_docs = documents[i:i + batch_size]
                texts = [doc.page_content for doc in batch_docs]
                embeddings = embed_texts(texts, embedding_model)
                all_texts.extend(batch_docs)
                all_embeddings.extend(embeddings)
                progress = min((i + batch_size) / len(documents), 1.0)
                progress_bar.progress(progress)
                time.sleep(0.1)
            
            pure_texts = [page.page_content for page in all_texts]
            textual_embeddings = list(zip(pure_texts, all_embeddings))
            vector_store = FAISS.from_embeddings(textual_embeddings, embedding_model)
            vector_store.save_local(faiss_index_file)
            st.success(f"Processed PDF and saved FAISS index.")
    
    return vector_store

def get_pdf_files(directory):
    return [f for f in os.listdir(directory) if f.endswith('.pdf')]

def main():
    st.title("PDF Chat Application")

    pdf_directory = '/users/CFII_DataScience/USERs/SPTADM/Agents_Experiment/pdf_files'
    pdf_files = get_pdf_files(pdf_directory)

    selected_pdf = st.selectbox("Select a PDF", pdf_files)

    if selected_pdf:
        pdf_path = os.path.join(pdf_directory, selected_pdf)
        vector_store = process_pdf(pdf_path)

        llm_instance = AzureChatOpenAI(
            deployment_name=llm.GPT_4_OMNI_MODEL, 
            openai_api_version="2023-07-01-preview",
            openai_api_key=openai_api_client.api_key,
            openai_api_base=openai_api_client.api_base,
            openai_api_type="azure",
            temperature=0.0,
        )

        retriever = vector_store.as_retriever(search_kwargs={"k": 20})

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

        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm_instance,
            retriever=retriever,
            combine_docs_chain_kwargs={"prompt": chat_prompt},
            return_source_documents=True,
        )

        st.subheader("Chat with PDF")

        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if prompt := st.chat_input("Ask a question about the PDF"):
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = qa_chain({"question": prompt, "chat_history": [(msg["role"], msg["content"]) for msg in st.session_state.chat_history]})
                    st.markdown(response['answer'])
                    st.session_state.chat_history.append({"role": "assistant", "content": response['answer']})

if __name__ == "__main__":
    main()