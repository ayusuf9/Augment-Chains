






import os
import warnings
import time
import pandas as pd
import streamlit as st

from streamlit_navigation_bar import st_navbar
from streamlit_extras.switch_page_button import switch_page
from tqdm import tqdm
import openai

import faiss
from tenacity import retry, stop_after_attempt, wait_random_exponential, retry_if_exception_type

from langchain.document_loaders import DirectoryLoader, PyMuPDFLoader, PyPDFLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores.faiss import FAISS
from langchain.chat_models import AzureChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain import LLMChain

import io
from io import StringIO
import os
from quanthub.util import s3_appdata_client_qh

from quanthub.util import llm

os.environ["ENV_FOR_DYNACONF"] = os.environ['ENV_FOR_DYNACONF']
os.environ["QUANTAMENTAL_API_KEY"] = os.environ['QUANTAMENTAL_API_KEY']

pdf_directory = os.environ['PDF_MUNI_QA_TABLE']


openai_api_client = llm.get_llm_client() 

st.set_page_config(layout="wide", 
                   page_icon="🏢",
                   page_title="Tables",
                   initial_sidebar_state="collapsed")

# st.warning("Please note that this is a Propotype and not the full-fledged application.")

styles = {
  "nav": {
      "background-color": "rgb(54, 69, 79)", #"rgb(115, 147, 179)",  
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
      "font-weight": "bold" # 'Gill Sans'
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
page = st_navbar(["Home", "Chat", "Tables"], selected="Tables", styles=styles)

if page=="Home":
    switch_page("streamlit_app")
if page=="Chat":
    switch_page("chat")

st.warning("Please note that this is a Propotype and not the full-fledged application.")



def process_pdf_from_s3(pdf_key):
    base_path = "appdata/muni_pdfs/pdfs_qa"

    filename = os.path.splitext(os.path.basename(pdf_key))[0]
    index_key = f"{base_path}/{filename}_faiss_index/index.faiss"
    store_key = f"{base_path}/{filename}_faiss_index/index.pkl"

    try:
        try:
            with st.spinner("Loading existing index from S3..."):
                index_response = s3_appdata_client_qh.get_object(key=index_key)
                store_response = s3_appdata_client_qh.get_object(key=store_key)
                
                index_data = BytesIO(index_response.read())
                store_data = BytesIO(store_response.read())
                
                index = faiss.read_index(index_data)
                store = pickle.loads(store_data.read())
                
                vector_store = FAISS(
                    embeddings=get_embedding_model(),
                    index=index,
                    docstore=store,
                    allow_dangerous_deserialization=True
                )
                st.success("Successfully loaded existing FAISS index.")
                return vector_store

        except Exception as e:
            logging.info(f"No existing index found or error loading index: {str(e)}")
            st.info("Creating new index from PDF...")

        with st.spinner("Downloading PDF from S3..."):
            pdf_response = s3_appdata_client_qh.get_object(key=pdf_key)
            pdf_data = pdf_response.read()

        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp_pdf:
            temp_pdf.write(pdf_data)
            temp_pdf_path = temp_pdf.name

        try:
            with st.spinner("Processing PDF..."):
                loader = PyPDFLoader(temp_pdf_path)
                documents = loader.load()
                
                batch_size = 10
                all_texts = []
                all_embeddings = []
                total_docs = len(documents)

                progress_bar = st.progress(0)
                for i in range(0, total_docs, batch_size):
                    batch_docs = documents[i:i + batch_size]
                    texts = [doc.page_content for doc in batch_docs]
                    
                    try:
                        embeddings = embed_texts(texts, get_embedding_model())
                        all_texts.extend(batch_docs)
                        all_embeddings.extend(embeddings)
                    except Exception as e:
                        st.error(f"Error generating embeddings for batch {i//batch_size}: {str(e)}")
                        continue

                    progress = min((i + batch_size) / total_docs, 1.0)
                    progress_bar.progress(progress)

                # Create vector store
                pure_texts = [page.page_content for page in all_texts]
                textual_embeddings = list(zip(pure_texts, all_embeddings))
                vector_store = FAISS.from_embeddings(textual_embeddings, get_embedding_model())

                return vector_store

        finally:
            if os.path.exists(temp_pdf_path):
                os.unlink(temp_pdf_path)

    except Exception as e:
        st.error(f"Error processing PDF: {str(e)}")
        logging.error(f"Error processing PDF: {str(e)}", exc_info=True)
        return None


@retry(
    reraise=True,
    stop=stop_after_attempt(6),
    wait=wait_random_exponential(min=1, max=60),
    retry=retry_if_exception_type(Exception),
)
def embed_texts(texts, embedding_model):
    return embedding_model.embed_documents(texts)


if 'custom_questions' not in st.session_state:
  st.session_state.custom_questions = []
if 'qa_results' not in st.session_state:
  st.session_state.qa_results = pd.DataFrame()

# def get_pdf_files_from_s3():
#   """List PDF files from S3 using s3_appdata_client"""
#   try:
#       response = s3_appdata_client_qh.list_objects(
#           key=pdf_directory
#       )
#       return [obj for obj in response if obj.lower().endswith('.pdf')]
#   except Exception as e:
#       st.error(f"Error listing S3 files: {str(e)}")
#       return []

def initialize_models():
  """Initialize the embedding and LLM models"""
  openai_api_client = llm.get_llm_client() 
  
  embedding_model = OpenAIEmbeddings(
      deployment_id="text-embedding-ada-002",
      model="text-embedding-ada-002",
      api_key=openai_api_client.api_key
  )
  
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
  
  return embedding_model, llm_instance

def create_new_index(pdf_key, index_file, embedding_model):
  """Create new FAISS index for PDF from S3"""
  with st.spinner(f"Creating new index for {os.path.basename(pdf_key)}..."):
      try:
          # Get PDF from S3
          obj = s3_appdata_client_qh.get_object(pdf_key)
          pdf_data = obj.read()
          pdf_file = io.BytesIO(pdf_data)
          
          loader = PyPDFLoader(pdf_file)
          documents = loader.load()
          
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
              time.sleep(5)
          progress_bar.empty()
          
          pure_texts = [page.page_content for page in all_texts]
          textual_embeddings = zip(pure_texts, all_embeddings)
          vector_store = FAISS.from_embeddings(textual_embeddings, embedding_model)
          vector_store.save_local(index_file)
          return vector_store
          
      except Exception as e:
          st.error(f"Error creating index: {str(e)}")
          return None

def load_vector_stores(selected_pdfs, embedding_model):
  """Load or create vector stores for selected PDFs"""
  vector_stores = {}
  
  for pdf_key in selected_pdfs:
      filename = os.path.basename(pdf_key)
      index_folder = f"faiss_index_{os.path.splitext(filename)[0]}"
      index_file = os.path.join(index_folder, "index")
      
      if os.path.exists(index_folder) and os.path.exists(f"{index_file}.faiss"):
          try:
              vector_stores[pdf_key] = FAISS.load_local(
                  index_file, 
                  embeddings=embedding_model, 
                  allow_dangerous_deserialization=True
              )
          except Exception as e:
              st.warning(f"Error loading index for {filename}, creating new one...")
              vector_stores[pdf_key] = create_new_index(pdf_key, index_file, embedding_model)
      else:
          vector_stores[pdf_key] = create_new_index(pdf_key, index_file, embedding_model)
  
  return vector_stores


st.markdown("""
  <style>
  span[data-baseweb="tag"] {
      background-color: rgb(54, 69, 79) !important;
  }
  </style>
  """, unsafe_allow_html=True)

# Get PDF files and create multiselect
pdf_files = get_pdf_files_from_s3()
selected_pdfs = st.multiselect("Select PDF files", pdf_files)

# Question input
new_question = st.text_input("Enter your question:")
if st.button("Add Question"):
  if new_question and new_question not in st.session_state.custom_questions:
      st.session_state.custom_questions.append(new_question)
      st.success("Question added successfully!")


def process_queries(vector_stores, llm_instance, queries):
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

            
            **Context (Based on the rules, provide an answer to the information to the following information):**

            {context}

            **Your response should be structured as follows:**

            **Answer:**

            [Provide a clear, precise, and concise answer to the question, including only the most relevant information and numerical results.]
            """,
            input_variables=["question", "context"]
        )


    simplify_prompt = PromptTemplate(
    input_variables=["answer"],
    template="""
        Given the following answer, extract and return only the key point. The key point should be a concise summary that directly conveys the main information, such as:

        - A specific number along with its unit or currency symbol (e.g., '180 units', '$65,338.68').
        - A percentage (e.g., '94.63%').
        - A name or proper noun (e.g., 'Waterscape Apartments', 'Riverside County, California').
        - A brief descriptive phrase that includes necessary qualifiers (e.g., 'trending positively', 'decreased by $59,800.84', 'increase of $1,243.72').

        Do not include any additional explanation, context, or restatement of the answer. Provide only the key point as it directly relates to the main information.

        Examples:

        1. Answer: The property is located at Waterscape Apartments, identified by the code "cawater2."

        Key point: Waterscape Apartments

        2. Answer: The property, Waterscape Apartments, is located in Riverside County, California.

        Key point: Riverside County, California

        3. Answer: The change in total income over the last three months is $65,338.68.

        Key point: $65,338.68

        4. Answer: The most recent occupancy rate is 94.63%.

        Key point: 94.63%

        5. Answer: The total number of units is 180.

        Key point: 180 units

        6. Answer: The occupancy rate is trending positively.

        Key point: trending positively

        7. Answer: The total concessions as a percentage of total income are approximately -1.05%.

        Key point: -1.05%

        8. Answer: The total property expenses decreased by $59,800.84 over the last three months.

        Key point: decreased by $59,800.84

        9. Answer: The change in total property expenses over the last 12 months is an increase of $1,243.72.

        Key point: increase of $1,243.72

        Now, given the following answer, extract and return just the key point.

        Answer:
        {answer}

        Key point:
        """
        )
  
    simplify_chain = LLMChain(llm=llm_instance, prompt=simplify_prompt)
    results = pd.DataFrame(index=queries, columns=[pdf[:-4] for pdf in vector_stores.keys()])
    
    progress_bar = st.progress(0)
    progress_text = st.empty()
    
    total_iterations = len(queries) * len(vector_stores)
    current_iteration = 0
    
    for pdf_file, vector_store in vector_stores.items():
        retriever = vector_store.as_retriever(search_kwargs={"k": 20})
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm_instance,
            chain_type="stuff",
            retriever=retriever,
            chain_type_kwargs={"prompt": chat_prompt},
            verbose=False,
        )
        
        for query in queries:
            try:
                progress_text.text(f"Processing: {pdf_file} - {query}")
                result = qa_chain.run(query)
                simplified_result = simplify_chain.run(answer=result)
                results.at[query, pdf_file[:-4]] = simplified_result.strip()
            except Exception as e:
                st.error(f"Error processing query '{query}' for file {pdf_file}: {str(e)}")
                results.at[query, pdf_file[:-4]] = "Error"
            
            current_iteration += 1
            progress_bar.progress(current_iteration / total_iterations)
    
    progress_bar.empty()
    progress_text.empty()
    return results

list_of_queries = [
    "where is this property located?",
    "what is the total number of units?",
    "What is the total income?"]

if selected_pdfs:
    #st.header("Analysis Results")
    embedding_model, llm_instance = initialize_models()
    
    vector_stores = load_vector_stores(selected_pdfs, embedding_model)
    
    all_questions = list_of_queries + st.session_state.custom_questions
    
    results_df = process_queries(vector_stores, llm_instance, all_questions)
    st.session_state.qa_results = results_df


    styled_df = results_df.style.set_properties(**{
        'text-align': 'left',
        'font-size': '16px',  # Increased font size
        'border-color': 'lightgrey',
        'border-style': 'solid',
        'border-width': '1px',
        'padding': '8px'  # Increased padding
    }).set_table_styles([
        {'selector': 'th', 'props': [
            ('background-color', '#f0f2f6'),
            ('color', 'black'),
            ('font-weight', 'bold'),
            ('text-align', 'left'),
            ('padding', '8px'),  # Increased padding
            ('font-size', '18px') 
        ]},
        {'selector': 'tr:nth-of-type(even)', 'props': [
            ('background-color', '#f9f9f9')
        ]},
        # Style for the 'Prompt' `column
        {'selector': 'td:first-child', 'props': [
            ('font-weight', 'bold'),
            ('background-color', '#f0f2f6'),
            ('font-size', '17px')
        ]},
        # Center the table
        {'selector': 'table', 'props': [
            ('margin-left', 'auto'),
            ('margin-right', 'auto')
        ]}
    ])

    
    #st.dataframe(styled_df)
    st.write(styled_df.to_html(escape=False, index=False), unsafe_allow_html=True)
    
    if st.button("Export Results to CSV"):
        csv = results_df.to_csv().encode('utf-8')
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name="analysis_results.csv",
            mime="text/csv"
        )
# else:
#     st.info("")

# st.markdown("""
#     <style>
#     .stButton>button {
#         width: 100%;
#     }
#     .stDataFrame {
#         width: 100%;
#     }
#     </style>
#     """, unsafe_allow_html=True)

