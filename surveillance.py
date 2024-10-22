from tenacity import retry, stop_after_attempt, wait_random_exponential, retry_if_exception_type
from langchain.document_loaders import PyMuPDFLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores.faiss import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain import LLMChain
from quanthub.util import llm2
from langchain.chat_models import AzureChatOpenAI
import pandas as pd
import os
import streamlit as st

openai_api_client = llm2.get_llm_client() 

# Initialize embeddings
embedding_model = OpenAIEmbeddings(
    deployment="text-embedding-ada-002",
    model="text-embedding-ada-002",
    openai_api_key=openai_api_client.api_key,
    openai_api_base=openai_api_client.api_base,
    openai_api_type=openai_api_client.api_type,
    chunk_size=1
)

# Paths
pdf_directory = '/Users/sptadm/Documents/muni_table_app/pdf_files/'
output_file = os.path.join(pdf_directory, "results.csv")

# Load existing results if available
if 'df' not in st.session_state:
    if os.path.exists(output_file):
        st.session_state.df = pd.read_csv(output_file, index_col=0)
        st.write("## Loaded Existing Results:")
        st.dataframe(st.session_state.df)
    else:
        st.session_state.df = pd.DataFrame()
        st.write("## No existing results found. Starting fresh.")

# Function definitions remain the same
@retry(
    reraise=True,
    stop=stop_after_attempt(6),
    wait=wait_random_exponential(min=1, max=60),
    retry=retry_if_exception_type(Exception),
)
def embed_texts(texts, embedding_model):
    return embedding_model.embed_documents(texts)

@st.cache_resource
def get_vector_stores(pdf_directory):
    # Only process PDFs if we need to handle a new query
    return {}  # Return empty dict initially - we'll load stores only when needed

# Initialize the LLM instance
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

# Prompt templates remain the same
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
    input_variables=['question', 'context']
)

simplify_prompt = PromptTemplate(
    template="""
Given the following answer, extract and return only the key point. The key point should be a concise summary that directly conveys the main information, such as:

- A specific number along with its unit or currency symbol (e.g., '180 units', '$65,338.68').
- A percentage (e.g., '94.63%').
- A name or proper noun (e.g., 'Waterscape Apartments', 'Riverside County, California').
- A brief descriptive phrase that includes necessary qualifiers (e.g., 'trending positively', 'decreased by $59,800.84', 'increase of $1,243.72').

Do not include any additional explanation, context, or restatement of the answer. Provide only the key point as it directly relates to the main information.

Now, given the following answer, extract and return just the key point.

Answer:
{answer}

Key point:
""",
    input_variables=['answer']
)

simplify_chain = LLMChain(
    llm=llm_instance,
    prompt=simplify_prompt
)

# Query input and processing
query = st.text_input("Enter your query:")
if st.button("Submit"):
    if query.strip() == '':
        st.warning("Please enter a query.")
    else:
        new_query = query.strip()
        if new_query in st.session_state.df.index:
            st.warning("Query already exists in the results.")
        else:
            st.write("Processing query...")
            # Initialize vector stores only when needed
            vector_stores = get_vector_stores(pdf_directory)
            if not vector_stores:
                pdf_files = [f for f in os.listdir(pdf_directory) if f.endswith('.pdf')]
                faiss_index_file = os.path.join(pdf_directory, "faiss_index")
                for pdf_file in pdf_files:
                    index_file = f"{faiss_index_file}_{pdf_file[:-4]}"
                    if os.path.exists(f"{index_file}.faiss") and os.path.exists(f"{index_file}.pkl"):
                        vector_stores[pdf_file[:-4]] = FAISS.load_local(index_file, embeddings=embedding_model)
                    else:
                        loader = PyMuPDFLoader(os.path.join(pdf_directory, pdf_file))
                        documents = loader.load()
                        vector_store = FAISS.from_documents(documents, embedding_model)
                        vector_store.save_local(index_file)
                        vector_stores[pdf_file[:-4]] = vector_store
            
            # Process query with progress bar
            progress_bar = st.progress(0)
            total_files = len(vector_stores)
            results = {}
            for idx, (pdf_name, vector_store) in enumerate(vector_stores.items()):
                try:
                    retriever = vector_store.as_retriever(search_kwargs={"k": 20})
                    qa_chain = RetrievalQA.from_chain_type(
                        llm=llm_instance,
                        chain_type="stuff",
                        retriever=retriever,
                        chain_type_kwargs={"prompt": chat_prompt},
                        verbose=False,
                    )
                    result = qa_chain.run(new_query)
                    simplified_result = simplify_chain.run(answer=result)
                    results[pdf_name] = simplified_result.strip()
                except Exception as e:
                    st.write(f"Error processing for {pdf_name}: {str(e)}")
                    results[pdf_name] = "Error"
                progress_bar.progress((idx + 1) / total_files)
            
            # Update and save results
            for pdf_name, result in results.items():
                st.session_state.df.loc[new_query, pdf_name] = result
            st.session_state.df.to_csv(output_file)
            st.success("Query processed and results updated.")
            st.write("## Updated Results:")
            st.dataframe(st.session_state.df)