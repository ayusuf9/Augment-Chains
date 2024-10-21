import os
import streamlit as st
import pandas as pd
import time
from tqdm import tqdm
from tenacity import retry, stop_after_attempt, wait_random_exponential, retry_if_exception_type
from langchain.document_loaders import DirectoryLoader, PyMuPDFLoader
from langchain.chat_models import AzureChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain import LLMChain
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings import OpenAIEmbeddings
from quanthub.util import llm

# Assuming the necessary Azure OpenAI setup is done in the quanthub.util.llm module
openai_api_client = llm.get_azure_openai_client()

# Directory and file paths
PDF_DIRECTORY = '/users/CFII_DataScience/USERs/SPTADM/Agents_Experiment/pdf_files/'
RESULTS_FILE = os.path.join(PDF_DIRECTORY, "results.csv")

# Initialize OpenAI embeddings and LLM
embedding_model = OpenAIEmbeddings(
    deployment_id="text-embedding-ada-002",
    model="text-embedding-ada-002",
    api_key=openai_api_client.api_key
)

llm_instance = AzureChatOpenAI(
    deployment_name=llm.GPT_4_OMNI_MODEL, 
    openai_api_version="2023-07-01-preview",
    openai_api_key=openai_api_client.api_key,
    openai_api_base=openai_api_client.api_base,
    openai_api_type=openai_api_client.api_type,
    temperature=0.0,
)

# Retry decorator for embedding
@retry(
    reraise=True,
    stop=stop_after_attempt(6),
    wait=wait_random_exponential(min=1, max=60),
    retry=retry_if_exception_type(openai_api_client.error.RateLimitError),
)
def embed_texts(texts, embedding_model):
    return embedding_model.embed_documents(texts)

# Function to process vectors and create FAISS index
def process_vectors(pdf_file):
    index_file = f"{PDF_DIRECTORY}/faiss_index_{pdf_file[:-4]}"
    if os.path.exists(f"{index_file}.faiss"):
        return FAISS.load_local(index_file, embeddings=embedding_model)
    
    loader = PyMuPDFLoader(os.path.join(PDF_DIRECTORY, pdf_file))
    documents = loader.load()
    st.write(f"Loaded {len(documents)} pages from {pdf_file}.")
    
    batch_size = 10
    all_texts = []
    all_embeddings = []

    for i in tqdm(range(0, len(documents), batch_size), desc=f"Embedding {pdf_file}"):
        batch_docs = documents[i:i + batch_size]
        texts = [doc.page_content for doc in batch_docs]
        embeddings = embed_texts(texts, embedding_model)
        all_texts.extend(batch_docs)
        all_embeddings.extend(embeddings)
        time.sleep(5)
    
    pure_texts = [page.page_content for page in all_texts]
    textual_embeddings = zip(pure_texts, all_embeddings)
    vector_store = FAISS.from_embeddings(textual_embeddings, embedding_model)
    st.write(f"Created FAISS index for {pdf_file}.")

    vector_store.save_local(index_file)
    st.write(f"Saved FAISS index for {pdf_file} to {index_file}.")
    return vector_store

# Load or create results DataFrame
@st.cache_data
def load_or_create_results(queries):
    if os.path.exists(RESULTS_FILE):
        df = pd.read_csv(RESULTS_FILE, index_col=0)
        # Add any new queries that aren't in the existing DataFrame
        new_queries = set(queries) - set(df.index)
        for query in new_queries:
            df.loc[query] = [None] * len(df.columns)
        return df
    else:
        pdf_files = [f[:-4] for f in os.listdir(PDF_DIRECTORY) if f.endswith('.pdf')]
        return pd.DataFrame(index=queries, columns=pdf_files)

# List of queries
list_of_queries = [
    "where is this property located?",
    "what county is this property located (use your knowledge of US geography for this)?",
    "What is the change in total income the last three months?",
    "What is the change in total income for the overall past four months?",
    "What is the occupancy rate most recently?",
    "What is the total number of units? ",
    "Is the occupancy rate trending positively, negatively or stable?",
    "What are total concessions as a percentage of total income income?",
    "Are total concessions trending positively, negatively, or stable?",
    "What is the change in total property expenses the last three months?",
    "What is the change in total property expenses the last 12 months?"
]

# Load or create results DataFrame
results_df = load_or_create_results(list_of_queries)

# Load or create vector stores
@st.cache_resource
def load_or_create_vector_stores():
    vector_stores = {}
    for pdf_file in os.listdir(PDF_DIRECTORY):
        if pdf_file.endswith('.pdf'):
            vector_stores[pdf_file] = process_vectors(pdf_file)
    return vector_stores

vector_stores = load_or_create_vector_stores()

# Prompts
chat_prompt = PromptTemplate(
    template="""
    Question: {question}
    
    Context (This is the data/information you should use in answering the question):
    {context}

    You are an AI assistant specializing in analyzing municipal bond prospectuses. 
    Provide accurate and insightful answers to questions, just like a financial professional working in a big financial company would.               
    
    When presented with extracted information from the municipal bond document and a question, focus on the following:
    1. Accuracy is key: Provide precise and correct information based on the prospectus content.
    2. Calculations matter: When asked to calculate, ensure your calculations are accurate and reflect a professional standard.
    3. Context is crucial: Frame your answers within the context of the specific bond issue and the overall municipal bond market.
        
    Answer: [Provide a clear, precise, and concise answer to the question, including only the most relevant information and numerical results.]
    """,
    input_variables=["question", "context"]
)

simplify_prompt = PromptTemplate(
    input_variables=["answer"],
    template="""
    Given the following answer, extract and return just the key point, which should be a concise summary such as a number, percentage, name, or brief phrase that directly conveys the main information. Do not include any additional explanation or context.

    Answer:
    {answer}

    Key point:"""
)

analyst_prompt = PromptTemplate(
    input_variables=["question", "table_data"],
    template="""
    You are an experienced municipal bond investment analyst. Analyze the following data from municipal bond prospectuses and provide insights based on the user's question. Use your expertise to interpret trends, identify potential risks or opportunities, and offer professional advice.

    Table Data:
    {table_data}

    User Question: {question}

    Provide a detailed analysis addressing the user's question, considering the following:
    1. Relevant financial metrics and their implications
    2. Trends or patterns in the data
    3. Potential risks or opportunities
    4. Comparisons between different properties or time periods, if applicable
    5. Professional advice or recommendations based on the analysis

    Your analysis should be thorough, insightful, and presented in a clear, professional manner.
    """
)

# Initialize chains
simplify_chain = LLMChain(llm=llm_instance, prompt=simplify_prompt)
analyst_chain = LLMChain(llm=llm_instance, prompt=analyst_prompt)

# Function to process a query
def process_query(query, pdf_file):
    if pd.notna(results_df.at[query, pdf_file]):
        return results_df.at[query, pdf_file]
    
    vector_store = vector_stores[pdf_file]
    retriever = vector_store.as_retriever(search_kwargs={"k": 20})
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm_instance,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": chat_prompt},
        verbose=False,
    )
    
    try:
        result = qa_chain.run(query)
        simplified_result = simplify_chain.run(answer=result)
        results_df.at[query, pdf_file] = simplified_result.strip()
        results_df.to_csv(RESULTS_FILE)
        return simplified_result.strip()
    except Exception as e:
        st.error(f"Error processing query: {str(e)}")
        return "Error"

# Streamlit app
st.title("Municipal Bond Analysis Dashboard")

# Display pre-processed table
st.header("Pre-Processed Data")
st.dataframe(results_df)

# Process all queries for all PDFs
if st.button("Process All Queries"):
    with st.spinner("Processing all queries..."):
        for pdf_file in vector_stores.keys():
            for query in list_of_queries:
                if pd.isna(results_df.at[query, pdf_file[:-4]]):
                    result = process_query(query, pdf_file)
                    st.write(f"Processed: {pdf_file} - {query}: {result}")
        st.success("All queries processed!")
    st.dataframe(results_df)

# Chat feature for PDF querying
st.header("PDF Query Chat")
pdf_file = st.selectbox("Select a PDF file", options=vector_stores.keys())
user_query = st.text_input("Enter your query about the PDF:")
if st.button("Submit Query"):
    with st.spinner("Processing query..."):
        result = process_query(user_query, pdf_file)
    st.write(f"Result: {result}")
    st.dataframe(results_df)  # Update the displayed table

# LLM-Driven Analyst Chat
st.header("Municipal Bond Analyst Chat")
analyst_query = st.text_area("Ask the Municipal Bond Analyst a question:")
if st.button("Get Analysis"):
    with st.spinner("Analyzing data..."):
        table_data = results_df.to_string()
        analysis = analyst_chain.run(question=analyst_query, table_data=table_data)
    st.write("Analyst's Response:")
    st.write(analysis)

# Run the Streamlit app
if __name__ == "__main__":
    st.set_page_config(page_title="Municipal Bond Analysis", layout="wide")