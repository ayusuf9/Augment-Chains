import os
import streamlit as st
import pandas as pd
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

# Load or initialize the results DataFrame
@st.cache_data
def load_results():
    if os.path.exists(RESULTS_FILE):
        return pd.read_csv(RESULTS_FILE, index_col=0)
    else:
        return pd.DataFrame()

results_df = load_results()

# Load FAISS indexes
@st.cache_resource
def load_vector_stores():
    vector_stores = {}
    for pdf_file in os.listdir(PDF_DIRECTORY):
        if pdf_file.endswith('.pdf'):
            index_file = f"{PDF_DIRECTORY}/faiss_index_{pdf_file[:-4]}"
            if os.path.exists(f"{index_file}.faiss"):
                vector_stores[pdf_file] = FAISS.load_local(index_file, embeddings=embedding_model)
    return vector_stores

vector_stores = load_vector_stores()

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

# Function to process a new query
def process_query(query, pdf_file):
    if query in results_df.index and pdf_file in results_df.columns:
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