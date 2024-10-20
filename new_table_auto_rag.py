import os
import warnings
import time
import pandas as pd
from tqdm import tqdm
from tenacity import retry, stop_after_attempt, wait_random_exponential, retry_if_exception_type

from langchain.document_loaders import DirectoryLoader, PyMuPDFLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores.faiss import FAISS
from langchain.chat_models import AzureChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain import LLMChain

from quanthub.util import llm

warnings.filterwarnings('ignore')

openai_api_client = llm.get_azure_openai_client()

@retry(
    reraise=True,
    stop=stop_after_attempt(6),
    wait=wait_random_exponential(min=1, max=60),
    retry=retry_if_exception_type(openai_api_client.error.RateLimitError),
)
def embed_texts(texts, embedding_model):
    return embedding_model.embed_documents(texts)

embedding_model = OpenAIEmbeddings(
    deployment_id="text-embedding-ada-002",
    model="text-embedding-ada-002",
    api_key=openai_api_client.api_key
)

# Directory path
pdf_directory = '/users/CFII_DataScience/USERs/SPTADM/Agents_Experiment/pdf_files/'
faiss_index_file = os.path.join(pdf_directory, "faiss_index")

# Get list of PDF files
pdf_files = [f for f in os.listdir(pdf_directory) if f.endswith('.pdf')]

# Create or load FAISS index for each PDF
vector_stores = {}
for pdf_file in pdf_files:
    index_file = f"{faiss_index_file}_{pdf_file[:-4]}"
    if os.path.exists(f"{index_file}.faiss"):
        vector_stores[pdf_file] = FAISS.load_local(index_file, embeddings=embedding_model)
        print(f"Loaded existing FAISS index for {pdf_file}.")
    else:
        loader = PyMuPDFLoader(os.path.join(pdf_directory, pdf_file))
        documents = loader.load()
        print(f"Loaded {len(documents)} pages from {pdf_file}.")
        
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
        vector_stores[pdf_file] = FAISS.from_embeddings(textual_embeddings, embedding_model)
        print(f"Created FAISS index for {pdf_file}.")

        vector_stores[pdf_file].save_local(index_file)
        print(f"Saved FAISS index for {pdf_file} to {index_file}.")

llm_instance = AzureChatOpenAI(
    deployment_name=llm.GPT_4_OMNI_MODEL, 
    openai_api_version="2023-07-01-preview",
    openai_api_key=openai_api_client.api_key,
    openai_api_base=openai_api_client.api_base,
    openai_api_type=openai_api_client.api_type,
    temperature=0.0,
)

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

simplify_prompt = PromptTemplate(
    input_variables=["answer"],
    template="""
    Given the following answer, extract and return just the key point, which should be a concise summary such as a number, percentage, name, or brief phrase that directly conveys the main information. Do not include any additional explanation or context.

    Answer:
    {answer}

    Key point:"""
)

simplify_chain = LLMChain(
    llm=llm_instance,
    prompt=simplify_prompt
)

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

# Create DataFrame
df = pd.DataFrame(index=list_of_queries, columns=[pdf[:-4] for pdf in pdf_files])

# Process each PDF and query
for pdf_file, vector_store in vector_stores.items():
    retriever = vector_store.as_retriever(search_kwargs={"k": 20})
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm_instance,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": chat_prompt},
        verbose=False,
    )
    
    for query in tqdm(list_of_queries, desc=f"Processing queries for {pdf_file}"):
        try:
            result = qa_chain.run(query)
            simplified_result = simplify_chain.run(answer=result)
            df.at[query, pdf_file[:-4]] = simplified_result.strip()
        except Exception as e:
            print(f"Error processing query '{query}' for file {pdf_file}: {str(e)}")
            df.at[query, pdf_file[:-4]] = "Error"

# Save the DataFrame to a CSV file
output_file = os.path.join(pdf_directory, "results.csv")
df.to_csv(output_file)
print(f"Results saved to {output_file}")

# Display the DataFrame
print(df)