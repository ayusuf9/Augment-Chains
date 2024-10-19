import os
import pandas as pd
import time
import openai
from tenacity import retry, stop_after_attempt, wait_random_exponential, retry_if_exception_type
from langchain.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from tqdm import tqdm

# Set up API key
os.environ["OPENAI_API_KEY"] = "your-openai-api-key"

# Define retry logic for rate limit handling
@retry(
    reraise=True,
    stop=stop_after_attempt(6),  # Retry up to 6 times
    wait=wait_random_exponential(min=1, max=60),  # Wait between retries
    retry=retry_if_exception_type(openai.error.RateLimitError),
)
def embed_texts(texts, embedding_model):
    return embedding_model.embed_documents(texts)

# Initialize the embedding model
embedding_model = OpenAIEmbeddings()

# Check if the vector store exists
if os.path.exists("faiss_index"):
    # Load the existing vector store
    vector_store = FAISS.load_local("faiss_index", embedding_model)
else:
    # Load documents
    loader = DirectoryLoader(
        '/path/to/pdf_directory/',
        glob='*.pdf',
        loader_cls=PyPDFLoader
    )
    documents = loader.load()

    # Split documents
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""]
    )
    docs = text_splitter.split_documents(documents)

    # Batch embedding to handle rate limits
    batch_size = 20
    batches = [docs[i:i + batch_size] for i in range(0, len(docs), batch_size)]

    all_texts = []
    all_embeddings = []
    for batch in tqdm(batches, desc="Embedding documents"):
        texts = [doc.page_content for doc in batch]
        embeddings = embed_texts(texts, embedding_model)
        all_texts.extend(batch)
        all_embeddings.extend(embeddings)

    # Create the vector store with the embeddings
    vector_store = FAISS.from_embeddings(all_embeddings, all_texts, embedding_model)

    # Save the vector store
    vector_store.save_local("faiss_index")

# Set up QA system
llm = OpenAI()
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vector_store.as_retriever()
)

# Initialize dataframe
df = pd.DataFrame(columns=["Query", "Result"])

# Interactive loop
while True:
    query = input("Enter your query (or 'exit' to quit): ")
    if query.lower() == 'exit':
        break
    result = qa.run(query)
    df = df.append({"Query": query, "Result": result}, ignore_index=True)
    print("Result:", result)

# Save the dataframe
df.to_csv('query_results.csv', index=False)
