import os
import pandas as pd
from langchain.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA

# Set up API key
os.environ["OPENAI_API_KEY"] = "your-openai-api-key"

# Load documents
loader = DirectoryLoader(
    '/path/to/pdf_directory/',
    glob='*.pdf',
    loader_cls=PyPDFLoader
)
documents = loader.load()

# Split documents
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100
)
docs = text_splitter.split_documents(documents)

# Create embeddings and vector store
embedding_model = OpenAIEmbeddings()
vector_store = FAISS.from_documents(docs, embedding_model)

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
