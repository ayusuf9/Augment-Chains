import os
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import tabula
import pandas as pd

# Prerequisites:
# 1. Set your OpenAI API key (for embeddings) via environment variable: 
#    os.environ["OPENAI_API_KEY"] = "your_openai_api_key"
# 2. Install required packages:
#    pip install langchain langchain_community tabula-py PyPDF2 pandas

# Given PDF
pdf_path = "your_document.pdf"

# Step 1: Load the PDF and split into pages as Documents
loader = PyPDFLoader(pdf_path)
pages = loader.load()

# Each element in 'pages' is a Document with:
#   page_content = the text content of that page
#   metadata = {"source": pdf_path, "page": page_number}

# If needed, further split text into chunks, but for page-level retrieval, we may skip this step.
# Here, we assume that the entire page content is used as is. If the page is huge, consider splitting.
text_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=0)
page_docs = []
for i, page_doc in enumerate(pages, start=1):
    # page_doc.metadata["page"] already provided by PyPDFLoader
    # If not, ensure page numbers are included:
    if "page" not in page_doc.metadata:
        page_doc.metadata["page"] = i
    # Split if needed (optional)
    chunks = text_splitter.split_documents([page_doc])
    for c in chunks:
        c.metadata["page"] = page_doc.metadata["page"]
        page_docs.append(c)

# Step 2: Create embeddings and load them into a FAISS vector store in memory
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(page_docs, embeddings)

# Step 3: Identify the page containing the desired table via similarity search
query = "The table listing quarterly financial results for 2023 Q2"
docs = vectorstore.similarity_search(query, k=1)

if not docs:
    raise ValueError("No relevant page found for the given query.")

target_page = docs[0].metadata["page"]  # The page number where the table is found

# Step 4: Extract the table from the identified page using Tabula
dfs = tabula.read_pdf(pdf_path, pages=target_page, multiple_tables=True)

if not dfs:
    raise ValueError(f"No tables found on page {target_page}.")

df = dfs[0]  # The first extracted table as a DataFrame

# 'df' is now a pandas DataFrame containing your table.
print("Extracted Table:")
print(df)
