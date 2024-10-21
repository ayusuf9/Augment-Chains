import os
import warnings
import time
from tqdm import tqdm
from tenacity import retry, stop_after_attempt, wait_random_exponential, retry_if_exception_type

from langchain.document_loaders import DirectoryLoader, PyPDFLoader, PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores.faiss import FAISS
from langchain.chat_models import AzureChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

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

faiss_index_file = "/users/CFII_DataScience/USERs/SPTADM/Agents_Experiment/pdf_files/faiss_index"

if os.path.exists(f"{faiss_index_file}.faiss"):
  vector_store = FAISS.load_local(faiss_index_file, embeddings=embedding_model)
  print("Loaded existing FAISS index.")
else:
  loader = DirectoryLoader(
      '/users/CFII_DataScience/USERs/SPTADM/Agents_Experiment/pdf_files',
      glob='*.pdf',
      loader_cls=PyMuPDFLoader
  )
  documents = loader.load()
  print(f"Loaded {len(documents)} documents.")
  
  
batch_size = 10
all_texts = []
all_embeddings = []

for i in tqdm(range(0, len(documents), batch_size), desc="Embedding documents"):
  batch_docs = documents[i:i + batch_size]
  texts = [doc.page_content for doc in batch_docs]
  embeddings = embed_texts(texts, embedding_model)
  all_texts.extend(batch_docs)
  all_embeddings.extend(embeddings)
  time.sleep(5)
  
  
pure_texts = [page.page_content for page in all_texts]
textual_embeddings = zip(pure_texts, all_embeddings)
vector_store = FAISS.from_embeddings(textual_embeddings, embedding_model)
print("Created FAISS index.")

vector_store.save_local(faiss_index_file)
print(f"Saved FAISS index to {faiss_index_file}.")


llm_instance = AzureChatOpenAI(
  deployment_name=llm.GPT_4_OMNI_MODEL, 
  openai_api_version="2023-07-01-preview",
  openai_api_key=openai_api_client.api_key,
  openai_api_base=openai_api_client.api_base,
  openai_api_type=openai_api_client.api_type,
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


qa_chain = RetrievalQA.from_chain_type(
  llm=llm_instance,
  chain_type="stuff",
  retriever=retriever,
  chain_type_kwargs={"prompt": chat_prompt},
  verbose=False,
)



query = "where is this property located?"
result = qa_chain.run(query)
print("Answer:")
print(result)
