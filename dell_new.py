import warnings
import logging
from typing import Tuple, Optional
import numpy as np
import time
import tiktoken
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.chat_models import AzureChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.schema import HumanMessage, Document
from langchain.embeddings import OpenAIEmbeddings
from quanthub.util import llm

warnings.filterwarnings('ignore')

openai = llm.get_llm_client()
INPUT_PRICE_PER_1K_TOKENS = 0.005
OUTPUT_PRICE_PER_1K_TOKENS = 0.015

def get_embedding(text: str) -> list:
    try:
        embedding = openai.Embedding.create(
            input=text,
            deployment_id="text-embedding-ada-002",
            model="text-embedding-ada-002",
            api_key=openai.api_key
        )
        return embedding.data[0].embedding
    except Exception as e:
        raise

def process_chunk(chunk: Document) -> Optional[Tuple[str, list, dict]]:
    try:
        embedding = get_embedding(chunk.page_content)
        return (chunk.page_content, embedding, chunk.metadata)
    except Exception:
        return None

def get_improved_text_splitter() -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=250,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

def process_text_with_metadata(text: str, metadata: dict = None) -> Optional[FAISS]:
    print("Processing text...")
    try:
        # Create a Document object from the input text
        doc = Document(page_content=text, metadata=metadata or {})
        
        text_splitter = get_improved_text_splitter()
        chunks = text_splitter.split_documents([doc])

        batch_size = 10
        embeddings = []
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i+batch_size]
            batch_results = [result for result in map(process_chunk, batch) if result is not None]
            embeddings.extend(batch_results)
            print(f"Processed {i+len(batch)}/{len(chunks)} chunks")
            time.sleep(5)

        if not embeddings:
            raise ValueError("No valid embeddings were generated.")

        vectorstore = FAISS.from_embeddings(
            text_embeddings=[(e[0], e[1]) for e in embeddings],
            embedding=get_embedding,
            metadatas=[e[2] for e in embeddings]
        )

        return vectorstore

    except Exception as e:
        print(f"An error occurred while processing the text: {str(e)}")
        return None

def get_enhanced_retriever(vectorstore: FAISS) -> ContextualCompressionRetriever:
    if vectorstore is None:
        raise ValueError("Vectorstore is None. Text processing may have failed.")
    
    base_retriever = vectorstore.as_retriever(search_kwargs={"k": 19})
    deploy_name = llm.GPT_4_OMNI_MODEL
    
    llm_instance = AzureChatOpenAI(
        deployment_name=deploy_name,
        openai_api_version="2023-07-01-preview",
        openai_api_key=openai.api_key,
        openai_api_base=openai.api_base,
        openai_api_type=openai.api_type,
        temperature=0.0,
        max_tokens=800
    )
    compressor = LLMChainExtractor.from_llm(llm_instance)
    return ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=base_retriever
    )

def get_qa_chain(vectorstore: FAISS) -> RetrievalQA:
    retriever = get_enhanced_retriever(vectorstore)
    
    deploy_name = llm.GPT_4_OMNI_MODEL
    
    llm_gpt = AzureChatOpenAI(
        deployment_name=deploy_name,
        openai_api_version="2023-07-01-preview",
        openai_api_key=openai.api_key,
        openai_api_base=openai.api_base,
        openai_api_type=openai.api_type,
        temperature=0.0,
        max_tokens=800
    )

    cot_prompt = PromptTemplate(
        template="""You are a senior member of the Federal Open Market Committee (FOMC) with extensive experience in monetary policy and a 
                    reputation for nuanced communication. Your statements are closely watched by financial markets, and you 
                    frequently use FedSpeak terminologies to convey the committee's views and potential policy directions.

        Analyze the following context to answer the question using appropriate FedSpeak terminologies. In your analysis:

        1) Identify key points relevant to the question, focusing on monetary policy, economic indicators, and financial market conditions.
        2) Summarize the relevant information using FedSpeak terms where applicable.
        3) Consider any uncertainties or conflicting views, expressing them in a manner consistent with central bank communication.
        4) Formulate a comprehensive answer that reflects the nuanced language typical of Federal Reserve statements.

        Question: {question}
        Context: {context}

        Generate all sentences that best capture the answer, adhering to FedSpeak conventions and your role as an FOMC member:""",
        input_variables=["question", "context"]
    )

    return RetrievalQA.from_chain_type(
        llm=llm_gpt,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": cot_prompt},
        return_source_documents=True
    )

def get_summary(text: str, prompt: str, metadata: dict = None) -> Tuple[str, float]:
    vectorstore = process_text_with_metadata(text, metadata)
    
    if vectorstore is None:
        return "Failed to process the text.", 0.0
    
    qa_chain = get_qa_chain(vectorstore)
    
    from langchain.callbacks import get_openai_callback
    with get_openai_callback() as cb:
        response = qa_chain({"query": prompt})
    
    output_cost = (cb.completion_tokens / 1000) * OUTPUT_PRICE_PER_1K_TOKENS
    return response["result"], output_cost

def compare_texts(comparison_prompt: str, text1: str, text2: str) -> Tuple[str, float]:
    # First get summaries of both texts
    summary_prompt = "Summarize the participants' views on current conditions and the economic outlook."
    summary1, cost1 = get_summary(text1, summary_prompt)
    summary2, cost2 = get_summary(text2, summary_prompt)
    
    deploy_name = llm.GPT_4_OMNI_MODEL
    llm_one = AzureChatOpenAI(
        deployment_name=deploy_name,
        openai_api_version="2023-07-01-preview",
        openai_api_key=openai.api_key,
        openai_api_base=openai.api_base,
        openai_api_type=openai.api_type,
        temperature=0.0,
        max_tokens=800
    )

    comparison_template = PromptTemplate(
        input_variables=["role_context", "comparison_prompt", "summary_first", "summary_second"],
        template="""
        {role_context}

        {comparison_prompt}

        Summary of first text:
        {summary_first}

        Summary of second text:
        {summary_second}
        """
    )

    role_context = """
    You are a seasoned economic analyst attending a Federal Open Market Committee (FOMC) meeting. 
    Analyze the following summaries for subtle shifts in sentiment and potential policy implications. 
    Pay attention to and identify Fedspeak terms within the text and interpret them using Fedspeak language.
    """

    full_prompt = comparison_template.format(
        role_context=role_context,
        comparison_prompt=comparison_prompt,
        summary_first=summary1,
        summary_second=summary2
    )

    from langchain.callbacks import get_openai_callback
    with get_openai_callback() as cb:
        comparison = llm_one([HumanMessage(content=full_prompt)])

    output_cost = (cb.completion_tokens / 1000) * OUTPUT_PRICE_PER_1K_TOKENS + cost1 + cost2
    return comparison.content, output_cost

# Example usage:
if __name__ == "__main__":
    # Example texts (you would replace these with your actual texts)
    text1 = """Your first document text here..."""
    text2 = """Your second document text here..."""
    
    # Get summary of first text
    prompt = "Summarize the participants' views on current conditions and the economic outlook."
    summary1, cost1 = get_summary(text1, prompt, metadata={"source": "document1"})
    print(f"Summary of First Text:\n{summary1}")
    print(f"Cost for summary 1: ${cost1:.4f}")
    
    # Compare the texts
    comparison_prompt = """
    Compare and contrast the views expressed in these two texts, highlighting any significant changes or differences in economic outlook. 
    Please respond in several sentences.
    """
    comparison, total_cost = compare_texts(comparison_prompt, text1, text2)
    print(f"\nComparison:\n{comparison}")
    print(f"Total cost: ${total_cost:.4f}")