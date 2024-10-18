import os
import warnings
import logging
import hashlib
from pathlib import Path
import numpy as np
import time
import asyncio
import tiktoken
import json
import requests
import io

from tenacity import retry, stop_after_attempt, wait_random_exponential

from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.chat_models import AzureChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.schema import HumanMessage, Document

from langchain.embeddings import OpenAIEmbeddings
from quanthub.util import llm

warnings.filterwarnings('ignore')

openai = llm.get_llm_client() 
INPUT_PRICE_PER_1K_TOKENS = 0.005
OUTPUT_PRICE_PER_1K_TOKENS = 0.015

# ... [previous functions remain unchanged] ...

def retrieve_documents(vectorstore, question):
    retriever = get_enhanced_retriever(vectorstore)
    retrieved_docs = retriever.get_relevant_documents(question)
    return retrieved_docs

def print_retrieved_documents(docs):
    for i, doc in enumerate(docs):
        print(f"\nDocument {i + 1}:")
        print(doc.page_content)
        print("Metadata:", doc.metadata)
        print("---")

def modify_documents(docs):
    modified_docs = []
    for i, doc in enumerate(docs):
        print(f"\nDocument {i + 1}:")
        print(doc.page_content)
        modify = input(f"Do you want to modify Document {i + 1}? (yes/no): ").lower()
        if modify == 'yes':
            new_content = input("Enter new content: ")
            modified_docs.append(Document(page_content=new_content, metadata=doc.metadata))
        else:
            modified_docs.append(doc)
    return modified_docs

def ask_question(question, chain):
    from langchain.callbacks import get_openai_callback

    with get_openai_callback() as cb:
        response = chain({"query": question})

    output_cost = (cb.completion_tokens / 1000) * OUTPUT_PRICE_PER_1K_TOKENS

    return response["result"], output_cost

def get_summary(prompt, url):
    pdf_pages = PyPDFLoader(url).load()
    vectorstore = process_pdf_with_metadata(pdf_pages)

    if vectorstore is None:
        return "Failed to process the PDF.", None

    retrieved_docs = retrieve_documents(vectorstore, prompt)
    print_retrieved_documents(retrieved_docs)

    decision = input("Do you want to modify the documents or proceed with processing? (modify/proceed): ").lower()

    if decision == 'modify':
        modified_docs = modify_documents(retrieved_docs)
        return modified_docs, "Documents modified"
    elif decision == 'proceed':
        return retrieved_docs, "Ready for LLM processing"
    else:
        return None, "Invalid decision"

def compare(comparison_prompt, summary_first_url, summary_second_url):
    from quanthub.util import llm
    deploy_name = llm.GPT_4_OMNI_MODEL

    llm_one = AzureChatOpenAI(
        deployment_name=deploy_name,
        openai_api_version="2023-07-01-preview",
        openai_api_key=openai.api_key,
        openai_api_base=openai.api_base,
        openai_api_type=openai.api_type,
        temperature=0.0,
        max_tokens=1000
    )

    comparison_template = PromptTemplate(
        input_variables=["role_context", "comparison_prompt", "summary_first", "summary_second"],
        template="""
        {role_context}

        {comparison_prompt}

        Summary of first document:
        {summary_first}

        Summary of second document:
        {summary_second}
        """
    )

    role_context = """
    You are a seasoned economist attending a Federal Open Market Committee (FOMC) meeting. 
    Analyze the following summaries of participant views on monetary policy for subtle shifts in sentiment and potential policy implications. 
    Pay close attention to the nuances of the language used, particularly any indications of evolving concerns about inflation, employment, or economic growth.
    Deliver your analysis by paying attention to and identifying Fedspeak terms within the text and ensure you interpret them using Fedspeak language.
    
    1) Identify key points relevant to the question, focusing on monetary policy, economic indicators, and financial market conditions.
    2) Summarize the relevant information using FedSpeak terms where applicable, such as "accommodative" to describe a 
       policy stance aimed at stimulating growth, "tapering" for the gradual reduction of asset purchases, "hawkish" to indicate a 
       preference for higher interest rates, or "dovish" to suggest a preference for lower interest rates 
       to support growth.
    3) Consider any uncertainties or conflicting views, expressing them in a manner consistent with central bank communication.
    4) Formulate a comprehensive answer that reflects the nuanced language typical of Federal Reserve statements.

    Focus on the quantity of FOMC members. When possible, acknowledge the quantity keywords. Quantity keywords are listed below. 

            # Quantity Keywords (these are listed in descending order):
            # 1. All but one
            # 2. Almost all; participants generally
            # 3. Most; majority
            # 4. Many
            # 5. Some
            # 6. Several
            # 7. Few
            # 8. A couple; two
            # 9. One; another
    """

    full_prompt = comparison_template.format(
        role_context=role_context,
        comparison_prompt=comparison_prompt,
        summary_first=summary_first_url,
        summary_second=summary_second_url
    )

    from langchain.callbacks import get_openai_callback

    with get_openai_callback() as cb:
        comparison = llm_one([HumanMessage(content=full_prompt)])

    output_cost = (cb.completion_tokens / 1000) * OUTPUT_PRICE_PER_1K_TOKENS

    return comparison.content, output_cost

# Main execution
url1 = 'https://www.federalreserve.gov/monetarypolicy/files/fomcminutes20240731.pdf'
url2 = 'https://www.federalreserve.gov/monetarypolicy/files/fomcminutes20240918.pdf'

prompt = "Summarize the participants' views on current conditions and the economic outlook."

summary1, cost1 = get_summary(prompt, url1)
if isinstance(summary1, list):  # If documents were modified
    print("Modified documents for first PDF:")
    for doc in summary1:
        print(doc.page_content)
        print("---")
else:
    print(f"Summary For First PDF:\n{summary1}")
print(f"Cost for summary 1: ${cost1:.4f}")

summary2, cost2 = get_summary(prompt, url2)
if isinstance(summary2, list):  # If documents were modified
    print("Modified documents for second PDF:")
    for doc in summary2:
        print(doc.page_content)
        print("---")
else:
    print(f"Summary For Second PDF:\n{summary2}")
print(f"Cost for summary 2: ${cost2:.4f}")

comparison_prompt = "Compare and contrast the views expressed in these two sets of minutes, focusing on any shifts in economic outlook or policy stance."

comparison_result, comparison_cost = compare(comparison_prompt, summary1, summary2)
print(f"\nComparison Result:\n{comparison_result}")
print(f"Cost for comparison: ${comparison_cost:.4f}")

total_cost = cost1 + cost2 + comparison_cost
print(f"\nTotal Cost: ${total_cost:.4f}")