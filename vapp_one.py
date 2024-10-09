import streamlit as st
import os
from pathlib import Path
import pandas as pd
from functions import process_pdf_with_metadata, get_qa_chain, ask_question, get_llm_instance
import openai  # Make sure to import openai if not already imported
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

def display_comp_table():
    # PDF Directory
    PDF_DIR = '/Users/sptadm/Documents/pwk_app/pdfs/'

    # Get list of PDFs
    pdf_files = {
        f.replace('.pdf', ''): os.path.join(PDF_DIR, f)
        for f in os.listdir(PDF_DIR) if f.endswith('.pdf')
    }

    # Multi-Select Widget for PDFs
    selected_pdfs = st.multiselect(
        '**Select Muni Documents to compare:**',
        options=list(pdf_files.keys()),
        default=[]
    )

    # Define the prompts
    prompts = [
        "What is the net operating income?",
        "What is the total debt?",
        "What is the issue date?"
    ]

    if selected_pdfs:
        # Initialize a DataFrame to store the results
        results_df = pd.DataFrame({'Prompt': prompts})

        # Process each selected PDF
        for pdf_name in selected_pdfs:
            pdf_path = pdf_files[pdf_name]

            # Process PDF and set up QA chain
            qa_chain = process_pdf_with_metadata_and_get_qa_chain(pdf_path)
            if qa_chain is None:
                st.error(f"Failed to process PDF: {pdf_name}")
                continue

            # Get answers for each prompt
            answers = []
            for prompt in prompts:
                result, cost, _ = ask_question(prompt, qa_chain)
                answers.append(result.strip() if result else "N/A")

            # Add answers as a new column in the DataFrame
            results_df[pdf_name] = answers

        # Display the table using Streamlit's DataFrame display
        st.data_editor(
            results_df.set_index('Prompt'),
            use_container_width=True,
            hide_index=False
        )
    else:
        st.info("Please select at least one PDF to compare.")

def process_pdf_with_metadata_and_get_qa_chain(pdf_path):
    # Check if we have already processed this PDF
    vectorstore = process_pdf_with_metadata(pdf_path)
    if vectorstore is not None:
        qa_chain = get_concise_qa_chain(vectorstore)  # Use the custom concise QA chain here
        return qa_chain
    else:
        return None

def get_concise_qa_chain(vectorstore):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    # Use get_llm_instance from functions.py
    llm_gpt = get_llm_instance(
        deployment_name="YOUR_DEPLOYMENT_NAME",  # Replace with your deployment name
        temperature=0.0,
        max_tokens=50
    )

    # Define a concise prompt to get short answers
    concise_prompt = PromptTemplate(
        template="""
            You are an AI assistant that provides brief and precise answers.

            Question: {question}

            Provide only the specific value or short answer, without any additional explanations.
            """,
        input_variables=["question"]
    )

    # Create a new QA chain with the concise prompt
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm_gpt,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": concise_prompt},
        return_source_documents=False
    )
    return qa_chain
