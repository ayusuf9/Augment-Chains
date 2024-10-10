import streamlit as st
import os
import pandas as pd
from functions import process_pdf_with_metadata_and_get_qa_chain, ask_question, get_llm_instance
import openai  # Make sure to import openai if not already imported
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# Define the prompts
prompts = [
    "What is the net operating income?",
    "What is the total debt?",
    "What is the issue date?"
]

# PDF Directory
PDF_DIR = '/Users/sptadm/Documents/pwk_app/pdfs/'

# Get list of PDFs
pdf_files = {
    f.replace('.pdf', ''): os.path.join(PDF_DIR, f)
    for f in os.listdir(PDF_DIR) if f.endswith('.pdf')
}

# Set default PDFs (first 3 PDFs in the list)
default_pdfs = list(pdf_files.keys())[:3]

def display_comp_table():
    # Add an option to recreate the CSV
    recreate_csv = st.checkbox('Recompute and update cached answers', value=False)

    # CSV file path
    csv_file_path = 'precomputed_answers.csv'

    # Load precomputed answers from CSV if not recreating
    if not recreate_csv and os.path.exists(csv_file_path):
        precomputed_df = pd.read_csv(csv_file_path)
    else:
        # Precompute answers and save to CSV
        precomputed_df = pd.DataFrame({'Prompt': prompts})
        with st.spinner('Precomputing answers for all PDFs...'):
            for pdf_name, pdf_path in pdf_files.items():
                # Process PDF and set up QA chain
                qa_chain = process_pdf_with_metadata_and_get_qa_chain(pdf_path)
                if qa_chain is None:
                    st.error(f"Failed to process PDF: {pdf_name}")
                    continue

                answers = []
                for prompt in prompts:
                    result, cost, _ = ask_question(prompt, qa_chain)
                    answers.append(result.strip() if result else "N/A")

                precomputed_df[pdf_name] = answers

        # Save the precomputed answers to CSV
        precomputed_df.to_csv(csv_file_path, index=False)

    # Multi-Select Widget for PDFs
    selected_pdfs = st.multiselect(
        '**Select Muni Documents to compare:**',
        options=list(pdf_files.keys()),
        default=default_pdfs
    )

    if selected_pdfs:
        # Filter the columns based on selected PDFs
        results_df = precomputed_df[['Prompt'] + selected_pdfs]

        # Display the table using Streamlit's DataFrame display
        st.data_editor(
            results_df.set_index('Prompt'),
            use_container_width=True,
            hide_index=False,
            num_rows="dynamic"
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
