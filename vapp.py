import streamlit as st
import os
from pathlib import Path
import pandas as pd
from functions import process_pdf_with_metadata, get_qa_chain, ask_question

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
        qa_chain = get_qa_chain(vectorstore)
        return qa_chain
    else:
        return None
