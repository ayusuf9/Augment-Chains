import streamlit as st
import os
import logging
from app_test import (
    process_pdf_with_metadata,
    get_qa_chain,
    ask_question,
    OUTPUT_PRICE_PER_1K_TOKENS
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize session state variables
if 'qa_chain' not in st.session_state:
    st.session_state['qa_chain'] = None
if 'pdf_path' not in st.session_state:
    st.session_state['pdf_path'] = None
if 'messages' not in st.session_state:
    st.session_state['messages'] = []

def process_pdf(pdf_path):
    try:
        with st.spinner('Processing PDF and setting up QA chain...'):
            vectorstore = process_pdf_with_metadata(pdf_path)
            if vectorstore is None:
                st.error(f"Failed to process PDF: {pdf_path}")
                logger.error(f"process_pdf_with_metadata returned None for {pdf_path}")
                return None
            
            qa_chain = get_qa_chain(vectorstore)
            if qa_chain is None:
                st.error(f"Failed to create QA chain for PDF: {pdf_path}")
                logger.error(f"get_qa_chain returned None for {pdf_path}")
                return None
            
            return qa_chain
    except Exception as e:
        st.error(f"An error occurred while processing the PDF: {str(e)}")
        logger.exception(f"Error in process_pdf for {pdf_path}")
        return None

PDF_DIR = '/Users/sptadm/Documents/pwk_app/pdfs/'
pdf_files = {
    f.replace('.pdf', ''): os.path.join(PDF_DIR, f)
    for f in os.listdir(PDF_DIR) if f.endswith('.pdf')
}

def main():
    st.title("PDF Question Answering Chatbot")

    pdf_option = st.selectbox(
        'Select a Muni Document to query:',
        list(pdf_files.keys())
    )

    pdf_path = pdf_files[pdf_option]

    # Process PDF and setup QA chain if not already done
    if st.session_state['qa_chain'] is None or st.session_state['pdf_path'] != pdf_path:
        qa_chain = process_pdf(pdf_path)
        if qa_chain is not None:
            st.session_state['qa_chain'] = qa_chain
            st.session_state['pdf_path'] = pdf_path
            st.session_state['messages'] = []
        else:
            st.error("Failed to process PDF. Please try another file or check the logs.")
            return

    for message in st.session_state['messages']:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    if prompt := st.chat_input("Ask a question about the PDF"):
        with st.chat_message("user"):
            st.write(prompt)
        st.session_state['messages'].append({"role": "user", "content": prompt})

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    result, cost, source_docs = ask_question(prompt, st.session_state['qa_chain'])
                    if result:
                        st.write(result)
                        st.session_state['messages'].append({"role": "assistant", "content": result})
                        st.write(f"Cost: ${cost:.4f}")
                    else:
                        st.error("Unable to get a response. Please try again.")
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
                    logger.exception("Error in ask_question")

    if st.button("Clear Conversation"):
        st.session_state['messages'] = []
        st.rerun()

if __name__ == "__main__":
    main()