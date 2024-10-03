import streamlit as st
from pathlib import Path
import time
from langchain.callbacks import get_openai_callback

# Import necessary functions and classes from your original code
from your_module import (
    process_pdf_with_metadata,
    get_qa_chain,
    ask_question,
    OUTPUT_PRICE_PER_1K_TOKENS
)

# Initialize session state variables
if 'qa_chain' not in st.session_state:
    st.session_state['qa_chain'] = None
if 'pdf_path' not in st.session_state:
    st.session_state['pdf_path'] = None
if 'messages' not in st.session_state:
    st.session_state['messages'] = []

def process_pdf(pdf_path):
    with st.spinner('Processing PDF and setting up QA chain...'):
        vectorstore = process_pdf_with_metadata(pdf_path)
        qa_chain = get_qa_chain(vectorstore)
        return qa_chain

def main():
    st.title("PDF Question Answering Chatbot")

    pdf_option = st.selectbox(
        'Select a PDF to query:',
        ('First PDF', 'Second PDF')
    )

    if pdf_option == 'First PDF':
        pdf_path = '/Users/sptadm/Documents/pwk_app/pdfs/first.pdf'
    else:
        pdf_path = '/Users/sptadm/Documents/pwk_app/pdfs/second.pdf'

    # Process PDF and setup QA chain if not already done
    if st.session_state['qa_chain'] is None or st.session_state['pdf_path'] != pdf_path:
        st.session_state['qa_chain'] = process_pdf(pdf_path)
        st.session_state['pdf_path'] = pdf_path
        st.session_state['messages'] = []

    # Display past messages
    for message in st.session_state['messages']:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # User input
    if prompt := st.chat_input("Ask a question about the PDF"):
        # Display user message
        with st.chat_message("user"):
            st.write(prompt)
        st.session_state['messages'].append({"role": "user", "content": prompt})

        # Get response from QA chain
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

    # Add a button to clear the conversation
    if st.button("Clear Conversation"):
        st.session_state['messages'] = []
        st.experimental_rerun()

if __name__ == "__main__":
    main()