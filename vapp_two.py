import streamlit as st
import os
from pathlib import Path
import time
import logging
from langchain.callbacks import get_openai_callback
from streamlit_navigation_bar import st_navbar
import tables  # Add this import to use tables.py

st.set_page_config(
page_title="Muni GPT",
page_icon="🏢",
initial_sidebar_state="expanded",
)

#st.set_page_config(initial_sidebar_state="collapsed", layout="wide")
styles = {
"nav": {
    "background-color": "rgb(54, 69, 79)", #"rgb(115, 147, 179)",  
     "height": "4rem",  
},
"div": {
    "max-width": "24rem",
},
"span": {
    "border-radius": "0.5rem",
    "color": "rgb(255, 255, 255)", 
    "margin": "0 0.125rem",
    "padding": "0.4375rem 0.625rem",
},
"active": {
    "background-color": "rgba(173, 216, 230, 0.25)", 
},
"hover": {
    "background-color": "rgba(173, 216, 230, 0.35)", 
},
}

logo_path = "/Users/sptadm/Documents/pwk_app/new_ic.svg"

page = st_navbar(["Home", "Comp-Table", "Surveillance"], selected="Home", styles=styles, logo_path=logo_path)

# Update the navigation logic
if page == "Home":
    main()  # Call the main function for the Home page
elif page == "Comp-Table":
    tables.display_comp_table()  # Call the function from tables.py directly
elif page == "Surveillance":
    st.write("Surveillance page content")

# Set the theme
st.markdown("""

""", unsafe_allow_html=True)

#st.info("**Muni GPT**")

# Import necessary functions and classes from your original code
from functions import (
  process_pdf_with_metadata,
  get_qa_chain,
  ask_question,
  OUTPUT_PRICE_PER_1K_TOKENS
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_avatar(role):
    if role == "user":
        return ":material/account_box:"
    elif role == "assistant":
        return ":material/psychology_alt:"
    else:
        return None 

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
  #st.title("Muni GPT")

  default_pdf = next(iter(pdf_files.keys()))
  pdf_option = st.selectbox(
      '**Select a Muni Document to query below:**',
      options=list(pdf_files.keys()),
      index=2
  )

  pdf_path = pdf_files[pdf_option]

  # Process PDF and setup QA chain if not already done
  #if st.session_state['qa_chain'] is None or st.session_state['pdf_path'] != pdf_path:
  if 'qa_chain' not in st.session_state or st.session_state.get('pdf_path') != pdf_path:
      with st.spinner('Processing PDF and setting up QA chain...'):
          qa_chain = process_pdf(pdf_path)
          if qa_chain is not None:
              st.session_state['qa_chain'] = qa_chain
              st.session_state['pdf_path'] = pdf_path
              st.session_state['messages'] = []
          else:
              st.error("Failed to process PDF. Please try another file or check the logs.")
              return

  for message in st.session_state['messages']:
      with st.chat_message(message["role"], avatar=get_avatar(message["role"])):
          st.write(message["content"])

  if prompt := st.chat_input("Ask a question about the PDF"):
      with st.chat_message("user", avatar=":material/account_box:"):
          st.write(prompt)
      st.session_state['messages'].append({"role": "user", "content": prompt})

      with st.chat_message("assistant", avatar=":material/psychology_alt:"):
          with st.spinner("Thinking..."):
              try:
                  result, cost, source_docs = ask_question(prompt, st.session_state['qa_chain'])
                  if result:
                      st.markdown(result)
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