import streamlit as st
from langchain.callbacks import StreamlitCallbackHandler
from qa_functions import *  # Import all functions from qa_functions.py

# Streamlit app
st.set_page_config(page_title="PDF Q&A Chatbot", page_icon="📚")
st.title("PDF Q&A Chatbot")

# PDF selection
pdf_options = {
    "PDF 1": '/users/CFII_DataScience/USERs/SPTADM/Improved_LLM_Chat/pwk_second.pdf',
    "PDF 2": '/users/CFII_DataScience/USERs/SPTADM/Improved_LLM_Chat/pwk_second.pdf'
}
selected_pdf = st.selectbox("Select a PDF to query:", list(pdf_options.keys()))

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None

# Load vectorstore and initialize qa_chain
if st.session_state.vectorstore is None or st.session_state.qa_chain is None:
    with st.spinner("Loading PDF and initializing Q&A system..."):
        st.session_state.vectorstore = process_pdf_with_metadata(pdf_options[selected_pdf])
        st.session_state.qa_chain = get_qa_chain(st.session_state.vectorstore)

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask a question about the PDF"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        stream_handler = StreamlitCallbackHandler(st.empty())
        response_container = st.empty()
        
        try:
            with st.spinner("Thinking..."):
                response = st.session_state.qa_chain({"query": prompt}, callbacks=[stream_handler])
            
            answer = response['result']
            
            # Fallback display in case streaming didn't work
            if not stream_handler.has_streamed:
                response_container.markdown(answer)
            
            st.session_state.messages.append({"role": "assistant", "content": answer})
            
            # Display source documents with page numbers
            if st.checkbox("Show source documents"):
                st.write("Source Documents:")
                for doc in response['source_documents']:
                    page_number = doc.metadata.get('page_number', 'Unknown')
                    st.write(f"Page {page_number}: {doc.page_content[:100]}...")  # Display first 100 characters of each source
        
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.write("Debug information:")
            st.write(f"Prompt: {prompt}")
            st.write(f"QA Chain type: {type(st.session_state.qa_chain)}")
            st.write(f"Vectorstore type: {type(st.session_state.vectorstore)}")

# Display cost (optional)
if st.checkbox("Show cost"):
    total_cost = sum(message.get("cost", 0) for message in st.session_state.messages)
    st.write(f"Total cost: ${total_cost:.4f}")