import streamlit as st
from langchain.schema import HumanMessage, AIMessage
from qa_functions import process_pdf_with_metadata, get_qa_chain  # Assuming these functions are in qa_functions.py

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
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None

# Load vectorstore and initialize qa_chain
if st.session_state.vectorstore is None or st.session_state.qa_chain is None:
    with st.spinner("Loading PDF and initializing Q&A system..."):
        st.session_state.vectorstore = process_pdf_with_metadata(pdf_options[selected_pdf])
        st.session_state.qa_chain = get_qa_chain(st.session_state.vectorstore)

def get_response(user_query, chat_history):
    return st.session_state.qa_chain.stream({
        "chat_history": chat_history,
        "user_question": user_query,
    })

# Display chat messages
for message in st.session_state.chat_history:
    with st.chat_message(message.type):
        st.markdown(message.content)

# User input
user_query = st.chat_input("Type your message here...")
if user_query is not None and user_query != "":
    st.session_state.chat_history.append(HumanMessage(content=user_query))

    with st.chat_message("Human"):
        st.markdown(user_query)

    with st.chat_message("AI"):
        response_placeholder = st.empty()
        full_response = ""
        for response_chunk in get_response(user_query, st.session_state.chat_history):
            full_response += response_chunk
            response_placeholder.markdown(full_response + "▌")
        response_placeholder.markdown(full_response)

    st.session_state.chat_history.append(AIMessage(content=full_response))

# Display source documents (optional)
if st.checkbox("Show source documents"):
    st.write("Source Documents:")
    retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 3})
    docs = retriever.get_relevant_documents(user_query)
    for i, doc in enumerate(docs):
        st.write(f"Document {i+1}:")
        st.write(doc.page_content[:300] + "...")  # Display first 300 characters
        st.write(f"Page: {doc.metadata.get('page_number', 'Unknown')}")
        st.write("---")