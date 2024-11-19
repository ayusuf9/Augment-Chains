import os
import openai
import json
import streamlit as st
import pandas as pd
from datetime import datetime
from streamlit_navigation_bar import st_navbar
from streamlit_extras.switch_page_button import switch_page
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores.faiss import FAISS
from langchain.chat_models import AzureChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain import LLMChain
from langchain.retrievers.multi_query import MultiQueryRetriever
from quanthub.util import llm2


CACHE_FILE = "cached_data.json"

st.set_page_config(layout="wide", 
                   page_icon="📑", 
                   page_title="Property Analysis",
                   initial_sidebar_state="collapsed")

styles = {
    "nav": {
        "background-color": "rgb(54, 69, 79)",
        "height": "5rem",
    },
    "div": {
        "max-width": "24rem",
    },
    "span": {
        "border-radius": "0.5rem",
        "color": "rgb(255, 255, 255)",
        "margin": "0.125rem",
        "padding": "0.4375rem 0.625rem",
        "font-family": "'Arial', sans-serif",
        "font-size": "20px",
        "font-weight": "bold"
    },
    "active": {
        "background-color": "rgba(173, 216, 230, 0.25)",
        "font-weight": "bold",
        "padding": "14px",
    },
    "hover": {
        "background-color": "rgba(173, 216, 230, 0.35)",
    },
}   

def initialize_session_state():
    session_vars = {
        'custom_questions': [],
        'qa_results': pd.DataFrame(),
        'processed_documents': set(),
        'vector_stores': {},
        'embeddings': None,
        'llm': None
    }
    
    for var, default in session_vars.items():
        if var not in st.session_state:
            st.session_state[var] = default

    if 'results_df' not in st.session_state:
        st.session_state.results_df = load_cached_results()
    if 'processing' not in st.session_state:
        st.session_state.processing = False
    if 'llm' not in st.session_state:
        st.session_state.llm = load_llm_model()
    if 'embeddings' not in st.session_state:
        st.session_state.embeddings = load_embeddings_model()

def load_cached_results():
    try:
        if os.path.exists(CACHE_FILE):
            with open(CACHE_FILE, 'r') as f:
                cache_data = json.load(f)
                return pd.DataFrame.from_dict(cache_data.get('results', {}))
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error loading cache: {str(e)}")
        return pd.DataFrame()

def load_embeddings_model():
    openai_api_client = llm2.get_llm_client()
    try:
        return OpenAIEmbeddings(
            deployment="text-embedding-3-large",
            model="text-embedding-3-large",
            openai_api_key=openai_api_client.api_key,
            openai_api_base=openai_api_client.api_base,
            openai_api_type=openai_api_client.api_type,
            chunk_size=100
        )
    except Exception as e:
        st.error(f"Failed to load embeddings: {str(e)}")
        return None

def load_llm_model():
    openai_api_client =llm2.get_llm_client()
    try:
        return AzureChatOpenAI(
            deployment_name=llm2.GPT_4_O1_MODEL,  
            model_name='o1-preview', 
            openai_api_version="2023-07-01-preview",
            openai_api_key=openai_api_client.api_key,
            openai_api_base=openai_api_client.api_base,
            openai_api_type="azure_ad",
            temperature=1,
        )
    except Exception as e:
        st.error(f"Failed to load LLM: {str(e)}")
        return None

def get_multi_query_retriever(vector_store):
    """Setup retriever"""
    question_prompt = PromptTemplate(
        input_variables=["question"],
        template="""You are an expert in municipal finance. Break down the question:
        Question: {question}
        Generate similar alternative questions."""
    )
    
    try:
        return MultiQueryRetriever.from_llm(
            llm=st.session_state.llm,
            retriever=vector_store.as_retriever(search_kwargs={"k": 10}),
            parser_key="questions",
            prompt=question_prompt
        )
    except Exception as e:
        st.error(f"Retriever error: {str(e)}")
        return None

def process_queries(vector_stores, queries, progress_bar=None):
    """Process queries against vector stores"""
    if not st.session_state.llm:
        st.error("LLM not initialized properly")
        return pd.DataFrame()

    chat_prompt = PromptTemplate(
        template="""
                **Question:**

                {question}

                **Instructions for the AI Assistant:**

                You are an expert investment analyst specializing in analyzing municipal bond issuance documents, particularly those from MuniOs.com. Your role is to provide accurate, insightful, and professional answers to questions about municipal bonds, just like a seasoned financial professional in a leading financial company.

                When generating your response, please follow these steps:

                1. **Understand the Question and Context**: Carefully read the question and the provided context to fully comprehend what is being asked.

                2. **Extract Relevant Information**: Identify the key pieces of information in the context that are pertinent to answering the question.

                3. **Perform Detailed Analysis**: Analyze the extracted information, performing any necessary calculations or evaluations. Ensure all calculations are accurate and clearly explained.

                4. **Synthesize Your Findings**: Combine your analysis into a coherent response that addresses all aspects of the question.

                5. **Provide a Clear and Professional Answer**: Present your conclusions in a concise and precise manner, using proper financial terminology and maintaining a professional tone.

                **Guidelines:**

                - **Use Only the Provided Context**: Base your answer solely on the information given in the context. Do not include information that is not present in the context. Avoid introducing any outside knowledge or assumptions.

                - **Accuracy and Precision**: Ensure all information you provide is correct, especially numerical data and financial terms.

                - **Calculations**: Show all relevant calculation steps and provide the final results.

                - **Clarity and Professionalism**: Write in clear, concise language appropriate for a professional audience.

                **Note**:

                - The total debt is the sum of the principal amounts without interest.

                **Context:**

                {context}

                **Answer:**

                [Provide your detailed analysis and final answer here.]
                """
                ,
        input_variables=["question", "context"]
    )
    
    simplify_prompt = PromptTemplate(
        input_variables=["answer"],
        template="""
          Given the following answer, extract and return only the key point. The key point should be a concise summary that directly conveys the main information, such as:
  
          - For property addresses, extract only the **city and state** (e.g., 'Santa Rosa, California').
          - If a **year** is mentioned, extract the year (e.g., '2003').
          - For **quantities or counts**, extract the numerical value (e.g., '277').
          - For **percentages or rates**, extract the numerical value along with the percent symbol (e.g., '3.50%').
          - A specific number along with its unit or currency symbol (e.g., '180 units', '$65,338.68').
          - A **percentage** (e.g., '94.63%').
          - A **name or proper noun** (e.g., 'Waterscape Apartments', 'Riverside County, California').
          - A **brief descriptive phrase** that includes necessary qualifiers (e.g., 'trending positively', 'decreased by $59,800.84', 'increase of $1,243.72').
  
          Do not include any additional explanation, context, or restatement of the answer. **Provide only the key point as it directly relates to the main information.**
  
          **Examples:**
  
          1. **Answer:** The subject property is located at 4656 Quigg Drive, Santa Rosa, California 95409.
  
          **Key point:** Santa Rosa, California
  
          2. **Answer:** The property located at 4656 Quigg Drive, Santa Rosa, California 95409 was built in 2003.
  
          **Key point:** 2003
  
          3. **Answer:** The total number of units at the property located at 4656 Quigg Drive, Santa Rosa, California 95409 is 277 residential apartment units.
  
          **Key point:** 277
  
          4. **Answer:** The Cap Rate for the property located at 4656 Quigg Drive, Santa Rosa, California 95409 is 3.50%. This is based on the concluded going-in capitalization rate derived from the income capitalization approach and market participant discussions.
  
          **Key point:** 3.50%
  
          5. **Answer:** The property is located at Waterscape Apartments, identified by the code "cawater2."
  
          **Key point:** Waterscape Apartments
  
          6. **Answer:** The property, Waterscape Apartments, is located in Riverside County, California.
  
          **Key point:** Riverside County, California
  
          7. **Answer:** The change in total income over the last three months is $65,338.68.
  
          **Key point:** $65,338.68
  
          8. **Answer:** The most recent occupancy rate is 94.63%.
  
          **Key point:** 94.63%
  
          9. **Answer:** The total number of units is 180.
  
          **Key point:** 180 units
  
          10. **Answer:** The occupancy rate is trending positively.
  
              **Key point:** trending positively
  
          11. **Answer:** The total concessions as a percentage of total income are approximately -1.05%.
  
              **Key point:** -1.05%
  
          12. **Answer:** The total property expenses decreased by $59,800.84 over the last three months.
  
              **Key point:** decreased by $59,800.84
  
          13. **Answer:** The change in total property expenses over the last 12 months is an increase of $1,243.72.
  
              **Key point:** increase of $1,243.72
  
          Now, given the following answer, extract and return just the **key point**.
  
          **Answer:**
          {answer}
  
          **[final answer here]**
                  """
  
    )
    
    simplify_chain = LLMChain(llm=st.session_state.llm, prompt=simplify_prompt)
    
    results = pd.DataFrame(columns=list(vector_stores.keys()), index=queries)
    total_operations = len(queries) * len(vector_stores)
    current_operation = 0
    
    for identifier, vector_store in vector_stores.items():
        multi_retriever = get_multi_query_retriever(vector_store)
        if not multi_retriever:
            continue
        
        qa_chain = RetrievalQA.from_chain_type(
            llm=st.session_state.llm,
            chain_type="stuff",
            retriever=multi_retriever,
            chain_type_kwargs={"prompt": chat_prompt}
        )
        
        for query in queries:
            try:
                result = qa_chain.run(query)
                simplified_result = simplify_chain.run(answer=result)
                results.at[query, identifier] = simplified_result.strip()
            except Exception as e:
                results.at[query, identifier] = f"Error: {str(e)}"
            
            current_operation += 1
            if progress_bar is not None:
                progress_bar.progress(current_operation / total_operations)
    
    return results


def load_vector_stores(identifier, base_path):
    try:
        if identifier in st.session_state.vector_stores:
            return st.session_state.vector_stores[identifier]
        
        index_folder = os.path.join(base_path, f"{identifier}_faiss_index")
        
        if not os.path.exists(index_folder):
            st.error(f"Index folder not found for {identifier}")
            return None
        
        vector_store = FAISS.load_local(
            index_folder,
            embeddings=st.session_state.embeddings,
            allow_dangerous_deserialization=True
        )
        st.session_state.vector_stores[identifier] = vector_store
        return vector_store
    except Exception as e:
        st.error(f"Error loading index for {identifier}: {str(e)}")
        return None


def style_dataframe(df):
    """Style the results table"""
    return df.style.set_properties(**{
        'text-align': 'left',
        'font-size': '16px',
        'padding': '8px',
        'border': '1px solid lightgrey'
    }).set_table_styles([
        {'selector': 'th', 'props': [
            ('background-color', '#f0f2f6'),
            #('font-weight', 'bold'),
            ('font-size', '15px')
        ]},
        {'selector': 'tr:nth-of-type(even)', 'props': [
            ('background-color', '#f9f9f9')
        ]}
    ])


def main():
  initialize_session_state()

  base_path = '/Users/sptadm/Desktop/point_local_2'

  options = {
      'show_menu': False
  }
  
  page = st_navbar(["Home", "Chat", "Tables"], selected="Tables", styles=styles, options=options)
  if page == "Home": switch_page("streamlit app")
  if page == "Chat": switch_page("chat")
  
  st.title("Comp Tables")
  
  # Ensure results_df exists in session state
  if 'results_df' not in st.session_state:
      st.error("No data loaded. Please initialize results_df first.")
      return

  available_docs = st.session_state.results_df.columns.tolist()
  st.markdown(
      """
  <style>
  span[data-baseweb="tag"] {
    background-color: rgb(54, 69, 79) !important;
  }
  </style>
  """,
      unsafe_allow_html=True,
  )
  
  selected_docs = st.multiselect("Select documents:", available_docs, default=available_docs[:5])
  
  if selected_docs:
      # Display current results
      display_df = st.session_state.results_df[selected_docs]
      styled_df = style_dataframe(display_df)
      st.write(styled_df.to_html(escape=False), unsafe_allow_html=True)
      
      st.markdown("### Analysis Results")

      default_questions = [
          "where is this property located?",
          "what year was the property built",
          "what is the total number of units?",
          "What is the Cap Rate?",
          "What is the Total Senior Debt?",
          "what is the Total Mezzanine Debt?",
          "what is the Total Debt?",
          "How much is in the coverage reserve fund",
          "How much is in the senior capitalized interest fund",
          "How much is in the mezzanine capitalized interest",
          "what is the Rental Revenues in 2024",
          "Net Operating Income in 2024?"
      ]

      # Question input section with validation
      new_question = st.text_input("Add a new question:", key="new_question_input")
      add_question = st.button("Add Question")

      if add_question and new_question:
          # Validate if question is not empty and not already in the list
          if new_question.strip():
              if 'custom_questions' not in st.session_state:
                  st.session_state.custom_questions = []
              
              if new_question not in st.session_state.custom_questions:
                  st.session_state.custom_questions.append(new_question)
                  st.success(f"Added new question: {new_question}")
              else:
                  st.warning("This question already exists!")
          else:
              st.warning("Please enter a valid question!")

      # Combine default and custom questions
      all_questions = default_questions + st.session_state.custom_questions

      # Load vector stores with progress indicator
      with st.spinner("Loading vector stores..."):
          vector_stores = {}
          for identifier in selected_docs:
              try:
                  vector_store = load_vector_stores(identifier, base_path)
                  if vector_store:
                      vector_stores[identifier] = vector_store
              except Exception as e:
                  st.error(f"Error loading vector store for {identifier}: {str(e)}")

      if vector_stores:

          try:
              with st.spinner("Processing questions..."):

                  if 'qa_results' not in st.session_state:
                      st.session_state.qa_results = pd.DataFrame()

                  results_df = process_queries(
                      vector_stores,
                      all_questions,
                      st.session_state.qa_results
                  )

                  st.session_state.qa_results = results_df

                  styled_df = style_dataframe(results_df)
                  st.write(styled_df.to_html(escape=False), unsafe_allow_html=True)

                  if st.button("Export Results"):
                      csv = results_df.to_csv().encode('utf-8')
                      st.download_button(
                          label="Download CSV",
                          data=csv,
                          file_name=f"analysis_results_{datetime.now().strftime('%Y%m%d')}.csv",
                          mime="text/csv"
                      )

          except Exception as e:
              st.error(f"Error processing questions: {str(e)}")
              st.exception(e)

      else:
          st.warning("No vector stores could be loaded. Please check your data sources.")
  else:
      st.info("Please select documents to view analysis")

  if st.checkbox("Show Debug Info"):
      st.write("Current custom questions:", st.session_state.get('custom_questions', []))
      st.write("Number of loaded vector stores:", len(vector_stores))

if __name__ == "__main__":
  main()