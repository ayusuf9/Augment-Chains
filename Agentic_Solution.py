import streamlit as st
import pandas as pd
from streamlit_navigation_bar import st_navbar
from langchain.agents import create_pandas_dataframe_agent
from langchain.llms import OpenAI
from langchain.callbacks import StreamlitCallbackHandler
from langchain.prompts import PromptTemplate

st.set_page_config(
    page_title="Surveillance",
    page_icon="",
    initial_sidebar_state="collapsed",
    layout="wide"
)

# ... [Previous code for styles and navigation remains unchanged]

@st.cache_data
def load_data():
    return pd.read_csv('/Users/sptadm/Documents/pwk_app/surveillance.csv')

df = load_data()

properties = list(df.columns)

selected_properties = st.multiselect(
    '**Select Properties to compare**',
    options=properties,
    default=properties[0:5]
)

if selected_properties:
    filtered_df = df[selected_properties]

    # ... [Code for styling and displaying the table remains unchanged]

    # Create the specialized Langchain agent
    openai_api_key = st.secrets["OPENAI_API_KEY"]
    llm = OpenAI(temperature=0, openai_api_key=openai_api_key)

    # Define a custom prompt template for the agent
    custom_prompt = PromptTemplate(
        input_variables=["df_info", "user_input"],
        template="""
        You are an expert municipal bond analyst. You have access to a dataframe with the following information:
        {df_info}

        This dataframe contains important credit metrics for various municipal bond projects. Your task is to analyze this data and provide insights on:
        1. Projects that should be monitored based on negatively trending credit metrics.
        2. Trends by county that should be followed.
        3. Projects performing the best relative to financial projections.

        Pay special attention to metrics such as:
        - Changes in total income (3 months and 12 months)
        - Occupancy rates and their trends
        - Total concessions as a percentage of total income and their trends
        - Allowance for doubtful accounts as a percentage of total income
        - Changes in total property expenses (3 months and 12 months)

        Given this context, please answer the following question:
        {user_input}

        Provide a thorough analysis based on the data available in the dataframe.
        """
    )

    agent = create_pandas_dataframe_agent(
        llm, 
        filtered_df, 
        verbose=True, 
        prefix=custom_prompt.template
    )

    # Chat interface
    st.markdown("## Chat with the Municipal Bond Analyst Agent")
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask about the municipal bonds"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            st_callback = StreamlitCallbackHandler(st.container())
            df_info = filtered_df.dtypes.to_string()
            full_prompt = custom_prompt.format(df_info=df_info, user_input=prompt)
            response = agent.run(full_prompt, callbacks=[st_callback])
            st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})

else:
    st.info("Please select at least one property to compare.")