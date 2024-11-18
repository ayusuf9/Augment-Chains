import os
import json
import streamlit as st
import pandas as pd
from datetime import datetime
from streamlit_navigation_bar import st_navbar
from streamlit_extras.switch_page_button import switch_page

# Configuration
CACHE_FILE = "qa_preprocessed_cache.json"
CUSTOM_CACHE_FILE = "custom_qa_cache.json"

# Page configuration
st.set_page_config(
    layout="wide",
    page_icon="📑",
    page_title="Property Analysis Dashboard",
    initial_sidebar_state="collapsed"
)

# Keep existing styles...

def load_cached_results():
    """Load pre-processed results from cache file"""
    try:
        if os.path.exists(CACHE_FILE):
            with open(CACHE_FILE, 'r') as f:
                cache_data = json.load(f)
                results_dict = cache_data.get('results', {})
                return pd.DataFrame.from_dict(results_dict)
        else:
            st.error(f"Cache file {CACHE_FILE} not found!")
            return pd.DataFrame()
    except Exception as e:
        st.error(f"Error loading cache: {str(e)}")
        return pd.DataFrame()

def load_custom_cache():
    """Load custom questions cache"""
    try:
        if os.path.exists(CUSTOM_CACHE_FILE):
            with open(CUSTOM_CACHE_FILE, 'r') as f:
                return json.load(f)
        return {'questions': [], 'results': {}}
    except Exception:
        return {'questions': [], 'results': {}}

def save_custom_cache(cache_data):
    """Save custom questions cache"""
    try:
        with open(CUSTOM_CACHE_FILE, 'w') as f:
            json.dump(cache_data, f)
    except Exception as e:
        st.error(f"Error saving custom cache: {str(e)}")

def style_dataframe(df):
    # Keep existing style_dataframe function...
    pass

def initialize_session_state():
    """Initialize session state variables"""
    if 'selected_docs' not in st.session_state:
        st.session_state.selected_docs = []
    if 'results_df' not in st.session_state:
        st.session_state.results_df = None
    if 'custom_questions' not in st.session_state:
        st.session_state.custom_questions = []
    if 'custom_results' not in st.session_state:
        st.session_state.custom_results = {}

def main():
    initialize_session_state()
    
    # Navigation bar
    page = st_navbar(["Home", "Chat", "Tables"], selected="Tables", styles=styles)
    
    if page == "Home":
        switch_page("streamlit app")
    if page == "Chat":
        switch_page("chat")

    st.title("Property Analysis Dashboard")
    
    # Load both pre-processed and custom results
    if st.session_state.results_df is None:
        st.session_state.results_df = load_cached_results()
    
    results_df = st.session_state.results_df
    
    if results_df.empty:
        st.error("No pre-processed data available. Please contact administrator.")
        return
    
    # Load custom cache
    custom_cache = load_custom_cache()
    st.session_state.custom_questions = custom_cache['questions']
    st.session_state.custom_results = custom_cache['results']
    
    # Get available documents from cached results
    available_identifiers = results_df.columns.tolist()
    
    # Document selection with session state persistence
    selected_identifiers = st.multiselect(
        "Select documents to analyze:",
        available_identifiers,
        default=st.session_state.selected_docs
    )
    st.session_state.selected_docs = selected_identifiers

    # Custom question input
    st.markdown("### Add Custom Question")
    col1, col2 = st.columns([3, 1])
    
    with col1:
        new_question = st.text_input("Enter your question:", key="new_question")
    
    with col2:
        if st.button("Add Question"):
            if new_question.strip():
                if new_question not in st.session_state.custom_questions:
                    st.session_state.custom_questions.append(new_question)
                    # Save to custom cache
                    custom_cache['questions'] = st.session_state.custom_questions
                    save_custom_cache(custom_cache)
                    st.success("Question added successfully!")
                else:
                    st.warning("This question has already been added.")
            else:
                st.warning("Please enter a question before adding.")

    # Display results
    if selected_identifiers:
        # Combine pre-processed and custom questions
        all_questions = list(results_df.index) + st.session_state.custom_questions
        
        # Create combined results DataFrame
        combined_results = results_df.copy()
        
        # Add custom questions to the DataFrame
        for question in st.session_state.custom_questions:
            if question not in combined_results.index:
                new_row = pd.Series("Not processed yet", index=combined_results.columns)
                combined_results.loc[question] = new_row
        
        # Display selected columns
        display_df = combined_results[selected_identifiers]
        
        # Create tabs for different views
        tab1, tab2 = st.tabs(["All Questions", "Custom Questions Only"])
        
        with tab1:
            st.markdown("### All Questions")
            styled_df = style_dataframe(display_df)
            st.write(styled_df.to_html(escape=False), unsafe_allow_html=True)
        
        with tab2:
            if st.session_state.custom_questions:
                st.markdown("### Custom Questions")
                custom_df = display_df.loc[st.session_state.custom_questions]
                styled_custom_df = style_dataframe(custom_df)
                st.write(styled_custom_df.to_html(escape=False), unsafe_allow_html=True)
            else:
                st.info("No custom questions added yet.")
        
        # Export functionality
        st.sidebar.markdown("### Export Options")
        export_option = st.sidebar.radio(
            "Choose what to export:",
            ["All Questions", "Custom Questions Only", "Default Questions Only"]
        )
        
        if st.sidebar.button("Export Results"):
            if export_option == "All Questions":
                export_df = display_df
            elif export_option == "Custom Questions Only":
                export_df = display_df.loc[st.session_state.custom_questions]
            else:
                export_df = display_df.loc[list(results_df.index)]
            
            csv = export_df.to_csv().encode('utf-8')
            st.sidebar.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"property_analysis_{datetime.now().strftime('%Y%m%d')}_{export_option}.csv",
                mime="text/csv"
            )
        
        # Display metadata
        st.sidebar.markdown("### Analysis Details")
        st.sidebar.markdown(f"**Selected Properties:** {len(selected_identifiers)}")
        st.sidebar.markdown(f"**Total Questions:** {len(display_df.index)}")
        st.sidebar.markdown(f"**Custom Questions:** {len(st.session_state.custom_questions)}")
        
    else:
        st.info("Please select one or more documents to view the analysis.")

    # Add footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666;'>
        For support, please contact your administrator
        </div>
        """, 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()