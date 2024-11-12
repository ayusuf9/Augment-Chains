import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import ast
from collections import Counter
import re
from textblob import TextBlob
import warnings
warnings.filterwarnings('ignore')

def perform_patent_eda(df):
    """
    Comprehensive EDA function for patent data
    """
    # 1. Basic Data Cleaning
    df_clean = (df
        .dropna(subset=['patent_abstract'])  # Drop rows with missing abstracts
        .assign(
            # Convert date to datetime
            patent_date=lambda x: pd.to_datetime(x['patent_date']),
            # Convert string representation of list to actual list
            assignees=lambda x: x['assignees'].apply(ast.literal_eval),
            # Extract assignee organization
            primary_assignee=lambda x: x['assignees'].apply(lambda y: y[0]['assignee_organization'] if y else None),
            # Extract assignee country
            assignee_country=lambda x: x['assignees'].apply(lambda y: y[0]['assignee_country'] if y else None),
            # Calculate abstract length
            abstract_length=lambda x: x['patent_abstract'].str.len(),
            # Extract year from patent date
            patent_year=lambda x: x['patent_date'].dt.year
        )
    )
    
    # 2. Time Series Analysis
    patents_by_year = (df_clean
        .groupby('patent_year')
        .size()
        .reset_index(name='count')
    )
    
    # 3. Top Assignees Analysis
    top_assignees = (df_clean
        .groupby('primary_assignee')
        .size()
        .sort_values(ascending=False)
        .head(10)
    )
    
    # 4. Country Distribution
    country_dist = (df_clean
        .groupby('assignee_country')
        .size()
        .sort_values(ascending=False)
    )
    
    # 5. Abstract Text Analysis
    def extract_tech_keywords(text):
        # Common technical keywords in patents
        tech_keywords = ['artificial intelligence', 'machine learning', 'neural network', 
                        'data processing', 'system', 'method', 'device', 'apparatus']
        return [word for word in tech_keywords if word.lower() in text.lower()]
    
    tech_analysis = (df_clean
        .assign(
            keywords=lambda x: x['patent_abstract'].apply(extract_tech_keywords),
            sentiment=lambda x: x['patent_abstract'].apply(lambda text: TextBlob(text).sentiment.polarity)
        )
    )
    
    # 6. Calculate Key Metrics
    metrics = {
        'total_patents': len(df_clean),
        'date_range': f"{df_clean['patent_date'].min().strftime('%Y-%m-%d')} to {df_clean['patent_date'].max().strftime('%Y-%m-%d')}",
        'unique_assignees': df_clean['primary_assignee'].nunique(),
        'countries_count': df_clean['assignee_country'].nunique(),
        'avg_abstract_length': df_clean['abstract_length'].mean(),
        'median_abstract_length': df_clean['abstract_length'].median()
    }
    
    # 7. Generate plots
    plt.figure(figsize=(15, 10))
    
    # Patents over time
    plt.subplot(2, 2, 1)
    sns.lineplot(data=patents_by_year, x='patent_year', y='count')
    plt.title('Patents Over Time')
    plt.xticks(rotation=45)
    
    # Top assignees
    plt.subplot(2, 2, 2)
    top_assignees.plot(kind='bar')
    plt.title('Top 10 Patent Assignees')
    plt.xticks(rotation=45, ha='right')
    
    # Abstract length distribution
    plt.subplot(2, 2, 3)
    sns.histplot(data=df_clean, x='abstract_length', bins=50)
    plt.title('Distribution of Abstract Lengths')
    
    # Sentiment distribution
    plt.subplot(2, 2, 4)
    sns.histplot(data=tech_analysis, x='sentiment', bins=30)
    plt.title('Distribution of Abstract Sentiments')
    
    plt.tight_layout()
    
    return {
        'clean_data': df_clean,
        'metrics': metrics,
        'tech_analysis': tech_analysis,
        'time_series': patents_by_year,
        'top_assignees': top_assignees,
        'country_distribution': country_dist
    }

# Example usage:
# results = perform_patent_eda(df)
# To access individual components:
# clean_data = results['clean_data']
# metrics = results['metrics']
# etc.