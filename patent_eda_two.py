import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import ast
from collections import Counter
import re
from sklearn.feature_extraction.text import TfidfVectorizer
import warnings
warnings.filterwarnings('ignore')

def analyze_patent_text(text):
    """
    Analyzes patent text for key characteristics
    """
    results = {}
    
    # 1. Innovation claim markers
    innovation_markers = [
        'novel', 'innovative', 'improvement', 'enhanced', 'efficient',
        'unique', 'advanced', 'breakthrough', 'new', 'improved'
    ]
    results['innovation_score'] = sum(1 for word in innovation_markers if word in text.lower())
    
    # 2. Technical domain identification
    domain_keywords = {
        'AI/ML': ['artificial intelligence', 'machine learning', 'neural network', 'deep learning'],
        'Software': ['software', 'algorithm', 'program', 'code', 'application'],
        'Hardware': ['device', 'hardware', 'circuit', 'processor', 'semiconductor'],
        'Communication': ['network', 'protocol', 'transmission', 'wireless', 'signal'],
        'Data Processing': ['database', 'data processing', 'analytics', 'information system']
    }
    
    results['domains'] = [domain for domain, keywords in domain_keywords.items() 
                         if any(keyword in text.lower() for keyword in keywords)]
    
    # 3. Method identification (common patent components)
    method_markers = ['method', 'system', 'apparatus', 'process', 'technique']
    results['patent_type'] = [marker for marker in method_markers if marker in text.lower()]
    
    return results

def perform_patent_eda(df):
    """
    Comprehensive EDA function for patent data with enhanced text analysis
    """
    # 1. Basic Data Cleaning
    df_clean = (df
        .dropna(subset=['patent_abstract'])
        .assign(
            patent_date=lambda x: pd.to_datetime(x['patent_date']),
            assignees=lambda x: x['assignees'].apply(ast.literal_eval),
            primary_assignee=lambda x: x['assignees'].apply(lambda y: y[0]['assignee_organization'] if y else None),
            assignee_country=lambda x: x['assignees'].apply(lambda y: y[0]['assignee_country'] if y else None),
            abstract_length=lambda x: x['patent_abstract'].str.len(),
            patent_year=lambda x: x['patent_date'].dt.year
        )
    )
    
    # 2. Enhanced Text Analysis
    text_analysis = pd.DataFrame([
        analyze_patent_text(text) for text in df_clean['patent_abstract']
    ])
    
    df_analyzed = pd.concat([df_clean, text_analysis], axis=1)
    
    # 3. Technical Domain Analysis
    domain_counts = (df_analyzed
        .explode('domains')
        .groupby('domains')
        .size()
        .sort_values(ascending=False)
    )
    
    # 4. Innovation Score Analysis
    innovation_metrics = {
        'mean_innovation_score': df_analyzed['innovation_score'].mean(),
        'high_innovation_patents': sum(df_analyzed['innovation_score'] > 2),
        'low_innovation_patents': sum(df_analyzed['innovation_score'] == 0)
    }
    
    # 5. Patent Type Analysis
    patent_types = (df_analyzed
        .explode('patent_type')
        .groupby('patent_type')
        .size()
        .sort_values(ascending=False)
    )
    
    # 6. Time Series Analysis by Domain
    time_domain_analysis = (df_analyzed
        .explode('domains')
        .groupby(['patent_year', 'domains'])
        .size()
        .unstack(fill_value=0)
    )
    
    # 7. Visualizations
    plt.figure(figsize=(20, 15))
    
    # Technical Domains Distribution
    plt.subplot(2, 2, 1)
    domain_counts.plot(kind='bar')
    plt.title('Distribution of Technical Domains')
    plt.xticks(rotation=45, ha='right')
    
    # Innovation Scores Distribution
    plt.subplot(2, 2, 2)
    sns.histplot(data=df_analyzed, x='innovation_score', bins=10)
    plt.title('Distribution of Innovation Scores')
    
    # Patent Types
    plt.subplot(2, 2, 3)
    patent_types.plot(kind='bar')
    plt.title('Distribution of Patent Types')
    plt.xticks(rotation=45, ha='right')
    
    # Domains over time
    plt.subplot(2, 2, 4)
    time_domain_analysis.plot(kind='line', marker='o')
    plt.title('Technical Domains Over Time')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    
    return {
        'clean_data': df_clean,
        'domain_analysis': domain_counts,
        'innovation_metrics': innovation_metrics,
        'patent_types': patent_types,
        'time_domain_analysis': time_domain_analysis,
        'analyzed_data': df_analyzed
    }

# Example usage:
# results = perform_patent_eda(df)