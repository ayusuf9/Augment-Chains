import pandas as pd
import re

# Function to clean up numeric values
def clean_numeric(value):
    if isinstance(value, str):
        # Remove commas and convert percentages
        value = value.replace(',', '')
        if '%' in value:
            value = float(value.replace('%', '')) / 100
        # Handle parentheses for negative numbers
        if '(' in value and ')' in value:
            value = -float(value.replace('(', '').replace(')', ''))
    return value

# Function to process the raw text and extract tables
def extract_tables(text):
    # Split the text into lines
    lines = text.split('\n')
    
    current_section = None
    sections_data = {}
    current_data = []
    
    # Column names for our tables
    columns = ['Security', 'Coupon', 'Maturity', 'Shares/Par Value', '% of Net Assets']
    
    for line in lines:
        # Skip empty lines and page headers/footers
        if not line.strip() or 'HARTFORD FUNDS' in line or 'Page' in line:
            continue
            
        # Check if this is a new section header (in all caps)
        if line.isupper() and 'SECURITIES' in line or 'BONDS' in line or 'CURRENCY' in line:
            if current_section and current_data:
                # Create DataFrame for previous section
                df = pd.DataFrame(current_data, columns=columns)
                sections_data[current_section] = df
            
            current_section = line.strip()
            current_data = []
            continue
        
        # Process data lines
        parts = line.split()
        if len(parts) >= 5 and any(c.isdigit() for c in line):
            try:
                # Extract values
                security = ' '.join(parts[:-4])
                coupon = clean_numeric(parts[-4])
                maturity = parts[-3]
                shares_value = clean_numeric(parts[-2])
                net_assets = clean_numeric(parts[-1])
                
                current_data.append([security, coupon, maturity, shares_value, net_assets])
            except:
                continue
    
    # Add the last section
    if current_section and current_data:
        df = pd.DataFrame(current_data, columns=columns)
        sections_data[current_section] = df
    
    return sections_data

# Read the text content
with open('sample_content.txt', 'r') as file:
    content = file.read()

# Extract tables
tables = extract_tables(content)

# Display tables by section
for section, df in tables.items():
    print(f"\n{'='*80}\n{section}\n{'='*80}")
    print(df.head())
    print(f"\nTotal rows in section: {len(df)}")