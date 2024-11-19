import tabula
import pandas as pd
import re

def clean_dataframe(df):
    # Drop rows where all elements are NaN
    df = df.dropna(how='all')
    
    # Reset index
    df = df.reset_index(drop=True)
    
    # Remove rows that contain only the page header
    df = df[~df.iloc[:,0].str.contains('Page', na=False)]
    df = df[~df.iloc[:,0].str.contains('HARTFORD FUNDS', na=False)]
    
    return df

def extract_and_organize_tables(pdf_path):
    # Read all tables from the PDF
    tables = tabula.read_pdf(pdf_path, pages='all', multiple_tables=True)
    
    # Initialize a dictionary to store asset-type grouped tables
    asset_groups = {}
    current_asset_type = None
    current_df = None
    
    # Define column names
    columns = ['Security', 'Coupon', 'Maturity', 'Shares/Par Value', '% of Net Assets']
    
    for table in tables:
        if table.empty:
            continue
            
        # Clean the table
        table = clean_dataframe(table)
        
        for idx, row in table.iterrows():
            # Convert row to string and check for asset type headers
            row_text = ' '.join([str(x) for x in row.values])
            
            # Check if this row is an asset type header
            if row_text.isupper() and not any(char.isdigit() for char in row_text):
                if current_asset_type and current_df is not None:
                    asset_groups[current_asset_type] = current_df
                
                current_asset_type = row_text.strip()
                current_df = pd.DataFrame(columns=columns)
                continue
            
            # If we have a current asset type, add data to the current dataframe
            if current_asset_type and not row.empty:
                try:
                    # Ensure row has correct number of columns
                    if len(row) >= 5:
                        new_row = pd.DataFrame([row.values[:5]], columns=columns)
                        current_df = pd.concat([current_df, new_row], ignore_index=True)
                except Exception as e:
                    print(f"Error processing row: {row}")
                    print(f"Error: {e}")
    
    # Add the last group
    if current_asset_type and current_df is not None:
        asset_groups[current_asset_type] = current_df
    
    return asset_groups

# Execute the extraction
pdf_path = "WRLDB.pdf"  # Replace with your PDF path
asset_tables = extract_and_organize_tables(pdf_path)

# Display results
for asset_type, df in asset_tables.items():
    print(f"\n{'='*80}")
    print(f"Asset Type: {asset_type}")
    print(f"{'='*80}")
    print(df.head())
    print(f"\nTotal entries: {len(df)}")