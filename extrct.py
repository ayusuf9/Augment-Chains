# Install required libraries if not already installed
# !pip install tabula-py pandas

from tabula import read_pdf
import pandas as pd

# File path to the uploaded PDF
file_path = "/mnt/data/WRLDB.pdf"

# Extract tables from the PDF
tables = read_pdf(file_path, pages="all", multiple_tables=True, stream=True)

# Define asset categories based on the structure of the PDF
asset_categories = [
    "ASSET & COMMERCIAL MORTGAGE-BACKED SECURITIES",
    "FOREIGN GOVERNMENT OBLIGATIONS",
    "FOREIGN CURRENCY",
    "CALL OPTIONS PURCHASED",
    "CONVERTIBLE BONDS",
    "CORPORATE BONDS",
    "PUT OPTIONS PURCHASED",
    "SENIOR FLOATING RATE INTERESTS",
]

# Dictionary to store data grouped by asset categories
grouped_assets = {category: [] for category in asset_categories}

# Helper function to identify asset categories in a table
def categorize_table(table, asset_categories):
    for category in asset_categories:
        if any(category in str(cell) for cell in table.iloc[:, 0]):
            return category
    return None

# Process each table and categorize
for table in tables:
    # Ensure the table is a valid DataFrame
    if not isinstance(table, pd.DataFrame) or table.empty:
        continue
    
    # Try to categorize the table
    category = categorize_table(table, asset_categories)
    
    if category:
        grouped_assets[category].append(table)

# Output grouped data to CSV files or display
for category, tables in grouped_assets.items():
    if tables:
        combined_table = pd.concat(tables, ignore_index=True)
        file_name = f"{category.replace('&', 'and').replace(' ', '_')}.csv"
        combined_table.to_csv(file_name, index=False)
        print(f"Saved {category} data to {file_name}.")
    else:
        print(f"No data found for {category}.")





Copyasset_categories = [
    "ASSET & COMMERCIAL MORTGAGE-BACKED SECURITIES",
    "BASE CURRENCY",
    "CALL OPTIONS PURCHASED",
    "COMMON STOCKS",
    "CONVERTIBLE BONDS", 
    "CORPORATE BONDS",
    "FOREIGN CURRENCY",
    "FOREIGN GOVERNMENT OBLIGATIONS",
    "PUT OPTIONS PURCHASED",
    "PUTS EXCHANGE-TRADED OPTIONS",
    "SENIOR FLOATING RATE INTERESTS",
    "SHORT-TERM INVESTMENTS",
    "U.S. GOVERNMENT AGENCIES",
    "U.S. GOVERNMENT SECURITIES",
    "WRITTEN OPTIONS PUTS",
    "DERIVATIVES - CENTRALLY CLEARED CREDIT DEFAULT SWAP CONTRACTS",
    "DERIVATIVES - CENTRALLY CLEARED INTEREST RATE SWAP CONTRACTS",
    "DERIVATIVES - FOREIGN CURRENCY CONTRACTS",
    "DERIVATIVES - FOREIGN CURRENCY EXCHANGE CONTRACTS (SPOT)",
    "DERIVATIVES - FUTURES CONTRACTS",
    "DERIVATIVES - OTC CREDIT DEFAULT SWAP CONTRACTS",
    "DERIVATIVES - OTC INTEREST RATE SWAP CONTRACTS",
    "DERIVATIVES - SWAPTIONS CALL",
    "DERIVATIVES - SWAPTIONS PUT",
    "DERIVATIVES - TBA SALE COMMITMENTS",
    "DERIVATIVES - WRITTEN SWAPTIONS PUT"
]
