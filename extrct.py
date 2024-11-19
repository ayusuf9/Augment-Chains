import fitz  # PyMuPDF
import pandas as pd

def extract_tables_from_pdf(pdf_path):
    """
    Extracts tables from a PDF, grouped by asset type.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        A dictionary where keys are asset types and values are pandas DataFrames.
    """

    doc = fitz.open(pdf_path)
    tables_by_asset = {}
    current_asset = None
    current_table_data = []

    for page_num in range(doc.page_count):
        page = doc[page_num]
        blocks = page.get_text("dict", flags=fitz.TEXT_DEHYPHENATE)["blocks"]

        for block in blocks:
            if "lines" in block:  # Check if it's a text block
                for line in block["lines"]:
                    for span in line["spans"]:
                        text = span["text"].strip()
                        if span["color"] == 2631895 and text != "HARTFORD FUNDS": # Blue separator with check
                            if current_table_data: # Append data before new asset
                                if current_asset not in tables_by_asset:
                                    tables_by_asset[current_asset] = []
                                df = pd.DataFrame(current_table_data)
                                # Forward fill the asset name and cleanup
                                df[0] = df[0].replace("", method='ffill')
                                df = df[df[0] == current_asset]
                                df = df.drop(0, axis=1).dropna(how='all') # Remove asset name col, drop empty rows
                                tables_by_asset[current_asset].append(df) # Append the df, not the list
                                current_table_data = []  # Start a new table
                            current_asset = text  #  Use separator as the next asset name
                        elif current_asset: # Only if under an asset section header
                            if text != "Page" and not text.startswith("09.30.24"):  # Exclude page numbers and dates
                                current_table_data.append(text.split())


    # Handle the last table (if any)
    if current_table_data:
        if current_asset not in tables_by_asset:
            tables_by_asset[current_asset] = []
        df = pd.DataFrame(current_table_data)
        # Forward fill the asset name and cleanup
        df[0] = df[0].replace("", method='ffill')
        df = df[df[0] == current_asset]
        df = df.drop(0, axis=1).dropna(how='all') # Remove asset name col, drop empty rows
        tables_by_asset[current_asset].append(df) # Append the df, not the list
        

    # Concatenate the tables within each asset for multi-page tables
    for asset, tables in tables_by_asset.items():
        tables_by_asset[asset] = pd.concat(tables, ignore_index=True) # Concat list of dfs

    return tables_by_asset


if __name__ == "__main__":
    pdf_file = "WRLDB.pdf"
    extracted_tables = extract_tables_from_pdf(pdf_file)

    for asset_type, table in extracted_tables.items():
        print(f"\nTable for {asset_type}:")
        display(table) # Display in Jupyter uses better formatting than print

        # Or save to CSV:
        #table.to_csv(f"{asset_type}.csv", index=False) 