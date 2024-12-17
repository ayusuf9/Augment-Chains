def extract_table(self, page_content: str) -> pd.DataFrame:
    """Extract JP Morgan Global Rates Strategy table with specific formatting."""
    prompt = ChatPromptTemplate.from_template("""
    Extract the Global Rates trade recommendations table with these exact columns:
    Region,Duration,Curve,Swap_spreads,Swap_spread_curve,Gamma,Vega,Inflation,Cross_market

    Important formatting rules:
    1. Each row starts with one of these regions: Euro area, UK, US, Japan, Australia / New Zealand, Scandinavia
    2. Keep "Neutral" entries as is
    3. Preserve exact trade descriptions
    4. Use comma separation
    5. Handle empty cells with "-"
    6. Do not modify or summarize the trade descriptions
    7. Maintain all numerical values and symbols exactly as shown
    8. Use quotes for cells containing commas
    
    Source text:
    {text}

    Respond ONLY with the CSV formatted data, starting with the header row.
    """)
    
    try:
        messages = prompt.format_messages(text=page_content)
        response = self.llm.invoke(messages)
        
        if not response or not response.content:
            raise ValueError("Empty response from LLM")
            
        # Clean up the response
        csv_string = response.content.strip()
        csv_string = csv_string.replace('```csv', '').replace('```', '').strip()
        
        # Define expected columns matching the table structure
        expected_columns = [
            'Region', 'Duration', 'Curve', 'Swap_spreads', 
            'Swap_spread_curve', 'Gamma', 'Vega', 'Inflation', 'Cross_market'
        ]
        
        # Read the CSV with specific handling for this table format
        df = pd.read_csv(
            StringIO(csv_string),
            dtype=str,
            keep_default_na=False,
            quotechar='"',
            escapechar='\\',
            names=expected_columns,
            skip_blank_lines=True
        )
        
        # Clean up the DataFrame
        df = df.apply(lambda x: x.str.strip() if isinstance(x, pd.Series) else x)
        
        # Fill empty or whitespace cells with "-"
        df = df.replace(r'^\s*$', '-', regex=True)
        
        # Validate regions
        expected_regions = [
            'Euro area', 'UK', 'US', 'Japan', 
            'Australia / New Zealand', 'Scandinavia'
        ]
        
        if not df['Region'].isin(expected_regions).any():
            raise ValueError("No valid regions found in the extracted data")
        
        # Validate each row has content
        if df.apply(lambda x: x.str.len() > 0).sum().sum() < len(df) * 2:
            raise ValueError("Some rows appear to be empty or invalid")
        
        return df
        
    except Exception as e:
        error_details = f"""
        Error Details:
        - Original error: {str(e)}
        - Raw response content:
        {csv_string if 'csv_string' in locals() else 'No response generated'}
        """
        raise ValueError(f"Table extraction failed: {error_details}")

def process_pdf(self, pdf_path: str, table_description: str = None) -> tuple[pd.DataFrame, int]:
    """Process PDF with specific handling for JP Morgan table format."""
    pages = self.load_pdf(pdf_path)
    
    # Look for table with "Figure 13: Current Global Rates trade recommendations"
    table_pages = []
    for page_num, page in enumerate(pages):
        if "Current Global Rates trade recommendations" in page.page_content:
            table_pages.append((page.page_content, page_num + 1))
            # Check next page in case table continues
            if page_num + 1 < len(pages):
                table_pages.append((pages[page_num + 1].page_content, page_num + 2))
    
    if not table_pages:
        raise ValueError("Could not find Global Rates trade recommendations table")
    
    # Combine content from all found pages
    combined_content = "\n".join(content for content, _ in table_pages)
    page_numbers = [page_num for _, page_num in table_pages]
    
    df = self.extract_table(combined_content)
    
    return df, page_numbers



from quanthub.util import llm2

llm_client = llm2.get_llm_client()

try:
    extractor = PDFTableExtractor(llm_client)
    df, page_nums = extractor.process_pdf(
        pdf_path="path_to_your_jpmorgan_pdf.pdf",
        table_description="Figure 13: Current Global Rates trade recommendations"
    )
    
    if df is not None:
        print(f"\nExtracted table from pages {page_nums}:")
        pd.set_option('display.max_columns', None)
        pd.set_option('display.max_rows', None)
        pd.set_option('display.width', None)
        pd.set_option('display.max_colwidth', None)
        print(df.to_string(index=False))
        
        # Save to CSV if needed
        df.to_csv('jpmorgan_rates_table.csv', index=False)
        
except Exception as e:
    print(f"Error occurred: {str(e)}")