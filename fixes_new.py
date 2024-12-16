def extract_table(self, page_content: str) -> pd.DataFrame:
    """Extract hierarchical table with region groupings."""
    prompt = ChatPromptTemplate.from_template("""
    Extract the hierarchical trading table data and return it in a normalized CSV format.
    
    Rules:
    1. Each row should have these columns:
    Region,Strategy,Duration,Curve,Swap_spreads,Swap_spread_curve,Gamma,Vega,Inflation,Cross_market
    2. The Region column should contain the main region (Euro area, UK, Japan, etc.)
    3. The Strategy column should contain the actual trade/strategy
    4. Empty cells should be filled with 'Neutral' or '-' as appropriate
    5. Do NOT include any markdown formatting
    6. Use commas as separators and quotes for cells containing commas
    7. Ensure each row has exactly the same number of columns
    
    Text to extract from:
    {text}
    """)
    
    csv_string = ""
    try:
        messages = prompt.format_messages(text=page_content)
        response = self.llm.invoke(messages)
        
        if not response or not response.content:
            raise ValueError("Empty response from LLM")
            
        # Clean up the response
        csv_string = response.content.strip()
        csv_string = csv_string.replace('```csv', '').replace('```', '').strip()
        
        # Define expected columns
        expected_columns = [
            'Region', 'Strategy', 'Duration', 'Curve', 'Swap_spreads',
            'Swap_spread_curve', 'Gamma', 'Vega', 'Inflation', 'Cross_market'
        ]
        
        # Read CSV with specific handling for hierarchical structure
        df = pd.read_csv(
            StringIO(csv_string),
            dtype=str,
            keep_default_na=False,
            quotechar='"',
            escapechar='\\',
            names=expected_columns,
            header=0,
            skip_blank_lines=True
        )
        
        # Clean up the DataFrame
        df = df.apply(lambda x: x.str.strip() if isinstance(x, pd.Series) else x)
        
        # Fill empty cells
        df = df.replace(r'^\s*$', '-', regex=True)
        
        # Forward fill region names
        df['Region'] = df['Region'].replace('-', pd.NA).fillna(method='ffill')
        
        # Remove rows where Strategy is empty or just whitespace
        df = df[df['Strategy'].str.strip() != '']
        
        return df
        
    except pd.errors.EmptyDataError:
        raise ValueError("No data found in the table")
    except pd.errors.ParserError as e:
        # Try alternative parsing approach for irregular structures
        try:
            # Read raw lines and process manually
            lines = csv_string.split('\n')
            processed_lines = []
            current_region = ""
            
            for line in lines:
                if line.strip():
                    parts = line.split(',')
                    if parts[0].strip():  # New region
                        current_region = parts[0].strip()
                    else:
                        parts[0] = current_region
                    
                    # Ensure exactly 10 columns
                    while len(parts) < 10:
                        parts.append('-')
                    parts = parts[:10]  # Trim if too many
                    
                    processed_lines.append(','.join(f'"{p.strip()}"' for p in parts))
            
            processed_csv = '\n'.join(processed_lines)
            df = pd.read_csv(
                StringIO(processed_csv),
                names=expected_columns,
                dtype=str,
                keep_default_na=False
            )
            return df
            
        except Exception as nested_e:
            raise ValueError(f"Failed to parse irregular table structure: {str(nested_e)}")
    except Exception as e:
        error_details = f"""
        Error Details:
        - Original error: {str(e)}
        - Raw response content:
        {csv_string}
        """
        raise ValueError(f"Table extraction failed: {error_details}")
    

    from quanthub.util import llm2

llm_client = llm2.get_llm_client()

try:
    extractor = PDFTableExtractor(llm_client)
    df, page_nums = extractor.process_pdf(
        pdf_path="your_pdf_path.pdf",
        table_description="Table with regional trading strategies"
    )
    
    if df is not None:
        print(f"\nExtracted table from pages {page_nums}:")
        # Format display for better readability
        pd.set_option('display.max_columns', None)
        pd.set_option('display.max_rows', None)
        pd.set_option('display.width', None)
        pd.set_option('display.max_colwidth', None)
        print(df.to_string(index=False))
        
        # Optionally save to CSV
        df.to_csv('extracted_table.csv', index=False)
        
except Exception as e:
    print(f"Error occurred: {str(e)}")