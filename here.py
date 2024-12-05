class PDFTableExtractor:
    def __init__(self, pdf_path: str):
        """
        Initialize the PDF Table Extractor with a PDF file path.
        
        Args:
            pdf_path (str): Path to the PDF file
        """
        self.pdf_path = pdf_path
        self.embeddings = OpenAIEmbeddings()
        self.vector_store = None
        self.pages = None
        
    def load_and_process_pdf(self) -> None:
        """
        Load the PDF and create embeddings for each page's content.
        This will help in identifying relevant pages for table extraction.
        """
        # Load PDF using LangChain's loader
        loader = PyPDFLoader(self.pdf_path)
        pages = loader.load()
        
        # Split pages into chunks for better processing
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,
            length_function=len
        )
        splits = text_splitter.split_documents(pages)
        
        # Create vector store for similarity search
        self.vector_store = FAISS.from_documents(splits, self.embeddings)
        self.pages = pages
        
    def find_relevant_pages(self, table_description: str, top_k: int = 3) -> List[int]:
        """
        Find pages that likely contain the described table using semantic search.
        
        Args:
            table_description (str): Description of the table to find
            top_k (int): Number of most relevant pages to return
            
        Returns:
            List[int]: List of page numbers (0-based index)
        """
        if not self.vector_store:
            raise ValueError("Please run load_and_process_pdf() first")
            
        # Perform similarity search
        relevant_docs = self.vector_store.similarity_search(
            table_description,
            k=top_k
        )
        
        # Extract page numbers from metadata
        relevant_pages = [doc.metadata['page'] for doc in relevant_docs]
        return sorted(list(set(relevant_pages)))
    
    def extract_tables_from_page(self, page_number: int) -> List[pd.DataFrame]:
        """
        Extract all tables from a specific page using Camelot.
        
        Args:
            page_number (int): Page number to extract tables from (0-based index)
            
        Returns:
            List[pd.DataFrame]: List of extracted tables as pandas DataFrames
        """
        # Camelot uses 1-based page numbers
        camelot_page_num = page_number + 1
        
        # Extract tables using Camelot
        tables = camelot.read_pdf(
            self.pdf_path,
            pages=str(camelot_page_num),
            flavor='lattice'  # Use 'stream' if lattice doesn't work well
        )
        
        # Convert to pandas DataFrames and clean
        return [self._clean_table(table.df) for table in tables]
    
    def _clean_table(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and process extracted table DataFrame.
        
        Args:
            df (pd.DataFrame): Raw DataFrame from Camelot
            
        Returns:
            pd.DataFrame: Cleaned DataFrame
        """
        # Make a copy to avoid modifying the original
        cleaned_df = df.copy()
        
        # Remove empty rows and columns
        cleaned_df = cleaned_df.replace(r'^\s*$', np.nan, regex=True)
        cleaned_df = cleaned_df.dropna(how='all', axis=0)
        cleaned_df = cleaned_df.dropna(how='all', axis=1)
        
        # Use first row as header if it contains string values
        if cleaned_df.iloc[0].notna().all():
            cleaned_df.columns = cleaned_df.iloc[0]
            cleaned_df = cleaned_df.iloc[1:].reset_index(drop=True)
        
        return cleaned_df
    
    def extract_specific_table(self, table_description: str) -> Optional[pd.DataFrame]:
        """
        Extract a specific table based on its description.
        
        Args:
            table_description (str): Description of the table to find and extract
            
        Returns:
            Optional[pd.DataFrame]: Extracted table if found, None otherwise
        """
        # Find relevant pages
        relevant_pages = self.find_relevant_pages(table_description)
        
        if not relevant_pages:
            return None
        
        # Extract tables from relevant pages
        all_tables = []
        for page_num in relevant_pages:
            page_tables = self.extract_tables_from_page(page_num)
            all_tables.extend(page_tables)
        
        if not all_tables:
            return None
        
        # If multiple tables found, return the largest one (most rows/columns)
        # You might want to modify this heuristic based on your specific needs
        return max(all_tables, key=lambda df: df.size)

    def extract_all_tables(self) -> Dict[int, List[pd.DataFrame]]:
        """
        Extract all tables from the PDF, organized by page number.
        
        Returns:
            Dict[int, List[pd.DataFrame]]: Dictionary mapping page numbers to lists of tables
        """
        if not self.pages:
            raise ValueError("Please run load_and_process_pdf() first")
            
        all_tables = {}
        for page_num in range(len(self.pages)):
            tables = self.extract_tables_from_page(page_num)
            if tables:
                all_tables[page_num] = tables
                
        return all_tables

# Example usage:
def main():
    # Initialize extractor
    pdf_path = "your_document.pdf"
    extractor = PDFTableExtractor(pdf_path)
    
    # Load and process the PDF
    extractor.load_and_process_pdf()
    
    # Extract a specific table
    table_description = "A table showing quarterly revenue breakdown by region"
    specific_table = extractor.extract_specific_table(table_description)
    
    if specific_table is not None:
        print("Found table:")
        print(specific_table)
    else:
        print("No matching table found")
        
    # Extract all tables
    all_tables = extractor.extract_all_tables()
    print(f"Found {sum(len(tables) for tables in all_tables.values())} tables in total")

if __name__ == "__main__":
    main()