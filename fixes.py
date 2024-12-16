import os
import re
from typing import Optional, List, Tuple
import pandas as pd
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import AzureChatOpenAI
from io import StringIO
import warnings
warnings.filterwarnings('ignore')

class PDFTableExtractor:
    def __init__(self, llm_client):
        self.embeddings = OpenAIEmbeddings(
            deployment='text-embedding-3-large',
            model='text-embedding-3-large', 
            openai_api_key=llm_client.api_key,
            openai_api_base=llm_client.api_base,
            openai_api_type=llm_client.api_type,
            chunk_size=100
        )
        self.llm = AzureChatOpenAI(
            deployment_name=llm_client.GPT_4_O1_MINI_MODEL,
            model_name='o1-mini', 
            openai_api_version="2024-09-01-preview",
            openai_api_key=llm_client.api_key,
            openai_api_base=llm_client.api_base,
            openai_api_type="azure_ad",
            temperature=0.0,  # Lower temperature for more consistent formatting
        )
    
    def load_pdf(self, pdf_path: str) -> List[str]:
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        loader = PyPDFLoader(pdf_path)
        return loader.load()
    
    def find_table_page(self, pages: List[str], table_description: str) -> Optional[tuple]:
        # Clean and normalize the table description
        table_description = re.sub(r'\s+', ' ', table_description).strip()
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,  # Increased chunk size to capture more context
            chunk_overlap=200  # Increased overlap to avoid missing table boundaries
        )
        splits = text_splitter.split_documents(pages)
        
        vectorstore = FAISS.from_documents(splits, self.embeddings)
        relevant_chunks = vectorstore.similarity_search(
            f"Find a table containing: {table_description}",
            k=4
        )
        
        if not relevant_chunks:
            return None
        
        # Search through all chunks to find the best match
        for chunk in relevant_chunks:
            for page_num, page in enumerate(pages):
                if chunk.page_content in page.page_content:
                    return (page.page_content, page_num + 1)
        
        return None
    
    def extract_table(self, page_content: str) -> pd.DataFrame:
        # Enhanced prompt for better table extraction
        prompt = ChatPromptTemplate.from_template("""
        Extract the table from the following text and format it as a CSV string.
        Requirements:
        1. Include ONLY the actual table data
        2. First row must be headers
        3. Use commas to separate columns
        4. Use newlines to separate rows
        5. Preserve all numerical values exactly as they appear
        6. Remove any special characters or formatting from cell values
        7. Ensure consistent number of columns across all rows
        
        Text content:
        {text}
        
        Respond ONLY with the CSV formatted table:
        """)
        
        messages = prompt.format_messages(text=page_content)
        
        try:
            response = self.llm.invoke(messages)
            csv_string = response.content.strip()
            
            # Validate CSV structure before creating DataFrame
            if not csv_string or ',' not in csv_string or '\n' not in csv_string:
                raise ValueError("Invalid CSV format in response")
            
            df = pd.read_csv(StringIO(csv_string))
            
            # Post-process the DataFrame
            df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
            df = df.replace(r'^\s*$', pd.NA, regex=True)
            
            return df
            
        except Exception as e:
            raise ValueError(f"Table extraction failed: {str(e)}\nResponse content: {csv_string}")
    
    def process_pdf(self, pdf_path: str, table_description: str = None) -> tuple[pd.DataFrame, int]:
        pages = self.load_pdf(pdf_path)
        
        if table_description:
            result = self.find_table_page(pages, table_description)
            if not result:
                raise ValueError("Could not find the specified table in the PDF")
            page_content, page_num = result
        else:
            if not pages:
                raise ValueError("PDF contains no pages")
            page_content = pages[0].page_content
            page_num = 1
        
        df = self.extract_table(page_content)
        
        if df.empty:
            raise ValueError(f"No table data was extracted from page {page_num}")
            
        return df, page_num

# Example usage:
def extract_table_from_pdf(pdf_path: str, table_description: str, llm_client) -> tuple[pd.DataFrame, int]:
    try:
        extractor = PDFTableExtractor(llm_client)
        df, page_num = extractor.process_pdf(pdf_path, table_description)
        return df, page_num
    except Exception as e:
        print(f"Error extracting table: {str(e)}")
        return None, None
    

### SECOND TWO ##################################################################################
### SECOND TWO ##################################################################################
### SECOND TWO ##################################################################################
### SECOND TWO ##################################################################################


import os
from typing import Optional, List, Tuple
import pandas as pd
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import AzureChatOpenAI
from io import StringIO
import warnings
warnings.filterwarnings('ignore')

class PDFTableExtractor:
    def __init__(self, llm_client):
        self.embeddings = OpenAIEmbeddings(
            deployment='text-embedding-3-large',
            model='text-embedding-3-large', 
            openai_api_key=llm_client.api_key,
            openai_api_base=llm_client.api_base,
            openai_api_type=llm_client.api_type,
            chunk_size=100
        )
        self.llm = AzureChatOpenAI(
            deployment_name=llm_client.GPT_4_O1_MINI_MODEL,
            model_name='o1-mini', 
            openai_api_version="2024-09-01-preview",
            openai_api_key=llm_client.api_key,
            openai_api_base=llm_client.api_base,
            openai_api_type="azure_ad",
            temperature=0.0,
        )
    
    def extract_table(self, page_content: str) -> pd.DataFrame:
        # Improved prompt specifically for this table format
        prompt = ChatPromptTemplate.from_template("""
        Extract the trading strategy table from the text and format it as a CSV string.
        Use these exact column headers in this order:
        Trade,Entry Level,Entry Date,Rationale,Risks
        
        Requirements:
        1. Ensure each cell's content is complete and not truncated
        2. Preserve numerical values exactly as they appear
        3. Use double quotes around any cell containing commas
        4. Keep all text in a single line within its cell
        5. Include ALL content from each cell
        
        Text content:
        {text}
        
        Respond ONLY with the CSV formatted table:
        """)
        
        messages = prompt.format_messages(text=page_content)
        
        try:
            response = self.llm.invoke(messages)
            csv_string = response.content.strip()
            
            # Custom parsing with specific column names
            df = pd.read_csv(
                StringIO(csv_string),
                dtype=str,  # Prevent automatic type conversion
                keep_default_na=False,  # Prevent NaN conversion
                quotechar='"',  # Handle quoted cells properly
                escapechar='\\'  # Handle escaped characters
            )
            
            # Ensure expected columns exist
            expected_columns = ['Trade', 'Entry Level', 'Entry Date', 'Rationale', 'Risks']
            if not all(col in df.columns for col in expected_columns):
                raise ValueError(f"Missing expected columns. Found: {df.columns.tolist()}")
            
            # Clean up the DataFrame
            df = df.apply(lambda x: x.str.strip() if isinstance(x, pd.Series) else x)
            
            # Reorder columns if needed
            df = df[expected_columns]
            
            return df
            
        except Exception as e:
            raise ValueError(f"Table extraction failed: {str(e)}\nResponse content: {csv_string}")
    
    def find_table_page(self, pages: List[str], table_description: str) -> Optional[tuple]:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,  # Increased for better context
            chunk_overlap=400
        )
        splits = text_splitter.split_documents(pages)
        
        vectorstore = FAISS.from_documents(splits, self.embeddings)
        relevant_chunks = vectorstore.similarity_search(
            f"Find a table containing trading strategies and risks: {table_description}",
            k=4
        )
        
        if not relevant_chunks:
            return None
        
        for chunk in relevant_chunks:
            for page_num, page in enumerate(pages):
                if chunk.page_content in page.page_content:
                    return (page.page_content, page_num + 1)
        
        return None
    
    def load_pdf(self, pdf_path: str) -> List[str]:
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        loader = PyPDFLoader(pdf_path)
        return loader.load()
    
    def process_pdf(self, pdf_path: str, table_description: str = None) -> tuple[pd.DataFrame, int]:
        pages = self.load_pdf(pdf_path)
        
        if table_description:
            result = self.find_table_page(pages, table_description)
            if not result:
                raise ValueError("Could not find the specified table in the PDF")
            page_content, page_num = result
        else:
            if not pages:
                raise ValueError("PDF contains no pages")
            page_content = pages[0].page_content
            page_num = 1
        
        df = self.extract_table(page_content)
        
        if df.empty:
            raise ValueError(f"No table data was extracted from page {page_num}")
            
        return df, page_num

# Usage example:
def extract_table_from_pdf(pdf_path: str, table_description: str, llm_client) -> tuple[pd.DataFrame, int]:
    try:
        extractor = PDFTableExtractor(llm_client)
        df, page_num = extractor.process_pdf(pdf_path, table_description)
        
        # Display settings for better output
        pd.set_option('display.max_colwidth', None)
        pd.set_option('display.max_rows', None)
        pd.set_option('display.width', None)
        
        return df, page_num
    except Exception as e:
        print(f"Error extracting table: {str(e)}")
        return None, None
    




### SECOND TWO ##################################################################################
### SECOND TWO ##################################################################################
### SECOND TWO ##################################################################################
### SECOND TWO ##################################################################################




def extract_table(self, page_content: str) -> pd.DataFrame:
    prompt = ChatPromptTemplate.from_template("""
    Extract the trading table data and return it in CSV format.
    Include these exact columns in order:
    Trade,Entry Level,Entry Date,Rationale,Risks
    
    Important:
    - Do NOT include any markdown formatting or code blocks
    - Do NOT include ```csv or ``` markers
    - Start directly with the header row
    - End with the last data row
    - Use commas as separators
    - Use quotes only when necessary (for cells containing commas)
    
    Text to extract from:
    {text}
    """)
    
    csv_string = ""
    try:
        # Get response from LLM
        messages = prompt.format_messages(text=page_content)
        response = self.llm.invoke(messages)
        
        if not response or not response.content:
            raise ValueError("Empty response from LLM")
            
        # Clean up the response - remove markdown code blocks
        csv_string = response.content.strip()
        csv_string = csv_string.replace('```csv', '').replace('```', '').strip()
        
        # Validate basic CSV structure
        if ',' not in csv_string or '\n' not in csv_string:
            raise ValueError("Response is not in valid CSV format")
        
        # Read the CSV data
        df = pd.read_csv(
            StringIO(csv_string),
            dtype=str,
            keep_default_na=False,
            quotechar='"',
            escapechar='\\'
        )
        
        # Clean column names
        df.columns = df.columns.str.strip()
        
        # Validate expected columns
        expected_columns = ['Trade', 'Entry Level', 'Entry Date', 'Rationale', 'Risks']
        missing_columns = [col for col in expected_columns if col not in df.columns]
        
        if missing_columns:
            # If columns are missing, try one more time with different column handling
            df = pd.read_csv(
                StringIO(csv_string),
                dtype=str,
                keep_default_na=False,
                quotechar='"',
                escapechar='\\',
                skipinitialspace=True,
                names=expected_columns,
                header=0
            )
        
        # Ensure we have all expected columns
        if not all(col in df.columns for col in expected_columns):
            raise ValueError(f"Missing expected columns. Found: {df.columns.tolist()}")
        
        # Clean up the DataFrame
        df = df[expected_columns]  # Reorder columns
        df = df.apply(lambda x: x.str.strip() if isinstance(x, pd.Series) else x)
        
        return df
        
    except Exception as e:
        error_details = f"""
        Error Details:
        - Original error: {str(e)}
        - Raw response content:
        {csv_string}
        
        - Columns found (if any): {df.columns.tolist() if 'df' in locals() else 'No DataFrame created'}
        """
        raise ValueError(f"Table extraction failed: {error_details}")

# Modified usage example to better handle the output:
def extract_and_format_table(pdf_path: str, table_description: str, llm_client) -> tuple[pd.DataFrame, int]:
    try:
        extractor = PDFTableExtractor(llm_client)
        df, page_num = extractor.process_pdf(pdf_path, table_description)
        
        if df is not None:
            # Set display options for better output
            pd.set_option('display.max_colwidth', None)
            pd.set_option('display.max_rows', None)
            pd.set_option('display.width', None)
            
            # Verify the data matches expected format
            for col in df.columns:
                if df[col].astype(str).str.contains('```').any():
                    df[col] = df[col].str.replace('```', '').str.strip()
            
            return df, page_num
            
    except Exception as e:
        print(f"Error extracting table: {str(e)}")
        return None, None
    

df, page_num = extract_and_format_table(
    pdf_path="/users/CFII_DataScience/USERs/SPTADM/LDO/MS 1.pdf",
    table_description="Table Title is 'Valuation Methodology and Risks'",
    llm_client=llm_client
)

if df is not None:
    print(f"\nExtracted table from page {page_num}:")
    print(df.to_string(index=False))