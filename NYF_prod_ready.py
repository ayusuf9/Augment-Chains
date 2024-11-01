import pandas as pd
import tabula.io
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
import logging
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import warnings
import os
from langchain.chat_models import AzureChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.callbacks import get_openai_callback
from datetime import datetime

# Suppress warnings
warnings.filterwarnings("ignore")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'portfolio_analysis_{datetime.now().strftime("%Y%m%d")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class PortfolioConfig:
    """Configuration settings for portfolio analysis."""
    OUTPUT_PRICE_PER_1K_TOKENS: float = 0.015
    PDF_FILE: str = 'analytics_report_20240827.pdf'
    CSV_TEMPLATE: str = 'Risk Contributions/var&asofdate=20240830&portfolio={}&var_type=AVaR_WKY'
    PORTFOLIOS: List[str] = (
        "BFA", "CGCB", "AMBAL-P-FI", "VIAA-P-IG", "CIB-P", "IFA-P-IG",
        "SBF", "CGCP", "STBF", "IBFA", "CGSD", "CGUI", "GVT", "AFMF", "ILBF"
    )
    STRESS_TEST_COLUMNS: List[str] = [
        'Active Stress P&L-SP500 Down 20',
        'Active Stress P&L-Inflation Overshoot'
    ]
    OUTPUT_DIR: str = 'output'
    
    def __post_init__(self):
        """Create output directory if it doesn't exist."""
        os.makedirs(self.OUTPUT_DIR, exist_ok=True)

class RiskAnalyzer:
    """Handles risk analysis calculations and classifications."""
    
    @staticmethod
    def classify_tracking_error(te_value: float, alpha_target: float) -> str:
        """Classify tracking error based on alpha target threshold."""
        try:
            if te_value > 2 * alpha_target:
                return "Profit"
            elif te_value < -2 * alpha_target:
                return "Loss"
            return "Normal"
        except Exception as e:
            logger.error(f"Error in tracking error classification: {e}")
            return "Unknown"

    @staticmethod
    def analyze_stress_tests(row: pd.Series, stress_columns: List[str]) -> List[Dict]:
        """Analyze stress test results."""
        stress_analysis = []
        for col in stress_columns:
            try:
                if col in row:
                    value = float(row[col])
                    stress_analysis.append({
                        'scenario': col,
                        'value': value,
                        'impact': 'Positive' if value > 0 else 'Negative' if value < 0 else 'Neutral',
                        'magnitude': abs(value)
                    })
            except Exception as e:
                logger.error(f"Error analyzing stress test {col}: {e}")
        return stress_analysis

class DataCleaner:
    """Handles data cleaning and standardization."""
    
    @staticmethod
    def standardize_column_names(df: pd.DataFrame) -> pd.DataFrame:
        """Standardize column names across the dataset."""
        column_mappings = {
            'USD Rate Derivati ves': 'USD Rate Derivatives',
            'Non- USD Rate Derivati ves': 'Non-USD Rate Derivatives',
            'Structu red Credit': 'Structured Credit',
            '15Y- 25Y': '15Y-25Y',
            'Active Stress P&L - SP500 Down 20': 'Active Stress P&L-SP500 Down 20',
            'Ex- Ante Tracki ng Error (bp/yr)': 'Ex-Ante Tracking Error (bp/yr)',
            'Active Stress P&L - Inflation Over- shoot': 'Active Stress P&L-Inflation Overshoot'
        }
        
        df = df.copy()
        df.columns = df.columns.str.replace(r'\r', ' ', regex=True)
        df.columns = df.columns.str.replace(r'\s+', ' ', regex=True)
        df.columns = df.columns.str.strip()
        
        for old, new in column_mappings.items():
            df.columns = df.columns.str.replace(old, new, regex=True)
            
        return df

class PDFExtractor:
    """Handles PDF data extraction and processing."""
    
    @staticmethod
    def extract_tables(pdf_path: str) -> Optional[pd.DataFrame]:
        """Extract tables from PDF and process them."""
        try:
            logger.info(f"Extracting tables from {pdf_path}")
            tables = tabula.io.read_pdf(pdf_path, pages='1', lattice=True)
            
            if not tables:
                raise ValueError("No tables found in PDF")
                
            tdf = tables[0]
            tdf = tdf.dropna(axis=1, how='all')
            tdf.columns = tdf.iloc[0]
            tdf = tdf.drop(0).reset_index(drop=True)
            
            return DataCleaner.standardize_column_names(tdf)
            
        except Exception as e:
            logger.error(f"Error extracting PDF data: {e}")
            return None

class PortfolioProcessor:
    """Handles portfolio data processing and analysis."""
    
    @staticmethod
    def process_csv(file_path: str, portfolio: str) -> Optional[pd.DataFrame]:
        """Process individual portfolio CSV file."""
        try:
            logger.info(f"Processing portfolio {portfolio}")
            df = pd.read_csv(file_path)
            
            # Filter and pivot
            df = df[df['Level'] == '1'][['Fund', 'Filter Level 1', 'Risk Contribution']]
            pivot_df = df.pivot(
                index='Fund',
                columns='Filter Level 1',
                values='Risk Contribution'
            )
            
            # Clean up and add portfolio identifier
            pivot_df = pivot_df.reset_index(drop=True)
            pivot_df = pivot_df.add_prefix('Tracking Error (T/E) Contribution - ')
            pivot_df['Portfolio'] = portfolio
            
            return pivot_df
            
        except Exception as e:
            logger.error(f"Error processing portfolio {portfolio}: {e}")
            return None

    @classmethod
    def process_portfolios(cls, config: PortfolioConfig) -> pd.DataFrame:
        """Process all portfolios in parallel."""
        dfs = []
        
        with ThreadPoolExecutor() as executor:
            futures = []
            for portfolio in config.PORTFOLIOS:
                file_path = config.CSV_TEMPLATE.format(portfolio)
                futures.append(
                    executor.submit(cls.process_csv, file_path, portfolio)
                )
            
            for future in futures:
                result = future.result()
                if result is not None:
                    dfs.append(result)
        
        return pd.concat(dfs, ignore_index=True)

class PortfolioAnalyzer:
    """Main class for portfolio analysis."""
    
    def __init__(self, config: PortfolioConfig):
        self.config = config
        self.prompt_template = self._create_prompt_template()
        
    def _create_prompt_template(self) -> PromptTemplate:
        """Create analysis prompt template."""
        template = """
        You are a highly qualified Risk and Quant Solution Analyst with expertise in Fixed Income, tasked with creating a comprehensive risk analysis summary focusing on tracking errors and stress test scenarios.

        Portfolio Name: {portfolio_name}

        Portfolio Data:
        {portfolio_data}

        Risk Analysis Focus:

        1. Tracking Error (T/E) Analysis:
        - HIGHLIGHT when T/E > 2 * alpha target (Profit scenario)
        - HIGHLIGHT when T/E < -2 * alpha target (Loss scenario)
        - Analyze contributing factors to T/E movements
        - Break down primary and secondary risk drivers

        2. Stress Test P&L Analysis:
        - Focus on significant stress test scenarios (SP500 Down 20, Inflation Overshoot)
        - Highlight extreme outcomes (both positive and negative)
        - Analyze portfolio resilience in stress scenarios
        - Identify key risk factors driving stress results

        3. Key Risk Metrics Deep Dive:
        - Active Duration positions and their impact
        - Spread Duration exposures across sectors
        - Credit quality distribution impacts
        - Curve positioning effects

        4. Action Items and Risk Mitigation:
        - Identify positions requiring attention based on T/E classification
        - Suggest potential hedge considerations for significant risks
        - Note any limit breaches or near-breach situations

        Please provide a concise analysis emphasizing:
        - Classification of T/E position (Profit/Loss/Normal)
        - Most impactful stress test scenarios
        - Key risk drivers requiring immediate attention
        - Specific recommendations based on the risk profile
        """
        return PromptTemplate(
            input_variables=["portfolio_name", "portfolio_data"],
            template=template
        )

    def analyze_portfolio(self, row: pd.Series, llm_client) -> Tuple[str, float]:
        """Analyze a single portfolio."""
        try:
            portfolio_name = row['Portfolio']
            
            # Extract key metrics
            alpha_target = float(row.get('Alpha Target', 0))
            te_value = float(row.get('Ex-Ante Tracking Error (bp/yr)', 0))
            
            # Perform risk analysis
            te_classification = RiskAnalyzer.classify_tracking_error(te_value, alpha_target)
            stress_analysis = RiskAnalyzer.analyze_stress_tests(
                row, 
                self.config.STRESS_TEST_COLUMNS
            )
            
            # Prepare portfolio data
            portfolio_data = {
                'tracking_error_classification': te_classification,
                'tracking_error_value': te_value,
                'alpha_target': alpha_target,
                'stress_analysis': stress_analysis,
                **{k: v for k, v in row.items() if pd.notna(v) and k != 'Portfolio'}
            }
            
            # Format data for prompt
            portfolio_data_str = "\n".join(
                f"{k}: {v}" for k, v in portfolio_data.items()
                if not isinstance(v, list)
            )
            portfolio_data_str += "\n\nStress Test Analysis:"
            for stress in stress_analysis:
                portfolio_data_str += f"\n{stress['scenario']}: {stress['value']} ({stress['impact']})"
            
            # Generate report
            llm_chain = LLMChain(llm=llm_client, prompt=self.prompt_template)
            with get_openai_callback() as cb:
                report = llm_chain.run(
                    portfolio_name=portfolio_name,
                    portfolio_data=portfolio_data_str
                )
                output_cost = (cb.completion_tokens / 1000) * self.config.OUTPUT_PRICE_PER_1K_TOKENS
            
            return report.strip(), output_cost
            
        except Exception as e:
            logger.error(f"Error analyzing portfolio {row.get('Portfolio', 'Unknown')}: {e}")
            return f"Error analyzing portfolio: {str(e)}", 0.0

def initialize_llm(openai_client) -> AzureChatOpenAI:
    """Initialize the Azure OpenAI client."""
    return AzureChatOpenAI(
        deployment_name=openai_client.GPT_4_OMNI_MODEL,
        openai_api_version="2023-07-01-preview",
        openai_api_key=openai_client.api_key,
        openai_api_base=openai_client.api_base,
        openai_api_type=openai_client.api_type,
        request_timeout=120,
        temperature=0.0,
    )

def main():
    """Main execution function."""
    try:
        # Initialize configuration
        config = PortfolioConfig()
        
        # Extract PDF data
        pdf_data = PDFExtractor.extract_tables(config.PDF_FILE)
        if pdf_data is None:
            raise ValueError("Failed to extract PDF data")
        
        # Process portfolio data
        portfolio_data = PortfolioProcessor.process_portfolios(config)
        
        # Merge datasets
        merged_df = pd.merge(pdf_data, portfolio_data, on='Portfolio', how='inner')
        
        # Initialize LLM client
        from quanthub.util import llm as openai_client
        llm_client = initialize_llm(openai_client)
        
        # Create analyzer
        analyzer = PortfolioAnalyzer(config)
        
        # Process all portfolios
        results = []
        total_cost = 0.0
        
        for _, row in merged_df.iterrows():
            report, cost = analyzer.analyze_portfolio(row, llm_client)
            results.append({
                'Portfolio': row['Portfolio'],
                'Report': report,
                'Cost': cost
            })
            total_cost += cost
        
        # Create results DataFrame
        results_df = pd.DataFrame(results)
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_df.to_csv(
            f"{config.OUTPUT_DIR}/portfolio_analysis_{timestamp}.csv",
            index=False
        )
        
        # Print results
        logger.info(f"Total Analysis Cost: ${total_cost:.2f}")
        for result in results:
            print(f"\nPortfolio Analysis: {result['Portfolio']}")
            print(result['Report'])
            print("-" * 50)
        
        return results_df
        
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        raise

if __name__ == "__main__":
    main()