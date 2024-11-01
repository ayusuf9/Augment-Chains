# Cell 1: Import Libraries and Setup
import pandas as pd
import tabula.io
import warnings
from typing import List, Dict, Tuple
from langchain.chat_models import AzureChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.callbacks import get_openai_callback
from quanthub.util import llm as openai_client
warnings.filterwarnings("ignore")

# Cell 2: Configuration
class Config:
    OUTPUT_PRICE_PER_1K_TOKENS = 0.015
    PDF_FILE = 'analytics_report_20240827.pdf'
    CSV_TEMPLATE = 'Risk Contributions/var&asofdate=20240830&portfolio={}&var_type=AVaR_WKY'
    PORTFOLIOS = [
        "BFA", "CGCB", "AMBAL-P-FI", "VIAA-P-IG", "CIB-P", "IFA-P-IG",
        "SBF", "CGCP", "STBF", "IBFA", "CGSD", "CGUI", "GVT", "AFMF", "ILBF"
    ]
    STRESS_TEST_COLUMNS = [
        'Active Stress P&L-SP500 Down 20',
        'Active Stress P&L-Inflation Overshoot'
    ]

# Cell 3: Risk Analysis Functions
def classify_tracking_error(te_value: float, alpha_target: float) -> str:
    """Classify tracking error based on alpha target threshold."""
    if te_value > 2 * alpha_target:
        return "Profit"
    elif te_value < -2 * alpha_target:
        return "Loss"
    return "Normal"

def analyze_stress_tests(row: pd.Series, stress_columns: List[str]) -> List[Dict]:
    """Analyze stress test results."""
    stress_analysis = []
    for col in stress_columns:
        if col in row:
            value = float(row[col])
            stress_analysis.append({
                'scenario': col,
                'value': value,
                'impact': 'Positive' if value > 0 else 'Negative' if value < 0 else 'Neutral',
                'magnitude': abs(value)
            })
    return stress_analysis

# Cell 4: Data Processing Functions
def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and standardize column names."""
    replacements = {
        'USD Rate Derivati ves': 'USD Rate Derivatives',
        'Non- USD Rate Derivati ves': 'Non-USD Rate Derivatives',
        'Structu red Credit': 'Structured Credit',
        '15Y- 25Y': '15Y-25Y',
        'Active Stress P&L - SP500 Down 20': 'Active Stress P&L-SP500 Down 20',
        'Ex- Ante Tracki ng Error (bp/yr)': 'Ex-Ante Tracking Error (bp/yr)',
        'Active Stress P&L - Inflation Over- shoot': 'Active Stress P&L-Inflation Overshoot'
    }
    
    df = df.copy()
    for old, new in replacements.items():
        df.columns = df.columns.str.replace(old, new, regex=True)
    df.columns = df.columns.str.replace(r'\r', ' ', regex=True)
    df.columns = df.columns.str.replace(r'\s+', ' ', regex=True)
    df.columns = df.columns.str.strip()
    
    return df

def extract_pdf_data(pdf_file: str) -> pd.DataFrame:
    """Extract data from PDF report."""
    tables = tabula.io.read_pdf(pdf_file, pages='1', lattice=True)
    tdf = tables[0]
    tdf = tdf.dropna(axis=1, how='all')
    tdf.columns = tdf.iloc[0]
    tdf = tdf.drop(0).reset_index(drop=True)
    return clean_column_names(tdf)

def process_portfolio_csv(portfolio: str, csv_template: str) -> pd.DataFrame:
    """Process individual portfolio CSV file."""
    file_path = csv_template.format(portfolio)
    df = pd.read_csv(file_path)
    df = df[df['Level'] == '1'][['Fund', 'Filter Level 1', 'Risk Contribution']]
    pivot_df = df.pivot(index='Fund', columns='Filter Level 1', values='Risk Contribution')
    pivot_df = pivot_df.reset_index(drop=True)
    pivot_df = pivot_df.add_prefix('Tracking Error (T/E) Contribution - ')
    pivot_df['Portfolio'] = portfolio
    return pivot_df

# Cell 5: Analysis Template
ANALYSIS_TEMPLATE = """
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

# Cell 6: Portfolio Analysis Function
def analyze_portfolio(row: pd.Series, llm_client, config: Config) -> Tuple[str, float]:
    """Generate analysis report for a single portfolio."""
    portfolio_name = row['Portfolio']
    
    # Extract key metrics
    alpha_target = float(row.get('Alpha Target', 0))
    te_value = float(row.get('Ex-Ante Tracking Error (bp/yr)', 0))
    
    # Perform risk analysis
    te_classification = classify_tracking_error(te_value, alpha_target)
    stress_analysis = analyze_stress_tests(row, config.STRESS_TEST_COLUMNS)
    
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
    
    # Initialize LLM chain
    prompt = PromptTemplate(
        input_variables=["portfolio_name", "portfolio_data"],
        template=ANALYSIS_TEMPLATE
    )
    llm_chain = LLMChain(llm=llm_client, prompt=prompt)
    
    # Generate report
    with get_openai_callback() as cb:
        report = llm_chain.run(
            portfolio_name=portfolio_name,
            portfolio_data=portfolio_data_str
        )
        output_cost = (cb.completion_tokens / 1000) * config.OUTPUT_PRICE_PER_1K_TOKENS
    
    return report.strip(), output_cost

# Cell 7: Main Execution
# Initialize configuration
config = Config()

# Extract PDF data
print("Extracting PDF data...")
pdf_data = extract_pdf_data(config.PDF_FILE)

# Process portfolio data
print("Processing portfolio data...")
portfolio_dfs = []
for portfolio in config.PORTFOLIOS:
    try:
        df = process_portfolio_csv(portfolio, config.CSV_TEMPLATE)
        portfolio_dfs.append(df)
    except Exception as e:
        print(f"Error processing portfolio {portfolio}: {e}")
        continue

portfolio_data = pd.concat(portfolio_dfs, ignore_index=True)

# Merge datasets
merged_df = pd.merge(pdf_data, portfolio_data, on='Portfolio', how='inner')

# Initialize LLM client
llm_client = AzureChatOpenAI(
    deployment_name=openai_client.GPT_4_OMNI_MODEL,
    openai_api_version="2023-07-01-preview",
    openai_api_key=openai_client.api_key,
    openai_api_base=openai_client.api_base,
    openai_api_type=openai_client.api_type,
    request_timeout=120,
    temperature=0.0,
)

# Process all portfolios
results = []
total_cost = 0.0

for _, row in merged_df.iterrows():
    print(f"\nAnalyzing portfolio: {row['Portfolio']}")
    report, cost = analyze_portfolio(row, llm_client, config)
    results.append({
        'Portfolio': row['Portfolio'],
        'Report': report,
        'Cost': cost
    })
    total_cost += cost
    print(report)
    print("-" * 50)

print(f"\nTotal Analysis Cost: ${total_cost:.2f}")

# Create results DataFrame
results_df = pd.DataFrame(results)
results_df.to_csv("portfolio_analysis_results.csv", index=False)
print("\nResults saved to portfolio_analysis_results.csv")