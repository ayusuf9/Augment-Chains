import dash
from dash import dash_table, html
import pandas as pd

# Initialize the Dash app
app = dash.Dash(__name__)

# Assume you have a CSV file - replace 'your_file.csv' with your actual file path
# df = pd.read_csv('your_file.csv')

# For demonstration, let's create sample data
df = pd.DataFrame({
    'Name': ['John Doe', 'Jane Smith', 'Bob Johnson'],
    'Age': [28, 34, 42],
    'City': ['New York', 'San Francisco', 'Chicago'],
    'Salary': ['$75,000', '$95,000', '$82,000']
})

# Define table styles
table_styles = {
    'cell': {
        'backgroundColor': 'white',
        'color': 'black',
        'fontSize': '14px',
        'font-family': 'Arial, sans-serif',
        'padding': '10px',
        'textAlign': 'left'
    },
    'header': {
        'backgroundColor': '#075985',  # Dark blue header
        'color': 'white',
        'fontSize': '16px',
        'font-family': 'Arial, sans-serif',
        'fontWeight': 'bold',
        'padding': '12px',
        'textAlign': 'left'
    },
    'data': {
        'whiteSpace': 'normal',
        'height': 'auto',
    }
}

# Define the layout
app.layout = html.Div([
    html.H1(
        'Data Table Dashboard',
        style={
            'textAlign': 'center',
            'color': '#075985',
            'fontSize': '32px',
            'font-family': 'Arial, sans-serif',
            'margin': '20px 0'
        }
    ),
    
    html.Div([
        dash_table.DataTable(
            id='table',
            columns=[{"name": i, "id": i} for i in df.columns],
            data=df.to_dict('records'),
            
            # Styling
            style_cell=table_styles['cell'],
            style_header=table_styles['header'],
            style_data=table_styles['data'],
            
            # Table configuration
            page_size=10,  # Number of rows per page
            style_table={
                'overflowX': 'auto',  # Horizontal scroll if needed
                'borderRadius': '8px',
                'boxShadow': '0 4px 6px -1px rgb(0 0 0 / 0.1)',
                'margin': '20px 0'
            },
            
            # Combined conditional styling for striped rows and hover
            style_data_conditional=[
                {
                    'if': {'row_index': 'odd'},
                    'backgroundColor': '#f8fafc'  # Light gray for odd rows
                },
                {
                    'if': {'state': 'hover'},
                    'backgroundColor': '#e2e8f0',
                    'cursor': 'pointer'
                }
            ],
            
            # Enable sorting
            sort_action='native',
            
            # Enable filtering
            filter_action='native',
            
            # Enable row selection
            row_selectable='multi',
            
            # Enable cell editing
            editable=True,
        )
    ], style={
        'padding': '20px',
        'maxWidth': '1200px',
        'margin': '0 auto'
    })
])

if __name__ == '__main__':
    app.run_server(debug=True)