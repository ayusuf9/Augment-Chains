import dash
from dash import html, dash_table, dcc
import pandas as pd
from dash.dependencies import Input, Output

# Initialize the Dash app
app = dash.Dash(__name__)

# Assume we have a CSV file. Replace 'your_data.csv' with your actual file path
# df = pd.read_csv('your_data.csv')

# For demonstration, let's create sample data
df = pd.DataFrame({
    'Name': ['John Doe', 'Jane Smith', 'Bob Johnson', 'Alice Brown'],
    'Age': [28, 34, 42, 31],
    'City': ['New York', 'Los Angeles', 'Chicago', 'Houston'],
    'Salary': [75000, 82000, 68000, 91000]
})

# Define the layout
app.layout = html.Div([
    html.H1('Interactive Data Table Dashboard',
            style={'textAlign': 'center', 'color': '#2c3e50', 'margin': '20px'}),
    
    html.Div([
        dash_table.DataTable(
            id='datatable',
            columns=[{'name': col, 'id': col} for col in df.columns],
            data=df.to_dict('records'),
            
            # Styling
            style_table={
                'overflowX': 'auto',
                'maxWidth': '1200px',
                'margin': '0 auto'
            },
            style_header={
                'backgroundColor': '#2c3e50',
                'color': 'white',
                'fontWeight': 'bold',
                'textAlign': 'center',
                'padding': '12px'
            },
            style_cell={
                'padding': '12px',
                'textAlign': 'left',
                'backgroundColor': 'white',
                'color': '#2c3e50',
                'fontSize': '14px',
                'fontFamily': 'Arial'
            },
            style_data_conditional=[
                {
                    'if': {'row_index': 'odd'},
                    'backgroundColor': '#f9f9f9'
                },
                {
                    'if': {'state': 'selected'},
                    'backgroundColor': '#e6f3ff',
                    'border': '1px solid #3498db'
                }
            ],
            
            # Interactive features
            sort_action='native',
            filter_action='native',
            page_action='native',
            page_size=10,
            
            # Enable row selection
            row_selectable='multi',
            selected_rows=[],
        )
    ], style={'margin': '20px'}),
    
    # Display selection information
    html.Div(id='selection-info', 
             style={'margin': '20px', 'textAlign': 'center', 'color': '#2c3e50'})
])

# Callback to update selection info
@app.callback(
    Output('selection-info', 'children'),
    Input('datatable', 'selected_rows')
)
def update_selection(selected_rows):
    if not selected_rows:
        return 'No rows selected'
    return f'Selected rows: {len(selected_rows)}'

if __name__ == '__main__':
    app.run_server(debug=True)