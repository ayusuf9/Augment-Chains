import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
from dash import Dash, html, dcc
import dash_bootstrap_components as dbc

def get_data():
  #data = pd.read_csv('pages/Page3/data.csv')
  data = pd.read_csv('pages/Page3/updated_data.csv')
  return data

def get_defaults():
  data = get_data()
  market_types = data['market_type'].unique()
  default_market_type = market_types[1]
  default_securities = data[data['market_type'] == default_market_type]['security_name'].unique()[-2:]
  return default_market_type, default_securities
   
def get_page3(app):
  # Load and preprocess data
  data = pd.read_csv('pages/Page3/data.csv')
  data['Date'] = pd.to_datetime(data['month_date'])
  data['Fiscal_Date'] = pd.to_datetime(data['month_fiscal']).dt.to_period('M')
  data['security_name'] = data['security_name'].astype('category')
  data['iso_country_symbol'] = data['iso_country_symbol'].astype('category')
  data['market_type'] = data['market_type'].astype('category')
  data['sedol'] = data['sedol'].astype('category')

  data = data[data['security_name'] != '3I Group']
  data['Year'] = data['Date'].dt.year
  securities_with_data = data.groupby('security_name')['Year'].nunique().reset_index()
  securities_with_data = securities_with_data[securities_with_data['Year'] >= 7]
  securities_with_data = securities_with_data['security_name'].tolist()

  data = data[data['security_name'].isin(securities_with_data)]

  data['country_exposure_pct'] = data['country_exposure(pct)']

  market_types = data['market_type'].unique()

  default_market_type = market_types[0]
  market_types_list = list(data['market_type'].unique())

  #-------------------EXPOSURE--------------------------#
  exposure_types = data['country_exposure_name'].unique()

  default_exposure_to = ['China', 'Hong Kong']  # Set default selected countries
  exposure_types_list = list(data['country_exposure_name'].unique())

  if any(market_type == "emerging market" for market_type in market_types_list):
      default_securities = ['Infosys', 'Lenovo Group']
  else:
      default_securities = ['Apple', 'Amazon.Com']

  # default_securities = data[data['market_type'] == default_market_type]['security_name'].unique()[-2:]

  styles = {
      'container': {
          'margin': '0 auto',
          'width': '100%',      # Ensure the container takes full width
          # 'maxWidth': '1600px',  # Removed to allow expansion on wider screens
          'padding': '20px 40px',
          'fontFamily': 'Gill Sans',
      },

      'header': {
          'color': 'white',
          'padding': '30px',
          'marginBottom': '30px',
          'backgroundImage': 'url(/assets/right.jpg)',
          'backgroundBlendMode': 'darken',
          'height': '3vh',
          'backgroundSize': 'cover',
          'backgroundPosition': 'center',
          'marginTop': '-27px'
      },
      'filter_container': {
          'display': 'flex',
          'flexWrap': 'wrap',
          'gap': '10px', 
          'marginBottom': '10px',  
          'backgroundColor': '#f2f2f2',
          'padding': '10px', 
          'marginTop': '-15px',
          'height': '120px'
      },
      'filter_item': {
          'flex': '1 1 300px',
          'minWidth': '0',
          'maxWidth': '100%'
      },
      'label': {
          'marginBottom': '5px',
          'marginTop': '5px',
          'fontWeight': 'bold',
          'fontFamily': 'Gill Sans',
          'fontSize': '14px'
      },
      'label_country': {
          'marginBottom': '5px',
          'marginTop': '5px',
          'fontWeight': 'bold',
          'fontFamily': 'Gill Sans',
          'fontSize': '14px',
          'marginLeft': '-240px'
      },
      'label_two': {
          'marginBottom': '5px',
          'marginTop': '5px',
          'fontWeight': 'bold',
          'fontFamily': 'Gill Sans',
          'fontSize': '14px',
          'marginLeft': '-478px'
      },
      'dropdown': {
          # 'width': '200px',  # Original fixed width
          'width': '15%',      # Adjusted to percentage for responsiveness
          'minWidth': '150px', # Ensure a minimum width
          'maxWidth': '250px', # Optional: limit the maximum width
          'fontFamily': 'Gill Sans',
          'fontSize': '14px'
      },

      'dropdown_one': {
          # 'width': '200px',
          'width': '15%',
          'minWidth': '150px',
          'maxWidth': '250px',
          'marginLeft': '-120px',
          'fontFamily': 'Gill Sans',
          'fontSize': '14px'
      },

      'dropdown_two': {
          # 'width': '1300px',
          'width': '65%',
          'minWidth': '300px',
          'maxWidth': '1000px',
          'marginLeft': '-239px',
          'fontFamily': 'Gill Sans',
          'fontSize': '14px'
      },

      'chart_container': {
          'display': 'flex',
          'flexDirection': 'column',
          'gap': '20px',
          'marginTop': '-38px',
          'width': '100%',  # Ensure full width
      },
              
      'chart': {
          'className': 'chart-item'
      },

      'chart_two': {
          'className': 'chart-item',
          'marginTop': '-40px'
      },
      'graph': {
          'height': '60vh',      # Use viewport height for responsiveness
          'width': '100%', 
          'maxWidth': '100%',   # Ensure graph doesn't exceed container
          'margin-left': 'auto',
          'margin-right': 'auto',
          'display': 'block'
      },
      'graph_two': {
          'height': '60vh',
          'width': '100%',
          'maxWidth': '100%',
          'margin-left': 'auto',
          'margin-right': 'auto',
          'margin-top': '20px',
          'display': 'block'
      }
  }

  layout = html.Div(style=styles['container'], children=[
      html.Div(style=styles['header'], children=[
          html.H1('Revenue Exposure Tool', style={'textShadow': '2px 2px 4px rgba(0, 0, 0.9, 0.9)', 'fontFamily': 'Gill Sans', 'fontWeight': 'bold', 'fontSize': 'clamp(24px, 4vw, 50px)', 'marginTop': '-12px'}),
      ]),

      html.Div(style=styles['filter_container'], children=[
          html.Div(style={'display': 'flex', 'justifyContent': 'space-between', 'alignItems': 'center', 'marginBottom': '20px'}, children=[

              # The market type filter
              html.Div(style=styles['filter_item'], children=[
                  html.Label('Select Market Type:', style=styles['label']),
                  dcc.Dropdown(
                      id='market-type-dropdown',
                      options=[{'label': market_type.title(), 'value': market_type} for market_type in market_types],
                      value=default_market_type,
                      style=styles['dropdown'],
                      clearable=False,
                      optionHeight=50
                  )
              ]),

              # The country exposure type filter
              html.Div(
                  style=styles['filter_item'],
                  children=[
                      html.Label('Select Countries:', style=styles['label_country']),
                      dcc.Dropdown(
                          id='exposure-type-dropdown',
                          options=[
                              {'label': exposure_type.title(), 'value': exposure_type}
                              for exposure_type in exposure_types
                          ],
                          value=default_exposure_to,
                          multi=True,  # Enable multi-select functionality
                          style=styles['dropdown_one'],
                          clearable=False,
                          optionHeight=50,
                      ),
                  ],
              ),

              # The company selection filter
              html.Div(style=styles['filter_item'], children=[
                  html.Label('Select Companies:', style=styles['label_two']),
                  dcc.Dropdown(
                      id='securities-dropdown',
                      multi=True,
                      value=default_securities,# default_securities.tolist(),
                      style=styles['dropdown_two'],
                      clearable=False,
                      placeholder="You can select up to 3 companies",
                      #optionHeight=50
                  ),
                  
                  ]),
          ]),
          html.Div(style={'marginLeft': 'auto', 'marginTop': '30px'}, children=[
              html.Button('Download Data', id='download-csv-button', n_clicks=0,
                          style={
                              'fontSize': '16px',
                              'fontFamily': 'Avenir Next LT Com, sans-serif',
                              'cursor': 'pointer',
                              'backgroundColor': '#1591ea',     #'#005F9E', #87CEEB
                              'color': 'white',
                              'border': 'none',
                              'borderRadius': '5px',
                              'transition': 'background-color 0.3s',
                              'width': '150px',
                              'height': '40px'  
                              }
                          ),
              dcc.Download(id='download-csv')
          ])
      ]),

      html.Div(style={'marginTop': '20px'}, children=[
          html.Div(style=styles['chart_container'], children=[
              html.Div(style=styles['chart'], children=[
                  dcc.Graph(
                      id='country-exposure-revenue-graph',
                      config={
                          'responsive': True,
                          'autosizable': True,
                          'displayModeBar': True,
                          'displaylogo': False,
                      },
                      className='dash-graph-wide',
                      style=styles['graph']
                  )
              ]),

               # New div for text between graphs
              html.Div(style={
                      'display': 'flex',
                      'justifyContent': 'space-between',
                      #'margin': '20px 0',
                      'padding': '10px',
                      #'backgroundColor': '#f0f0f0',  # Light gray background
                      'borderRadius': '5px',
                  }, children=[
               html.Div("Source: MSCI Economic Exposure Data", style={
                          'width': '45%',
                          'marginLeft': '60px',
                          'textAlign': 'left',
                          'fontFamily': 'Gill Sans',
                          'fontSize': '17px',
                          'marginTop': '-20px',
                          'color': '#5a5a5a'
                      }),
               html.Div("SPG CSR | CMGW FRG | SPTADM Quanthub", style={
                          'width': '45%',
                          'marginRight': '60px',
                          'textAlign': 'right',
                          'fontFamily': 'Gill Sans',
                          'fontSize': '17px',
                          'marginTop': '-20px',
                          'color': '#5a5a5a'
                      }),
                  ]),
              html.Div(style=styles['chart_two'], children=[
                  dcc.Graph(
                      id='country-exposure-pct-graph',
                      config={
                          'responsive': True,
                          'autosizable': True,
                          'displayModeBar': True,
                          'displaylogo': False,
                      },
                      className='dash-graph-wide',
                      style=styles['graph_two']
                  )
              ]),

               html.Div(style={
                      'display': 'flex',
                      'justifyContent': 'space-between',
                      #'margin': '20px 0',
                      'padding': '10px',
                      #'backgroundColor': '#f0f0f0',  # Light gray background
                      'borderRadius': '5px',
                  }, children=[
               html.Div("Source: MSCI Economic Exposure Data", style={
                          'width': '45%',
                          'marginLeft': '60px',
                          'textAlign': 'left',
                          'fontFamily': 'Gill Sans',
                          'fontSize': '18px',
                          'marginTop': '-20px',
                          'color': '#5a5a5a'
                      }),
               html.Div("SPG CSR, CMGW FRG, SPTADM Quanthub", style={
                          'width': '45%',
                          'marginRight': '60px',
                          'textAlign': 'right',
                          'fontFamily': 'Gill Sans',
                          'fontSize': '18px',
                          'marginTop': '-20px',
                          'color': '#5a5a5a'
                      }),
                  ]),
          ])
      ]),
      dbc.Alert(
          id='security-alert',
          is_open=False,
          duration=4000,
          color='warninßg',
          dismissable=True,
          style={'margin-top': '10px'}
          )
        ]
      )

  return layout