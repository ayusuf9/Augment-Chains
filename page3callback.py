from dash.dependencies import Input, Output
import plotly.graph_objects as go
import pandas as pd
from dash import no_update
from .Page3Dash import *
import datetime
from dash import Input, Output, State, no_update

data = pd.read_csv('pages/Page3/updated_data.csv')
data['Date'] = pd.to_datetime(data['month_date'])
#data['Fiscal_Date'] = pd.to_datetime(data['month_fiscal']).dt.to_period('M')
data['security_name'] = data['security_name'].astype('category')
data['iso_country_symbol'] = data['iso_country_symbol'].astype('category')
data['market_type'] = data['market_type'].astype('category')
data['sedol'] = data['sedol'].astype('category')

data['security_name'] = data['security_name'].astype(str)
data['sedol'] = data['sedol'].astype(str)

data['security_name'] = data['security_name'] + ' (' + data['sedol'] + ')'

# Filter securities with data from 2015 to 2023
data = data[data['security_name'] != '3I Group (B1YW440)']
data['Year'] = data['Date'].dt.year
#securities_with_data = data.groupby('security_name')['Year'].nunique().reset_index()
#securities_with_data = securities_with_data[securities_with_data['Year'] >= 8]
securities_with_data = data['security_name'].tolist()

# Filter country_exposure_data to only include securities with data from 2015 to 2023
data = data[data['security_name'].isin(securities_with_data)]

data['country_exposure_pct'] = data['country_exposure(pct)']

# Create a list of unique market types
market_types = data['market_type'].unique()

# Select default market type and securities
default_market_type = market_types[0]

market_types_list = list(data['market_type'].unique())
print(market_types_list)

#-------------------EXPOSURE--------------------------#
exposure_types_list = list(data['country_exposure_name'].unique())
exposure_types = data['country_exposure_name'].unique()

default_exposure_to = exposure_types[0]


if any(market_type.lower() == "developed market" for market_type in market_types_list):
        default_securities = ['Apple (2046251)', 'Amazon.Com (2000019)']
elif any(market_type.lower() == "emerging market" for market_type in market_types_list):
    default_securities = ['Infosys (6205122)', 'Lenovo Group (6218089)']

def register_page3_callbacks(app):
  @app.callback(
    Output('exposure-type-dropdown', 'options'),
    Output('exposure-type-dropdown', 'value'),
    Input('market-type-dropdown', 'value'),
  )
  def update_country_dropdown(market_type):
    countries = data[data['market_type'] == market_type]['country_exposure_name'].unique()
    options = [{'label': country.title(), 'value': country} for country in countries]
    # Ensure default countries are in the available options
    default_countries = ['China', 'Hong Kong']
    default_value = [country for country in default_countries if country in countries]
    if not default_value:
        default_value = countries[:2].tolist() if len(countries) >= 2 else countries.tolist()
    return options, default_value

  @app.callback(
    Output('securities-dropdown', 'options'),
    Output('securities-dropdown', 'value'),
    Input('market-type-dropdown', 'value'),
    Input('exposure-type-dropdown', 'value'),
  )
  def update_securities_dropdown(market_type, countries):
    if not isinstance(countries, list):
        countries = [countries]
    securities = data[
        (data['market_type'] == market_type)
        & (data['country_exposure_name'].isin(countries))
    ]['security_name'].unique()
    options = [{'label': security, 'value': security} for security in securities]

    default_value = [s for s in default_securities if s in securities]
    if not default_value:
        default_value = securities[-2:].tolist() if len(securities) >= 2 else securities.tolist()

    return options, default_value

  @app.callback(
    [Output('country-exposure-pct-graph', 'figure'),
     Output('country-exposure-revenue-graph', 'figure'),
     Output('security-alert', 'is_open'),
     Output('security-alert', 'children')],
     [Input('market-type-dropdown', 'value'),
      Input('exposure-type-dropdown', 'value'),
      Input('securities-dropdown', 'value')]
  )
  def update_graphs(market_type, countries, securities):
      if securities is None or len(securities) == 0:
          return no_update, no_update, True, "At least one security option is required."
      elif len(securities) > 4:
          return no_update, no_update, True, "Maximum 4 securities can be selected."

      if not isinstance(countries, list):
          countries = [countries]

      filtered_data = data[
          (data['market_type'] == market_type)
          & (data['country_exposure_name'].isin(countries))
          & (data['security_name'].isin(securities))
      ]

      # Aggregate over selected countries
      filtered_data = (
          filtered_data.groupby(['security_name', 'Date'])
          .agg(
              {
                  'country_exposure_pct': 'sum',
                  'country_exposure_revenue': 'sum',
                  'isd_currency_symbol': 'first',  # Assuming currency symbol is the same per security
              }
          )
          .reset_index()
      )

      fig_pct = go.Figure()
      fig_revenue = go.Figure()

      cg_color_pallet = {
          'sapphire': '#005F9E',
          'dark_sapphire': '#00294B',
          'ocean': '#009CDC',
          'light_ocean': '#7BD0E2',
          'turquoise': '#00AEA9',
          'dark_turquoise_1': '#008E77',
          'dark_turquoise_2': '#004C46',
          'raspberry': '#B42573',
          'dark_raspberry_1': '#762157',
          'dark_raspberry_2': '#532a45',
          'neutral_7': '#554742',
          'neutral_4': '#A39E99',
          'neutral_2': '#D5D0CA',
          'cg_recession': '#f3f3f3'
      }

      cg_font_families = {
          'default': 'Avenir Next LT Com',
          'alternative': "Arial"
      }

      color_list = [
                    cg_color_pallet['sapphire'],
                    cg_color_pallet['turquoise'],
                    cg_color_pallet['neutral_4'],
                    cg_color_pallet['raspberry'],
                     cg_color_pallet['dark_sapphire'],
                    cg_color_pallet['light_ocean'],   
                  ]
      
      overall_pct_min = float('inf')
      overall_pct_max = float('-inf')
      overall_revenue_min = float('inf')
      overall_revenue_max = float('-inf')

      def format_revenue(value):
          if abs(value) >= 1e12:
              return f"{value/1e12:.1f}T"
          elif abs(value) >= 1e9:
              return f"{value/1e9:.1f}B"
          elif abs(value) >= 1e6:
              return f"{value/1e6:.1f}M"
          else:
              return f"{value:.0f}"


      for i, security in enumerate(securities):

          security_data = filtered_data[filtered_data['security_name'] == security]

          overall_pct_min = min(overall_pct_min, security_data['country_exposure_pct'].min())
          overall_pct_max = max(overall_pct_max, security_data['country_exposure_pct'].max())
          
          overall_revenue_min = min(overall_revenue_min, security_data['country_exposure_revenue'].min())
          overall_revenue_max = max(overall_revenue_max, security_data['country_exposure_revenue'].max())
          
          fig_pct.add_trace(go.Scatter(
              x=security_data['Date'],
              y=security_data['country_exposure_pct'],
              name=f"<b>{security}</b>",
              line=dict(shape='spline', width=4.5, color=color_list[i % len(color_list)]),
              mode='lines',
              showlegend=True,
              marker=dict(symbol=i),
              yaxis='y2',
              hovertemplate=(
                  "<b>%{fullData.name}</b><br>"
                  "Date: %{x|%Y-%m-%d}<br>"
                  "Exposure: <b>%{y:.2f}%</b><br>"
                  "<extra></extra>"
              )
          ))
          
          fig_revenue.add_trace(go.Scatter(
              x=security_data['Date'],
              y=security_data['country_exposure_revenue'],
              name=f"<b>{security}</b>",
              line=dict(shape='spline', width=4.5, color=color_list[i % len(color_list)]),
              mode='lines',
              showlegend=True,
              marker=dict(symbol=i),
              yaxis='y2',
              customdata=security_data['isd_currency_symbol'],
              hovertemplate=(
                  "<b>%{fullData.name}</b><br>"
                  "Date: %{x|%Y-%m-%d}<br>"
                  "Revenue: <b>%{y:.2f}</b> %{customdata}<br>"
                  "<extra></extra>"
              ),
          ))

          max_date = security_data['Date'].max()
          start_date = max_date - pd.DateOffset(years=5)


          layout_updates = dict(
          hoverlabel=dict(
              bgcolor="white",
              font_size=12,
              font_family='Gill Sans',
          ),
          hovermode="closest"
          )

          fig_pct.update_layout(**layout_updates)
          fig_revenue.update_layout(**layout_updates)

      pct_range = overall_pct_max - overall_pct_min
      revenue_range = overall_revenue_max - overall_revenue_min

      fig_pct.add_hline(y=overall_pct_max + 0.1 * pct_range, line=dict(color='gray', width=4), opacity=0.7)
      fig_pct.add_hline(y=overall_pct_min - 0.1 * pct_range, line=dict(color='gray', width=4), opacity=0.7)

      fig_revenue.add_hline(y=overall_revenue_max + 0.1 * revenue_range, line=dict(color='gray', width=4), opacity=0.7)
      fig_revenue.add_hline(y=-0.1, line=dict(color='gray', width=4), opacity=0.7)

      font_family = cg_font_families['default']
      font = {'family':font_family, 'size':14, 'color':'grey'}

      fig_pct.update_layout(
          modebar_remove=['zoom', 'pan', 'autoscale', 'resetScale2d', 'zoomIn2d', 'zoomOut2d'],
          title_font=font,
          xaxis_title_font=font,
          yaxis_title_font=font,
          title=dict(
              text="Exposure (Percentage)",
              font=dict(
                  size=21,
                  color="#00294b",
                  weight='bold'),
              x=0.04
          ),


          annotations=[
              dict(
                  text="Data as of: 2024-07-31",  
                  font=dict(
                      size=15,
                      color="#5a5a5a",
                      weight='bold'
                  ),
                  x=1, 
                  y=1.12, 
                  xref="paper",
                  yref="paper",
                  showarrow=False,
                  xanchor="right",
                  yanchor="bottom"
              )],
          xaxis_title="Years",
          hovermode="x unified",
          template="plotly_white",
          autosize=True,
          yaxis=dict(
              title=dict(
                  text="Exposure",
                  font=dict(
                      family='Gill Sans',
                      size=15,               
                      color="#00003f",
                      weight='bold',
                  )
              ),
              showline=False,
              zeroline=False,
              showgrid=False,
              ticks="outside",
              ticklen=5,
              tickformat=".0f",
              tickprefix='',
              ticksuffix='%',
              tickfont=dict(
                  family='Gill Sans',
                  size=15,
                  color='#00003f',
                  weight='bold'
              )
          ),
          yaxis2=dict(
              title=dict(
                  text="Exposure",
                  font=dict(
                      family='Gill Sans',
                      size=15,               
                      color="#00003f",
                      weight='bold',
                  )
              ),
              overlaying='y',
              side='right',
              showticklabels=True,
              ticks="outside",
              tickformat=".0f",
              tickprefix='',
              ticksuffix='%',
              showgrid=True,
              gridwidth=0.2,
              matches='y',
              gridcolor='lightgray',
              tickfont=dict(
                  family='Gill Sans',
                  size=15,
                  color='#00003f',
                  weight='bold'
              )
          ),
          )

      fig_revenue.update_layout(
          modebar_remove=['zoom', 'pan', 'autoscale', 'resetScale2d', 'zoomIn2d', 'zoomOut2d'],
          title_font=font,
          xaxis_title_font=font,
          yaxis_title_font=font,
          title=dict(
              text=f"{country} Revenue Exposure",
              font=dict(
                  size=24,
                  color="#00294b",
                  weight='bold'),
              x=0.04
          ),

          annotations=[
              dict(
                  text="Data as of: 2024-07-31", 
                  font=dict(
                      size=15,
                      color="#5a5a5a",
                      weight='bold'
                  ),
                  x=1,  
                  y=1.12, 
                  xref="paper",
                  yref="paper",
                  showarrow=False,
                  xanchor="right",
                  yanchor="bottom"
              )
          ],
          xaxis_title="",
          hovermode="x unified",
          template="plotly_white",
          autosize=True,
          yaxis=dict(
              title=dict(
                  text="Revenue",
                  font=dict(
                      family='Gill Sans',
                      size=15,               
                      color="#00003f",
                      weight='bold',
                  )
              ),
              showline=False,
              zeroline=False,
              showgrid=False,
              ticks="outside",
              ticklen=5,
              tickfont=dict(
                  family='Gill Sans',
                  size=15,
                  color='#00003f',
                  weight='bold'
              ),
              rangemode='tozero'
          ),
          yaxis2=dict(
              title=dict(
                  text="Revenue",
                  font=dict(
                      family='Gill Sans',
                      size=15,               
                      color="#00003f",
                      weight='bold',
                  )
              ),
              ticklen=5,
              ticks="outside",
              overlaying='y',
              side='right',
              showticklabels=True,
              showgrid=True,
              gridwidth=0.2,
              gridcolor='lightgray',
              matches='y',
              tickfont=dict(
                  family='Gill Sans',
                  size=15,
                  color='#00003f',
                  weight='bold'
              ),
              rangemode='tozero'
          ),

          )


      for fig in [fig_pct, fig_revenue]:
          fig.update_layout(
              xaxis=dict(
                  tickfont=dict(
                      family='Gill Sans',
                      size=15,
                      color='#00003f', 
                      weight='bold'
                  ),
                  showgrid=False,
                  ticks="outside",
          ),
          legend=dict(
                  orientation="h", 
                  yanchor="bottom", 
                  y=1.02,
                  xanchor="right", 
                  x=1, 
                  font=dict(
                      family='Gill Sans',
                      size=11,
                      color="#5a5a5a"
                  ),
                  itemsizing='constant',
                  itemwidth=30
              ),
              margin=dict(t=120, r=20, l=20, b=50)
          )

          def get_max_date(fig):
              max_date = max(max(trace.x) for trace in fig.data if len(trace.x) > 0)
              return max_date

          def get_date_range(fig, days):
              x_data = fig.data[0].x
              end_date = get_max_date(fig)
              start_date = max(x_data[0], end_date - pd.Timedelta(days=days))
              return [start_date, end_date]
          
          end_date = datetime.datetime.now()
          start_date = end_date - datetime.timedelta(days=3*365)

          for fig in [fig_pct, fig_revenue]:

              max_date = get_max_date(fig)
              initial_range = [max_date - datetime.timedelta(days=365*3), max_date]

              fig.update_layout(
                  xaxis=dict(
                      range=initial_range 
                  ),
                  updatemenus=[
                      dict(
                          type="buttons",
                          showactive=True,
                          active=0,
                          buttons=[
                              dict(
                                  label="3Y",
                                  method="relayout",
                                  args=[{"xaxis.range": get_date_range(fig, 365*3)}],
                      
                              ),
                              dict(
                                  label="5Y",
                                  method="relayout",
                                  args=[{"xaxis.range": get_date_range(fig, 365*5)}],
                             
                              ),
                              dict(
                                  label="All",
                                  method="relayout",
                                  args=[{"xaxis.autorange": True}],
                         
                              ),
                          ],
                          direction="right",
                          pad={"r": 10, "t": 10, "b": 10, "l": 10},
                          x=0,
                          xanchor="left",
                          y=1.14,
                          yanchor="top",
                          bgcolor="white",
                          bordercolor="rgba(100, 149, 237, 0.7)",   
                          borderwidth=1.3,      
                          font=dict(family="Gill Sans", size=12, color="black"),  #'fontFamily': 'Gill Sans'
                      )
                  ]
              )


      return fig_pct, fig_revenue, False, ""
  

  @app.callback(
      Output('download-csv', 'data'),
      Input('download-csv-button', 'n_clicks'),
      State('market-type-dropdown', 'value'),
      State('securities-dropdown', 'value'),
      prevent_initial_call=True
  )
  def download_csv(n_clicks, market_type, securities):
      if n_clicks is None or n_clicks == 0:
          return no_update
      filtered_data = data[(data['market_type'] == market_type) & (data['security_name'].isin(securities))]
      csv_data = filtered_data.to_csv(index=False)
      return {
          'content': csv_data,
          'filename': 'china_exposure.csv',
          'type': 'text/csv'
      }