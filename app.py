import os
from dotenv import load_dotenv
import datetime
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import requests
import yfinance as yf
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate
from flask import Flask
from flask_cors import CORS

load_dotenv()

AA_API_KEY = os.environ.get('AA_API_KEY')
FRED_API_KEY = os.environ.get('FRED_API_KEY')


def fetch_sp500_data():
    sp500_ticker = '^GSPC'
    sp500 = yf.Ticker(sp500_ticker)
    df = sp500.history(period='max')
    df = df.reset_index()[['Date', 'Close']].set_index('Date')
    df.index = pd.to_datetime(df.index).tz_localize(None)

    df.columns = ['5. adjusted close']
    return df


class AlphaVantageLimitReachedException(Exception):
    pass

def fetch_stock_data(ticker):
    base_url = 'https://www.alphavantage.co/query?'
    function = 'TIME_SERIES_DAILY_ADJUSTED'
    datatype = 'json'

    url = f"{base_url}function={function}&symbol={ticker}&apikey={AA_API_KEY}&datatype={datatype}"
    try:
        response = requests.get(url)
        response.raise_for_status()
    except requests.exceptions.HTTPError as err:
        if response.status_code == 403 and 'api limit reached' in response.text.lower():
            raise AlphaVantageLimitReachedException('API request limit reached. Please try again in a minute.')
        else:
            raise err

    data = response.json()

    if 'Error Message' in data or 'Note' in data:
        return None

    df = pd.DataFrame(data['Time Series (Daily)']).T
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df = df.sort_index(ascending=True)
    df = df.astype(float)

    return df



def fetch_company_overview(ticker):
    base_url = 'https://www.alphavantage.co/query?'
    function = 'OVERVIEW'

    url = f"{base_url}function={function}&symbol={ticker}&apikey={AA_API_KEY}"
    response = requests.get(url)
    data = response.json()

    return data


def fetch_earnings_data(ticker):
    base_url = 'https://www.alphavantage.co/query?'
    function = 'EARNINGS'

    url = f"{base_url}function={function}&symbol={ticker}&apikey={AA_API_KEY}"
    print(url)
    response = requests.get(url)
    data = response.json()

    return data['quarterlyEarnings']


def fetch_risk_free_rate(fred_api_key, start_date, end_date):
    base_url = "https://api.stlouisfed.org/fred/series/observations"
    series_id = "GS10"  # 10-year US Treasury Yield
    file_type = "json"

    url = f"{base_url}?series_id={series_id}&api_key={fred_api_key}&file_type={file_type}&sort_order=asc&start_date={start_date}&end_date={end_date}"
    response = requests.get(url)
    data = response.json()

    df = pd.DataFrame(data['observations'])
    df['date'] = pd.to_datetime(df['date'])
    df['value'] = df['value'].astype(float)
    df.set_index('date', inplace=True)
    df = df[df['value'] != -1]

    return df


def calculate_moving_averages(df, window):
    return df['5. adjusted close'].rolling(window=window).mean()


def normalize_data(df, column):
    return (df[column] - df[column].min()) / (df[column].max() - df[column].min())
# Create a Flask app and wrap the Dash app with it
server = Flask(__name__)
CORS(server)
app = dash.Dash(__name__, server=server, external_stylesheets=[dbc.themes.BOOTSTRAP]) # type: ignore

# Update the app.layout to be a function
def serve_layout():
    return dbc.Container([
    dbc.Row([
        dbc.Col(html.H1('Stock Market Dashboard', className='text-center mb-4'), width=12),
        dbc.Col(html.Sub('API request limit: 5 per min - Refresh if hangs', className='text-center mb-4'), width=12)

    ]),
    dbc.Row([
        dbc.Col(
            dbc.InputGroup([
                dbc.Input(id='stock-ticker', type='text', value='MSFT', placeholder='Enter a stock ticker'),
                dbc.Button('Submit', id='submit-button', color='primary', className='ms-2')
            ], className='mb-4'),
            width=12
        )
    ]),
    dbc.Row([
        dbc.Col(
            dcc.Loading(
                id="stock-price-loading",
                type="circle",
                children=[dcc.Graph(id='stock-price-graph')],
            ),
            width=12,
        )
    ]),
    dbc.Row([
        dbc.Col(
            dcc.Loading(
                id="company-overview-loading",
                type="circle",
                children=[html.Div(id='company-overview')],
            ),
            width=12,
        )
    ]),
    # Add the following row to display the risk-free rate chart at the bottom
    dbc.Row([
        dbc.Col(
            dcc.Graph(id='risk-free-rate-graph', style={'height': '400px'})
        ),
    ]),
   ], fluid=True)


app.layout = serve_layout


@app.callback(
    [Output('stock-price-graph', 'figure'),
     Output('company-overview', 'children'),
     Output('risk-free-rate-graph', 'figure'),
     Output('submit-button', 'disabled')],
    [Input('submit-button', 'n_clicks')],
    [dash.dependencies.State('stock-ticker', 'value')],
    prevent_initial_call=True)

def update_graph_and_financials(n_clicks, stock_ticker):
    # Disable the submit button
    disabled = True

    if n_clicks == 0:
        return go.Figure(), None, go.Figure(), disabled

    try:
        # Fetch stock data
        df = fetch_stock_data(stock_ticker)
        if df is None:
            return go.Figure(), None, go.Figure(), disabled
    except AlphaVantageLimitReachedException as e:
        return go.Figure(), html.P(str(e)), go.Figure(), disabled
    except Exception as e:
        # Handle other exceptions here
        return go.Figure(), None, go.Figure(), disabled

    # Fetch S&P 500 data
    sp500_df = fetch_sp500_data()

    # Calculate stock returns and S&P 500 returns
    df['returns'] = df['5. adjusted close'].pct_change()
    sp500_df['returns'] = sp500_df['5. adjusted close'].pct_change()

    # Calculate stock beta
    covariance = df['returns'].cov(sp500_df['returns'])
    sp500_variance = sp500_df['returns'].var()
    beta = covariance / sp500_variance # type: ignore
    # Get the company overview, earnings data, and risk-free rate
    company_overview = fetch_company_overview(stock_ticker)
    earnings_data = fetch_earnings_data(stock_ticker)
    start_date = '2019-11-01'
    end_date = df.index.max().strftime("%Y-%m-%d")
    risk_free_rate_df = fetch_risk_free_rate(
        FRED_API_KEY, '1980-01-01', '2023-02-28')

    # Calculate the moving averages and plot the stock
    df['SMA10'] = calculate_moving_averages(df, 10)
    df['SMA50'] = calculate_moving_averages(df, 50)

    # Calculate daily risk-free rate
    daily_risk_free_rate = (
        1 + risk_free_rate_df['value'].mean() / 100) ** (1 / 252) - 1

    # Calculate CAPM daily expected returns
    df['capm_expected_returns'] = daily_risk_free_rate + \
        beta * (sp500_df['returns'] - daily_risk_free_rate)

    trace_close = go.Scatter(
        x=df.loc[start_date:end_date].index,
        y=df.loc[start_date:end_date, '5. adjusted close'],
        mode='lines',
        name='Close',
        yaxis='y1'
    )
    trace_sma10 = go.Scatter(
        x=df.loc[start_date:end_date].index,
        y=df.loc[start_date:end_date, 'SMA10'],
        mode='lines',
        name='SMA10',
        yaxis='y1'
    )
    trace_sma50 = go.Scatter(
        x=df.loc[start_date:end_date].index,
        y=df.loc[start_date:end_date, 'SMA50'],
        mode='lines',
        name='SMA50',
        yaxis='y1'
    )

    # Find earnings dates
    earnings_dates = [pd.to_datetime(earning['fiscalDateEnding'])
                    for earning in earnings_data]
    earnings_dates = [date for date in earnings_dates if date in df.index]
    latest_earning_date = pd.to_datetime(earnings_data[0]['fiscalDateEnding'])

    earnings_trace = go.Scatter(
        x=[latest_earning_date, latest_earning_date],
        y=[df['5. adjusted close'].min(), df['5. adjusted close'].max()],
        mode='lines',
        name='Earnings',
        line=dict(color='red', width=2, dash='dash'),
        yaxis='y1'
    )
    

    trace_capm_expected_returns = go.Scatter(
        x=df.loc[start_date:end_date].index,
        y=df.loc[start_date:end_date, 'capm_expected_returns'],
        mode='lines',
        name='CAPM Expected Returns',
        yaxis='y2'
    )

    layout = go.Layout(
        title=f'{stock_ticker} Stock Price and CAPM Expected Returns',
        yaxis=dict(
            title='Price',
            side='left',
            showgrid=False,
            zeroline=False
        ),
        yaxis2=dict(
            title='CAPM Expected Returns',
            side='right',
            showgrid=False,
            zeroline=False,
            overlaying='y'
        ),
        legend=dict(
            x=0.01,
            y=0.99,
            bgcolor='rgba(255, 255, 255, 0.5)',
            bordercolor='rgba(0, 0, 0, 0.5)'
        ),
        hovermode="x unified",
        shapes=[
                # Vertical line for each earnings date
                dict(
                    type='line',
                    x0=date,
                    x1=date,
                    y0=0,
                    y1=1,
                    xref='x',
                    yref='paper',
                    line=dict(
                        color='red',
                        width=1,
                        dash='dot'
                    )
                ) for date in earnings_dates
            ]
        )


    # Display company overview
    company_overview_elements = [
        html.H3('Company Overview:'),
        dbc.Row([
            dbc.Col(html.P(f"Name: {company_overview['Name']}"), width=6),
            dbc.Col(html.P(f"Industry: {company_overview['Industry']}"), width=6),
        ]),
        dbc.Row([
            dbc.Col(html.P(f"Sector: {company_overview['Sector']}"), width=6),
            dbc.Col(html.P(f"Latest Financial Report Date: {company_overview['LatestQuarter']}"), width=6),
        ]),
        dbc.Row([
            dbc.Col(html.P(f"Revenue (TTM): ${float(company_overview['RevenueTTM']):,.0f}"), width=6),
            dbc.Col(html.P(f"Gross Profit (TTM): ${float(company_overview['GrossProfitTTM']):,.0f}"), width=6),
        ]),
        dbc.Row([
            dbc.Col(html.P(f"Market Capitalization: ${float(company_overview['MarketCapitalization']):,.0f}"), width=6),
            dbc.Col(html.P(f"EBITDA (TTM): ${float(company_overview['EBITDA']):,.0f}"), width=6),
        ]),
        dbc.Row([
            dbc.Col(html.P(f"PE Ratio (TTM): {company_overview['PERatio']}"), width=6),
            dbc.Col(html.P(f"Dividend Yield: {company_overview['DividendYield']}"), width=6),
        ]),
        html.P(f"Description: {company_overview['Description']}"),
    ]


    # Plot risk-free rate
    risk_free_trace = go.Scatter(
        x=risk_free_rate_df.index,
        y=risk_free_rate_df['value'],
        mode='lines',
        name='Risk-Free Rate'
    )
    risk_free_layout = go.Layout(
        title='Risk-Free Rate (10-Year US Treasury Yield)',
        yaxis=dict(
            title='Yield'
        )
    )
    # Enable the submit button after processing
    disabled = False
    return (
        go.Figure(data=[trace_close, trace_sma10, trace_sma50,
                earnings_trace, trace_capm_expected_returns], layout=layout),
        company_overview_elements,
        go.Figure(data=[risk_free_trace], layout=risk_free_layout),
        disabled
    )
    
    
if __name__ == '__main__':
    app.run_server(debug=True)
