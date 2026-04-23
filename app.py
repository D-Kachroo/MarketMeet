from __future__ import annotations

import asyncio
from datetime import date

import pandas as pd
import streamlit as st

from src.config import DEFAULT_START, DEFAULT_UNIVERSE
from src.data import build_metrics, clean_tickers, download_market_data, fetch_metadata, get_cadusd_rate
from src.optimizer import build_portfolio_table, choose_portfolio, optimize_weights, portfolio_stats
from src.ui import cumulative_chart, metric_card, sector_donut, weights_bar

st.set_page_config(page_title='MarketMeet', page_icon='📈', layout='wide')

st.markdown(
    """
    <style>
    .stApp {
        background: radial-gradient(circle at top left, #111827 0%, #0b1020 45%, #050816 100%);
        color: #f9fafb;
    }
    .block-container {
        padding-top: 1.4rem;
        padding-bottom: 2rem;
        max-width: 1350px;
    }
    h1, h2, h3 { letter-spacing: -0.02em; }
    div[data-testid='stMetric'] {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.08);
        padding: 14px;
        border-radius: 18px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown("## MarketMeet")
st.caption('Portfolio optimization and benchmark tracking with a cleaner front end and a faster workflow.')

with st.sidebar:
    st.subheader('Controls')
    raw_tickers = st.text_area('Ticker universe', ', '.join(DEFAULT_UNIVERSE), height=180)
    start_date = st.date_input('Start date', value=pd.to_datetime(DEFAULT_START).date(), min_value=date(2018, 1, 1), max_value=date.today())
    holdings = st.slider('Number of holdings', min_value=8, max_value=25, value=15)
    risk_aversion = st.slider('Risk penalty', min_value=0.0, max_value=2.0, value=0.8, step=0.1)
    run = st.button('Build portfolio', use_container_width=True)

if not run:
    st.info('Set the inputs in the sidebar and click **Build portfolio**.')
    st.stop()

try:
    tickers = clean_tickers(raw_tickers)
    if len(tickers) < holdings:
        st.error('The ticker universe must contain at least as many symbols as the number of holdings.')
        st.stop()

    with st.spinner('Downloading market data and optimizing portfolio...'):
        prices, volume, benchmark, benchmark_returns, invalid = download_market_data(tickers, str(start_date))
        cadusd_rate = get_cadusd_rate()
        metadata = asyncio.run(fetch_metadata(list(prices.columns), cadusd_rate))
        metrics = build_metrics(prices, volume, benchmark_returns, metadata)
        selected = choose_portfolio(metrics, holdings)
        selected_prices = prices[selected]
        selected_returns = selected_prices.pct_change().dropna()
        weights = optimize_weights(selected_returns, benchmark_returns, risk_aversion)
        portfolio = build_portfolio_table(selected_prices.iloc[-1], weights, cadusd_rate)
        stats = portfolio_stats(selected_returns, weights, benchmark_returns)

    top = st.columns(5)
    cards = [
        ('Holdings', str(len(portfolio))),
        ('Expected Daily Return', f"{stats['ExpectedDailyReturnPct']:.2f}%"),
        ('Daily Volatility', f"{stats['DailyVolatilityPct']:.2f}%"),
        ('Beta', f"{stats['Beta']:.2f}"),
        ('Tracking Error', f"{stats['TrackingErrorPct']:.2f}%"),
    ]
    for col, (label, value) in zip(top, cards):
        col.markdown(metric_card(label, value), unsafe_allow_html=True)

    if invalid:
        st.warning('Removed invalid tickers: ' + ', '.join(invalid))

    left, right = st.columns([1.3, 1])
    with left:
        st.plotly_chart(cumulative_chart(selected_returns, weights, benchmark_returns), use_container_width=True)
    with right:
        st.plotly_chart(sector_donut(portfolio, metrics), use_container_width=True)

    left, right = st.columns([1.2, 1])
    with left:
        st.plotly_chart(weights_bar(portfolio), use_container_width=True)
    with right:
        st.subheader('Final Portfolio')
        view = portfolio.copy()
        view['PriceCAD'] = view['PriceCAD'].map(lambda x: round(float(x), 2))
        view['Shares'] = view['Shares'].map(lambda x: round(float(x), 2))
        view['ValueCAD'] = view['ValueCAD'].map(lambda x: round(float(x), 2))
        view['WeightPct'] = view['WeightPct'].map(lambda x: round(float(x), 2))
        st.dataframe(view, use_container_width=True, hide_index=True)
        csv = portfolio.to_csv(index=False).encode('utf-8')
        st.download_button('Download portfolio CSV', data=csv, file_name='marketmeet_portfolio.csv', mime='text/csv', use_container_width=True)

    st.subheader('Filtered Security Metrics')
    metrics_view = metrics.loc[selected].copy()[['Sector', 'Industry', 'MarketCapCAD', 'AvgVolume', 'Beta', 'Correlation', 'WeeklyVolatility', 'IdiosyncraticVolatility']]
    metrics_view['MarketCapCAD'] = metrics_view['MarketCapCAD'].map(lambda x: round(float(x), 0) if pd.notna(x) else x)
    metrics_view['AvgVolume'] = metrics_view['AvgVolume'].map(lambda x: round(float(x), 0) if pd.notna(x) else x)
    for col in ['Beta', 'Correlation', 'WeeklyVolatility', 'IdiosyncraticVolatility']:
        metrics_view[col] = metrics_view[col].map(lambda x: round(float(x), 4) if pd.notna(x) else x)
    st.dataframe(metrics_view.reset_index(names='Ticker'), use_container_width=True, hide_index=True)

except Exception as exc:
    st.error(f'Unable to build the portfolio: {exc}')
