from __future__ import annotations

import asyncio
from datetime import date

import pandas as pd
import streamlit as st

from config import DEFAULT_START, DEFAULT_TICKERS
from data import build_metrics, clean_tickers, download_market_data, fetch_metadata, get_cadusd_rate
from optimizer import build_portfolio_table, choose_portfolio, optimize_weights, portfolio_stats
from ui import cumulative_chart, metric_card, sector_donut, weights_bar

st.set_page_config(
    page_title='MarketMeet',
    page_icon='📈',
    layout='wide',
    initial_sidebar_state='expanded',
)

st.markdown(
    """
    <style>
    [data-testid="stHeader"] {
        background: transparent;
        height: 0rem;
    }

    [data-testid="stToolbar"] {
        display: none;
    }

    .stApp {
        background: radial-gradient(circle at top left, #111827 0%, #0b1020 45%, #050816 100%);
        color: #f9fafb;
    }

    .block-container {
        padding-top: 0.2rem;
        padding-bottom: 2rem;
        max-width: 1350px;
    }

    h1, h2, h3 {
        letter-spacing: -0.02em;
    }

    .stTextArea textarea,
    .stDateInput input {
        border-radius: 14px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('# MarketMeet')
st.caption('Developers: David Kachroo, Tanvi Batchu, and Johan Naresh')

with st.sidebar:
    st.subheader('Sidebar Controls')

    raw_tickers = st.text_area(
        'Ticker List:',
        ', '.join(DEFAULT_TICKERS),
        height=180,
    )

    start_date = st.date_input(
        'Start Date (YYYY/MM/DD):',
        value=pd.to_datetime(DEFAULT_START).date(),
        min_value=date(2018, 1, 1),
        max_value=date.today(),
    )

    holdings = st.slider(
        'Number of Stocks/Holdings:',
        min_value=8,
        max_value=25,
        value=15,
    )

    risk_weight = st.slider(
        'Volatility Weighting:',
        min_value=0.0,
        max_value=2.0,
        value=0.8,
        step=0.1,
        help='Higher values prioritize lower volatility when selecting stock weights.',
    )

    run = st.button('Generate portfolio', use_container_width=True)

if not run:
    st.info('Set the inputs in the sidebar and click "Generate Portfolio".')
    st.stop()

try:
    tickers = clean_tickers(raw_tickers)

    if len(tickers) < holdings:
        st.error('The ticker list must contain at least as many symbols as the target number of stocks.')
        st.stop()

    with st.spinner('Downloading yfinance market data and generating the portfolio...'):
        prices, volume, benchmark, benchmark_returns, invalid = download_market_data(tickers, str(start_date))

        cadusd_rate = get_cadusd_rate()
        metadata = asyncio.run(fetch_metadata(list(prices.columns), cadusd_rate))
        metrics = build_metrics(prices, volume, benchmark_returns, metadata)

        selected = choose_portfolio(metrics, holdings)
        if not selected:
            st.error('No stocks passed the liquidity and benchmark-correlation filters. Add more tickers or lower the target number of stocks.')
            st.stop()

        selected_prices = prices[selected]
        selected_returns = selected_prices.pct_change().dropna()

        weights = optimize_weights(selected_returns, benchmark_returns, risk_weight)
        portfolio = build_portfolio_table(selected_prices.iloc[-1], weights, cadusd_rate)
        stats = portfolio_stats(selected_returns, weights, benchmark_returns)

    top = st.columns(5)
    cards = [
        ('Number of Stocks', str(len(portfolio))),
        ('Expected Return (Daily)', f"{stats['ExpectedDailyReturnPct']:.2f}%"),
        ('Volatility (Daily)', f"{stats['DailyVolatilityPct']:.2f}%"),
        ('Beta (TSX 60 / S&P 500)', f"{stats['Beta']:.2f}"),
        ('Tracking Error', f"{stats['TrackingErrorPct']:.2f}%"),
    ]

    for column, (label, value) in zip(top, cards):
        column.markdown(metric_card(label, value), unsafe_allow_html=True)

    if invalid:
        st.warning('Removed invalid tickers: ' + ', '.join(invalid))

    left, right = st.columns([1.3, 1])

    with left:
        st.plotly_chart(
            cumulative_chart(selected_returns, weights, benchmark_returns),
            use_container_width=True,
        )

    with right:
        st.plotly_chart(
            sector_donut(portfolio, metrics),
            use_container_width=True,
        )

    left, right = st.columns([1.2, 1])

    with left:
        st.plotly_chart(weights_bar(portfolio), use_container_width=True)

    with right:
        st.subheader('Recommended Portfolio')

        view = portfolio.copy()
        for column in ['PriceCAD', 'Shares', 'ValueCAD', 'Weight%']:
            view[column] = view[column].map(lambda value: round(float(value), 2))

        st.dataframe(view, use_container_width=True, hide_index=True)

        csv = portfolio.to_csv(index=False).encode('utf-8')
        st.download_button(
            'Download CSV File',
            data=csv,
            file_name='marketmeet_portfolio.csv',
            mime='text/csv',
            use_container_width=True,
        )

    st.subheader('Metadata of Stocks')

    metrics_view = metrics.loc[selected].copy()[[
        'Sector',
        'Industry',
        'MarketCapCAD',
        'AvgVolume',
        'Beta',
        'Correlation',
        'WeeklyVolatility',
        'IdiosyncraticVolatility',
    ]]

    metrics_view['MarketCapCAD'] = metrics_view['MarketCapCAD'].map(
        lambda value: round(float(value), 0) if pd.notna(value) else value
    )
    metrics_view['AvgVolume'] = metrics_view['AvgVolume'].map(
        lambda value: round(float(value), 0) if pd.notna(value) else value
    )

    for column in ['Beta', 'Correlation', 'WeeklyVolatility', 'IdiosyncraticVolatility']:
        metrics_view[column] = metrics_view[column].map(
            lambda value: round(float(value), 4) if pd.notna(value) else value
        )

    st.dataframe(
        metrics_view.reset_index(names='Ticker'),
        use_container_width=True,
        hide_index=True,
    )

except Exception as error:
    st.error(f'Unable to build the portfolio: {error}')
