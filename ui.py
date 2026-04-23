from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def metric_card(label: str, value: str):
    return f"""
    <div style='padding:18px 16px;border:1px solid rgba(255,255,255,0.08);border-radius:18px;background:rgba(255,255,255,0.03)'>
        <div style='font-size:0.82rem;color:#9ca3af;margin-bottom:6px'>{label}</div>
        <div style='font-size:1.45rem;font-weight:700;color:#f9fafb'>{value}</div>
    </div>
    """


def weights_bar(portfolio: pd.DataFrame):
    fig = px.bar(
        portfolio.sort_values('WeightPct', ascending=True),
        x='WeightPct',
        y='Ticker',
        orientation='h',
        title='Position Weights',
        text='WeightPct',
    )
    fig.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
    fig.update_layout(height=520, margin=dict(l=10, r=20, t=60, b=10))
    return fig


def sector_donut(portfolio: pd.DataFrame, metrics: pd.DataFrame):
    df = portfolio.merge(metrics[['Sector']], left_on='Ticker', right_index=True, how='left')
    sector = df.groupby('Sector', as_index=False)['WeightPct'].sum().sort_values('WeightPct', ascending=False)
    fig = px.pie(sector, names='Sector', values='WeightPct', hole=0.58, title='Sector Mix')
    fig.update_layout(height=420, margin=dict(l=10, r=10, t=60, b=10))
    return fig


def cumulative_chart(returns: pd.DataFrame, weights, benchmark_returns: pd.Series):
    portfolio_returns = pd.Series(returns[weights.index].values @ weights.values, index=returns.index, name='MarketMeet')
    benchmark = benchmark_returns.reindex(portfolio_returns.index).fillna(0)
    growth = pd.DataFrame({
        'MarketMeet': (1 + portfolio_returns).cumprod(),
        'Benchmark': (1 + benchmark).cumprod(),
    })
    fig = go.Figure()
    for column in growth.columns:
        fig.add_trace(go.Scatter(x=growth.index, y=growth[column], mode='lines', name=column))
    fig.update_layout(title='Cumulative Growth', height=420, margin=dict(l=10, r=10, t=60, b=10))
    return fig
