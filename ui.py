from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def metric_card(label: str, value: str) -> str:
    return f"""
    <div style='padding:18px 16px;border:1px solid rgba(255,255,255,0.08);border-radius:18px;background:rgba(255,255,255,0.03)'>
        <div style='font-size:0.82rem;color:#9ca3af;margin-bottom:6px'>{label}</div>
        <div style='font-size:1.45rem;font-weight:700;color:#f9fafb'>{value}</div>
    </div>
    """


def _base_layout(fig, title: str, height: int):
    fig.update_layout(
        title=dict(text=title, font=dict(color='white')),
        height=height,
        margin=dict(l=10, r=10, t=60, b=10),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(255,255,255,0.02)',
        font=dict(color='white'),
        legend=dict(font=dict(color='white')),
    )

    fig.update_xaxes(
        gridcolor='rgba(255,255,255,0.08)',
        zerolinecolor='rgba(255,255,255,0.08)',
        color='white',
        title_font=dict(color='white'),
        tickfont=dict(color='white'),
    )

    fig.update_yaxes(
        gridcolor='rgba(255,255,255,0.08)',
        zerolinecolor='rgba(255,255,255,0.08)',
        color='white',
        title_font=dict(color='white'),
        tickfont=dict(color='white'),
    )

    return fig


def weights_bar(portfolio: pd.DataFrame):
    fig = px.bar(
        portfolio.sort_values('Weight%', ascending=True),
        x='Weight%',
        y='Ticker',
        orientation='h',
        title='Weight of Each Stock (%)',
        text='Weight%',
    )
    fig.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
    return _base_layout(fig, 'Weight of Each Stock (%)', 520)


def sector_donut(portfolio: pd.DataFrame, metrics: pd.DataFrame):
    if portfolio.empty:
        fig = go.Figure()
        return _base_layout(fig, 'Sector Allocation', 420)

    portfolio_view = portfolio.copy()
    portfolio_view['Ticker'] = portfolio_view['Ticker'].astype(str)

    metric_view = metrics.reset_index()[['Ticker', 'Sector']].copy()
    metric_view['Ticker'] = metric_view['Ticker'].astype(str)

    joined = portfolio_view.merge(metric_view, on='Ticker', how='left')
    sector = joined.groupby('Sector', as_index=False)['Weight%'].sum().sort_values('Weight%', ascending=False)

    fig = px.pie(sector, names='Sector', values='Weight%', hole=0.58)
    return _base_layout(fig, 'Sector Allocation', 420)


def cumulative_chart(returns: pd.DataFrame, weights: pd.Series, benchmark_returns: pd.Series):
    if weights.empty or returns.empty:
        growth = pd.DataFrame({'Benchmark': (1 + benchmark_returns).cumprod()})
    else:
        portfolio_returns = pd.Series(
            returns[weights.index].values @ weights.values,
            index=returns.index,
            name='MarketMeet',
        )
        benchmark = benchmark_returns.reindex(portfolio_returns.index).fillna(0.0)
        growth = pd.DataFrame({
            'MarketMeet': (1 + portfolio_returns).cumprod(),
            'S&P 500 + TSX 60': (1 + benchmark).cumprod(),
        })

    fig = go.Figure()

    for column in growth.columns:
        fig.add_trace(
            go.Scatter(
                x=growth.index,
                y=growth[column],
                mode='lines',
                name=column,
            )
        )

    fig = _base_layout(fig, 'Cumulative Return (Portfolio vs Benchmark)', 420)

    fig.update_xaxes(
        title_text='Date',
        tickformat='%b %Y',
        tickangle=0,
    )

    fig.update_yaxes(
        title_text='Investment Value (Starting at $1)',
    )

    return fig
