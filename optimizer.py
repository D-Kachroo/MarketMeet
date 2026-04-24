from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from config import MAX_WEIGHT, MIN_AVG_VOLUME, MIN_CORRELATION, MIN_WEIGHT, PORTFOLIO_VALUE_CAD


def select_candidates(metrics: pd.DataFrame) -> pd.DataFrame:
    if metrics.empty:
        return metrics

    clean = metrics.copy()
    clean['AvgVolume'] = pd.to_numeric(clean['AvgVolume'], errors='coerce').fillna(0.0)
    clean['Correlation'] = pd.to_numeric(clean['Correlation'], errors='coerce').fillna(0.0)
    clean['Beta'] = pd.to_numeric(clean['Beta'], errors='coerce').fillna(1.0)

    strict = clean[
        (clean['AvgVolume'] >= MIN_AVG_VOLUME) &
        (clean['Correlation'] >= MIN_CORRELATION)
    ].copy()

    if len(strict) >= 1:
        ranked = strict.sort_values(['Correlation', 'Beta', 'AvgVolume'], ascending=[False, True, False])
    else:
        ranked = clean.sort_values(['Correlation', 'Beta', 'AvgVolume'], ascending=[False, True, False])

    return ranked


def choose_portfolio(metrics: pd.DataFrame, target_holdings: int) -> list[str]:
    ranked = select_candidates(metrics)

    if ranked.empty:
        return []

    target_holdings = min(target_holdings, len(ranked))
    selected: list[str] = []
    sector_counts: dict[str, int] = {}
    sector_limit = max(2, target_holdings // 4)

    for ticker, row in ranked.iterrows():
        sector = str(row.get('Sector', 'Other'))

        if sector_counts.get(sector, 0) >= sector_limit:
            continue

        selected.append(str(ticker))
        sector_counts[sector] = sector_counts.get(sector, 0) + 1

        if len(selected) >= target_holdings:
            break

    if len(selected) < target_holdings:
        leftovers = [str(ticker) for ticker in ranked.index if str(ticker) not in selected]
        selected.extend(leftovers[: target_holdings - len(selected)])

    return selected[:target_holdings]


def optimize_weights(
    returns: pd.DataFrame,
    benchmark_returns: pd.Series,
    risk_weight: float,
) -> pd.Series:
    if returns.empty:
        return pd.Series(dtype=float)

    returns = returns.copy()
    returns.columns = returns.columns.astype(str)
    returns = returns.replace([np.inf, -np.inf], np.nan).dropna(axis=1, how='all').fillna(0.0)

    columns = [str(col) for col in returns.columns]
    n = len(columns)

    if n == 0:
        return pd.Series(dtype=float)

    lower_bound = MIN_WEIGHT if MIN_WEIGHT * n <= 1 else 0.0
    upper_bound = MAX_WEIGHT if MAX_WEIGHT * n >= 1 else 1.0

    initial = np.ones(n) / n
    benchmark = benchmark_returns.reindex(returns.index).fillna(0.0)

    def objective(weights: np.ndarray) -> float:
        portfolio = returns.values @ weights
        tracking_error = np.std(portfolio - benchmark.values)
        volatility = np.std(portfolio)

        correlation_term = 1.0
        if np.std(portfolio) > 0 and np.std(benchmark.values) > 0:
            corr = np.corrcoef(portfolio, benchmark.values)[0, 1]
            correlation_term = 1 - corr if not np.isnan(corr) else 1.0

        return tracking_error + risk_weight * volatility + 0.1 * correlation_term

    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
    bounds = [(lower_bound, upper_bound) for _ in range(n)]

    result = minimize(
        objective,
        initial,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options={'maxiter': 1000},
    )

    values = initial if not result.success else result.x
    weights = pd.Series(values, index=columns, dtype=float)
    weights = weights.clip(lower=0.0)

    if weights.sum() <= 0:
        weights = pd.Series(initial, index=columns, dtype=float)

    return weights / weights.sum()


def build_portfolio_table(prices: pd.Series, weights: pd.Series, cadusd_rate: float) -> pd.DataFrame:
    if weights.empty:
        return pd.DataFrame(columns=['Ticker', 'PriceCAD', 'Shares', 'ValueCAD', 'Weight%'])

    if cadusd_rate <= 0 or pd.isna(cadusd_rate):
        cadusd_rate = 0.73

    weights = weights.copy()
    weights.index = weights.index.astype(str)

    prices_local = prices.copy()
    prices_local.index = prices_local.index.astype(str)
    prices_local = pd.to_numeric(prices_local.reindex(weights.index), errors='coerce')

    prices_cad = pd.Series(
        [
            price if ticker.endswith('.TO') else price / cadusd_rate
            for ticker, price in prices_local.items()
        ],
        index=weights.index,
        dtype=float,
    )

    gross_allocation = weights * PORTFOLIO_VALUE_CAD
    gross_shares = gross_allocation / prices_cad

    fees_usd = np.minimum(2.15, 0.001 * gross_shares)
    fees_cad = fees_usd / cadusd_rate

    net_capital = PORTFOLIO_VALUE_CAD - float(fees_cad.sum())
    allocation = weights * net_capital
    shares = allocation / prices_cad
    value_cad = shares * prices_cad

    portfolio = pd.DataFrame({
        'Ticker': weights.index.astype(str),
        'PriceCAD': prices_cad.values,
        'Shares': shares.values,
        'ValueCAD': value_cad.values,
        'Weight%': (value_cad / value_cad.sum() * 100).values,
    })

    portfolio['Ticker'] = portfolio['Ticker'].astype(str)

    return portfolio.sort_values('Weight%', ascending=False).reset_index(drop=True)


def portfolio_stats(returns: pd.DataFrame, weights: pd.Series, benchmark_returns: pd.Series) -> dict:
    if weights.empty or returns.empty:
        return {
            'ExpectedDailyReturnPct': 0.0,
            'DailyVolatilityPct': 0.0,
            'WeeklyVolatilityPct': 0.0,
            'Beta': 0.0,
            'Correlation': 0.0,
            'TrackingErrorPct': 0.0,
        }

    returns = returns.copy()
    returns.columns = returns.columns.astype(str)
    returns = returns.reindex(columns=weights.index).fillna(0.0)

    portfolio = returns.values @ weights.values
    portfolio = pd.Series(portfolio, index=returns.index)

    benchmark = benchmark_returns.reindex(portfolio.index).fillna(0.0)

    covariance = float(np.cov(portfolio, benchmark)[0, 1]) if len(portfolio) > 1 else np.nan
    variance_b = float(np.var(benchmark))
    beta = covariance / variance_b if variance_b > 0 else np.nan

    correlation = (
        float(np.corrcoef(portfolio, benchmark)[0, 1])
        if np.std(portfolio) > 0 and np.std(benchmark) > 0
        else np.nan
    )

    tracking_error = float(np.std(portfolio - benchmark))
    weekly = ((1 + portfolio).resample('W-FRI').prod() - 1).std() if len(portfolio) > 1 else 0.0

    return {
        'ExpectedDailyReturnPct': float(portfolio.mean() * 100),
        'DailyVolatilityPct': float(portfolio.std() * 100),
        'WeeklyVolatilityPct': float(weekly * 100),
        'Beta': 0.0 if np.isnan(beta) else float(beta),
        'Correlation': 0.0 if np.isnan(correlation) else float(correlation),
        'TrackingErrorPct': tracking_error * 100,
    }
