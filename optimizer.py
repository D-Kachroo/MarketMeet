from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from config import MAX_WEIGHT, MIN_AVG_VOLUME, MIN_CORRELATION, MIN_WEIGHT, PORTFOLIO_VALUE_CAD


def select_candidates(metrics: pd.DataFrame) -> pd.DataFrame:
    filtered = metrics[(metrics['AvgVolume'] >= MIN_AVG_VOLUME) & (metrics['Correlation'] >= MIN_CORRELATION)].copy()
    filtered = filtered.sort_values(['Correlation', 'Beta'], ascending=[False, True])
    return filtered


def choose_portfolio(metrics: pd.DataFrame, n_holdings: int) -> list[str]:
    ranked = select_candidates(metrics)
    if ranked.empty:
        return []
    diversified = []
    sector_counts = {}
    for ticker, row in ranked.iterrows():
        sector = row['Sector']
        if sector_counts.get(sector, 0) >= max(2, n_holdings // 4):
            continue
        diversified.append(ticker)
        sector_counts[sector] = sector_counts.get(sector, 0) + 1
        if len(diversified) >= n_holdings:
            break
    if len(diversified) < n_holdings:
        leftovers = [t for t in ranked.index if t not in diversified]
        diversified.extend(leftovers[: n_holdings - len(diversified)])
    return diversified[:n_holdings]


def optimize_weights(returns: pd.DataFrame, benchmark_returns: pd.Series, risk_aversion: float) -> pd.Series:
    cols = list(returns.columns)
    n = len(cols)
    if n == 0:
        return pd.Series(dtype=float)
    min_w = min(MIN_WEIGHT, 1 / n)
    max_w = max(MAX_WEIGHT, 1 / n)
    init = np.ones(n) / n
    benchmark_aligned = benchmark_returns.reindex(returns.index).fillna(0)

    def objective(weights: np.ndarray) -> float:
        portfolio = returns.values @ weights
        tracking_error = np.std(portfolio - benchmark_aligned.values)
        volatility = np.std(portfolio)
        correlation_penalty = 1 - np.corrcoef(portfolio, benchmark_aligned.values)[0, 1] if np.std(portfolio) > 0 and np.std(benchmark_aligned.values) > 0 else 1
        return tracking_error + risk_aversion * volatility + 0.1 * correlation_penalty

    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
    bounds = [(min_w, max_w) for _ in range(n)]
    result = minimize(objective, init, method='SLSQP', bounds=bounds, constraints=constraints)
    if not result.success:
        return pd.Series(init, index=cols)
    weights = pd.Series(result.x, index=cols)
    weights = weights / weights.sum()
    return weights


def build_portfolio_table(prices: pd.Series, weights: pd.Series, cadusd_rate: float) -> pd.DataFrame:
    prices_local = prices.reindex(weights.index)
    prices_cad = pd.Series(
        [price if ticker.endswith('.TO') else price / cadusd_rate for ticker, price in prices_local.items()],
        index=weights.index,
        dtype=float,
    )
    gross_alloc = weights * PORTFOLIO_VALUE_CAD
    gross_shares = gross_alloc / prices_cad
    fees_usd = np.minimum(2.15, 0.001 * gross_shares)
    fees_cad = fees_usd / cadusd_rate
    net_capital = PORTFOLIO_VALUE_CAD - float(fees_cad.sum())
    allocation = weights * net_capital
    shares = allocation / prices_cad
    values = shares * prices_cad
    final = pd.DataFrame({
        'Ticker': weights.index,
        'PriceCAD': prices_cad.values,
        'Shares': shares.values,
        'ValueCAD': values.values,
        'WeightPct': (values / values.sum() * 100).values,
    })
    return final.sort_values('WeightPct', ascending=False).reset_index(drop=True)


def portfolio_stats(returns: pd.DataFrame, weights: pd.Series, benchmark_returns: pd.Series) -> dict:
    portfolio = returns[weights.index].values @ weights.values
    portfolio = pd.Series(portfolio, index=returns.index)
    benchmark = benchmark_returns.reindex(portfolio.index).fillna(0)
    covariance = float(np.cov(portfolio, benchmark)[0, 1]) if len(portfolio) > 1 else np.nan
    beta = covariance / float(np.var(benchmark)) if float(np.var(benchmark)) > 0 else np.nan
    corr = float(np.corrcoef(portfolio, benchmark)[0, 1]) if np.std(portfolio) > 0 and np.std(benchmark) > 0 else np.nan
    tracking_error = float(np.std(portfolio - benchmark))
    return {
        'ExpectedDailyReturnPct': float(portfolio.mean() * 100),
        'DailyVolatilityPct': float(portfolio.std() * 100),
        'WeeklyVolatilityPct': float(((1 + portfolio).resample('W-FRI').prod() - 1).std() * 100),
        'Beta': beta,
        'Correlation': corr,
        'TrackingErrorPct': tracking_error * 100,
    }
