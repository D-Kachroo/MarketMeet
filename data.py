from __future__ import annotations

import asyncio
from datetime import datetime

import numpy as np
import pandas as pd
import yfinance as yf

from config import BENCHMARKS, FX_FALLBACK, LARGE_CAP_CAD, SMALL_CAP_CAD


def clean_tickers(raw: str | list[str]) -> list[str]:
    if isinstance(raw, str):
        parts = raw.replace('\n', ',').split(',')
    else:
        parts = raw
    seen = set()
    out = []
    for item in parts:
        ticker = str(item).strip().upper()
        if ticker and ticker not in seen:
            seen.add(ticker)
            out.append(ticker)
    return out


def weekly_volatility(returns: pd.Series) -> float:
    weekly = (1 + returns).resample('W-FRI').prod() - 1
    weekly = weekly.dropna()
    return float(weekly.std()) if not weekly.empty else np.nan


def average_volume(series: pd.Series) -> float:
    s = series.dropna()
    if s.empty:
        return np.nan
    month = s.index.to_period('M')
    valid = month.value_counts()[lambda x: x >= 18].index
    filtered = s[month.isin(valid)]
    return float(filtered.mean()) if not filtered.empty else np.nan


def get_cadusd_rate() -> float:
    fx = yf.Ticker('CADUSD=X').history(period='5d')['Close']
    if len(fx) == 0:
        return FX_FALLBACK
    return float(fx.iloc[-1])


def download_market_data(tickers: list[str], start: str, end: str | None = None):
    tickers_all = tickers + BENCHMARKS
    raw = yf.download(tickers_all, start=start, end=end, auto_adjust=False, progress=False, group_by='column', threads=True)
    close = raw['Close'].copy()
    volume = raw['Volume'].copy()
    valid = [t for t in tickers if t in close.columns]
    invalid = sorted(set(tickers) - set(valid))
    benchmark = ((close[BENCHMARKS[0]] + close[BENCHMARKS[1]]) / 2).dropna()
    benchmark_returns = benchmark.pct_change().dropna()
    prices = close[valid].loc[benchmark_returns.index].dropna(how='all', axis=1)
    returns = prices.pct_change().dropna()
    volume = volume[prices.columns].loc[prices.index]
    return prices, volume, benchmark, benchmark_returns, invalid


async def _fetch_single_metadata(ticker: str, usd_to_cad: float) -> dict:
    try:
        obj = yf.Ticker(ticker)
        try:
            info = await asyncio.to_thread(lambda: obj.info)
        except Exception:
            info = {}
        market_cap_raw = info.get('marketCap', np.nan)
        if isinstance(market_cap_raw, (int, float)) and not pd.isna(market_cap_raw):
            market_cap_cad = market_cap_raw if ticker.endswith('.TO') else market_cap_raw / usd_to_cad
        else:
            market_cap_cad = np.nan
        return {
            'Ticker': ticker,
            'Sector': info.get('sector', np.nan),
            'Industry': info.get('industry', np.nan),
            'MarketCapCAD': market_cap_cad,
            'SmallCap': bool(market_cap_cad < SMALL_CAP_CAD) if not pd.isna(market_cap_cad) else False,
            'LargeCap': bool(market_cap_cad > LARGE_CAP_CAD) if not pd.isna(market_cap_cad) else False,
        }
    except Exception:
        return {
            'Ticker': ticker,
            'Sector': np.nan,
            'Industry': np.nan,
            'MarketCapCAD': np.nan,
            'SmallCap': False,
            'LargeCap': False,
        }


async def fetch_metadata(tickers: list[str], usd_to_cad: float) -> pd.DataFrame:
    rows = await asyncio.gather(*[_fetch_single_metadata(t, usd_to_cad) for t in tickers])
    return pd.DataFrame(rows).set_index('Ticker')


def build_metrics(prices: pd.DataFrame, volume: pd.DataFrame, benchmark_returns: pd.Series, metadata: pd.DataFrame) -> pd.DataFrame:
    returns = prices.pct_change().dropna()
    rows = []
    for ticker in prices.columns:
        r = returns[ticker].dropna()
        b = benchmark_returns.reindex(r.index).dropna()
        idx = r.index.intersection(b.index)
        r = pd.Series(r.loc[idx], dtype=float)
        b = pd.Series(b.loc[idx], dtype=float)
        if len(r) < 2 or len(b) < 2:
            beta = corr = cov = weekly = idio = std = np.nan
        else:
            std = float(r.std())
            cov = float(r.cov(b))
            beta = cov / float(b.var()) if float(b.var()) > 0 else np.nan
            corr = float(r.corr(b))
            weekly = weekly_volatility(r)
            alpha = r.mean() - beta * b.mean() if not np.isnan(beta) else np.nan
            residuals = r - (alpha + beta * b) if not np.isnan(alpha) else pd.Series(dtype=float)
            idio = float(residuals.std()) if len(residuals) else np.nan
        rows.append({
            'Ticker': ticker,
            'AvgVolume': average_volume(volume[ticker]),
            'DailyVolatility': std,
            'Covariance': cov,
            'Beta': beta,
            'Correlation': corr,
            'WeeklyVolatility': weekly,
            'IdiosyncraticVolatility': idio,
        })
    metrics = pd.DataFrame(rows).set_index('Ticker')
    joined = metrics.join(metadata, how='left')
    joined = joined.dropna(subset=['Sector', 'Industry'], how='any')
    return joined.sort_values(['Correlation', 'AvgVolume'], ascending=[False, False])
