from __future__ import annotations

import asyncio

import numpy as np
import pandas as pd
import yfinance as yf

from config import BENCHMARKS, FX_FALLBACK, LARGE_CAP_CAD, SMALL_CAP_CAD, SECTOR_FALLBACKS


def clean_tickers(raw: str | list[str]) -> list[str]:
    if isinstance(raw, str):
        parts = raw.replace('\n', ',').split(',')
    else:
        parts = raw

    seen = set()
    tickers: list[str] = []

    for item in parts:
        ticker = str(item).strip().upper()
        if ticker and ticker not in seen:
            seen.add(ticker)
            tickers.append(ticker)

    return tickers


def weekly_volatility(returns: pd.Series) -> float:
    weekly = (1 + returns).resample('W-FRI').prod() - 1
    weekly = weekly.dropna()
    return float(weekly.std()) if not weekly.empty else np.nan


def average_volume(series: pd.Series) -> float:
    values = series.dropna()
    if values.empty:
        return np.nan

    month = values.index.to_period('M')
    valid_months = month.value_counts()[lambda x: x >= 18].index
    filtered = values[month.isin(valid_months)]

    return float(filtered.mean()) if not filtered.empty else np.nan


def get_cadusd_rate() -> float:
    fx = yf.Ticker('CADUSD=X').history(period='5d')['Close']
    if fx.empty:
        return FX_FALLBACK
    return float(fx.iloc[-1])


def download_market_data(tickers: list[str], start: str, end: str | None = None):
    symbols = tickers + BENCHMARKS

    raw = yf.download(
        symbols,
        start=start,
        end=end,
        auto_adjust=False,
        progress=False,
        group_by='column',
        threads=True,
    )

    close = raw['Close'].copy()
    volume = raw['Volume'].copy()

    valid = [ticker for ticker in tickers if ticker in close.columns]
    invalid = sorted(set(tickers) - set(valid))

    benchmark = ((close[BENCHMARKS[0]] + close[BENCHMARKS[1]]) / 2).dropna()
    benchmark_returns = benchmark.pct_change().dropna()

    prices = close[valid].loc[benchmark_returns.index].dropna(how='all', axis=1)
    prices = prices.loc[:, ~prices.columns.duplicated(keep='first')]
    prices.columns = prices.columns.astype(str)

    volume = volume.reindex(columns=prices.columns).loc[prices.index]
    volume.columns = volume.columns.astype(str)

    return prices, volume, benchmark, benchmark_returns, invalid


async def _fetch_single_metadata(ticker: str, cadusd_rate: float) -> dict:
    try:
        obj = yf.Ticker(ticker)

        try:
            info = await asyncio.to_thread(lambda: obj.info)
        except Exception:
            info = {}

        market_cap_raw = info.get('marketCap', np.nan)

        if isinstance(market_cap_raw, (int, float)) and not pd.isna(market_cap_raw):
            market_cap_cad = market_cap_raw if ticker.endswith('.TO') else market_cap_raw / cadusd_rate
        else:
            market_cap_cad = np.nan

        return {
            'Ticker': str(ticker),
            'Sector': info.get('sector', np.nan),
            'Industry': info.get('industry', np.nan),
            'MarketCapCAD': market_cap_cad,
            'SmallCap': bool(market_cap_cad < SMALL_CAP_CAD) if not pd.isna(market_cap_cad) else False,
            'LargeCap': bool(market_cap_cad > LARGE_CAP_CAD) if not pd.isna(market_cap_cad) else False,
        }

    except Exception:
        return {
            'Ticker': str(ticker),
            'Sector': np.nan,
            'Industry': np.nan,
            'MarketCapCAD': np.nan,
            'SmallCap': False,
            'LargeCap': False,
        }


async def fetch_metadata(tickers: list[str], cadusd_rate: float) -> pd.DataFrame:
    rows = await asyncio.gather(*[_fetch_single_metadata(ticker, cadusd_rate) for ticker in tickers])
    frame = pd.DataFrame(rows)
    frame['Ticker'] = frame['Ticker'].astype(str)
    return frame.set_index('Ticker')


def build_metrics(
    prices: pd.DataFrame,
    volume: pd.DataFrame,
    benchmark_returns: pd.Series,
    metadata: pd.DataFrame,
) -> pd.DataFrame:
    prices = prices.copy()
    volume = volume.copy()

    prices.columns = prices.columns.astype(str)
    volume.columns = volume.columns.astype(str)

    returns = prices.pct_change().dropna()

    rows: list[dict] = []

    for ticker in prices.columns:
        r = returns[str(ticker)].dropna()
        b = benchmark_returns.reindex(r.index).dropna()

        idx = r.index.intersection(b.index)
        r = pd.Series(r.loc[idx], dtype=float)
        b = pd.Series(b.loc[idx], dtype=float)

        if len(r) < 2 or len(b) < 2:
            std = cov = beta = corr = weekly = idio = np.nan
        else:
            std = float(r.std())
            cov = float(r.cov(b))
            variance_b = float(b.var())
            beta = cov / variance_b if variance_b > 0 else np.nan
            corr = float(r.corr(b))
            weekly = weekly_volatility(r)

            alpha = r.mean() - beta * b.mean() if not np.isnan(beta) else np.nan
            residuals = r - (alpha + beta * b) if not np.isnan(alpha) else pd.Series(dtype=float)
            idio = float(residuals.std()) if not residuals.empty else np.nan

        rows.append({
            'Ticker': str(ticker),
            'AvgVolume': average_volume(volume[str(ticker)]),
            'DailyVolatility': std,
            'Covariance': cov,
            'Beta': beta,
            'Correlation': corr,
            'WeeklyVolatility': weekly,
            'IdiosyncraticVolatility': idio,
        })

    metrics = pd.DataFrame(rows)
    metrics['Ticker'] = metrics['Ticker'].astype(str)

    metadata = metadata.reset_index().copy()
    metadata['Ticker'] = metadata['Ticker'].astype(str)

    joined = metrics.merge(metadata, on='Ticker', how='left')

    joined['Sector'] = joined.apply(
        lambda row: row['Sector']
        if pd.notna(row['Sector']) and str(row['Sector']).strip() != ''
        else SECTOR_FALLBACKS.get(str(row['Ticker']), 'Other'),
        axis=1,
    )

    joined['Industry'] = joined['Industry'].fillna(joined['Sector'])
    joined['MarketCapCAD'] = joined['MarketCapCAD'].fillna(0)
    joined['SmallCap'] = joined['SmallCap'].fillna(False).astype(bool)
    joined['LargeCap'] = joined['LargeCap'].fillna(False).astype(bool)

    joined = joined.dropna(subset=['AvgVolume', 'Correlation', 'Beta'], how='any')
    joined = joined.sort_values(['Correlation', 'AvgVolume'], ascending=[False, False])

    return joined.set_index('Ticker')

    return joined.set_index('Ticker')
