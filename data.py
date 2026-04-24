from __future__ import annotations

import asyncio

import numpy as np
import pandas as pd
import yfinance as yf

from config import BENCHMARKS, DEFAULT_SECTOR_FALLBACK, FX_FALLBACK, LARGE_CAP_CAD, SMALL_CAP_CAD


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
    values = pd.to_numeric(series, errors='coerce').dropna()
    values = values[values > 0]

    if values.empty:
        return np.nan

    return float(values.tail(60).mean())


def get_cadusd_rate() -> float:
    try:
        fx = yf.Ticker('CADUSD=X').history(period='5d', auto_adjust=True)['Close']
        fx = pd.to_numeric(fx, errors='coerce').dropna()

        if fx.empty or float(fx.iloc[-1]) <= 0:
            return FX_FALLBACK

        return float(fx.iloc[-1])

    except Exception:
        return FX_FALLBACK


def _extract_field(raw: pd.DataFrame, field: str) -> pd.DataFrame:
    if raw.empty:
        return pd.DataFrame()

    if isinstance(raw.columns, pd.MultiIndex):
        if field in raw.columns.get_level_values(0):
            frame = raw[field].copy()
        elif field in raw.columns.get_level_values(1):
            frame = raw.xs(field, axis=1, level=1).copy()
        else:
            return pd.DataFrame()
    else:
        if field not in raw.columns:
            return pd.DataFrame()
        frame = raw[[field]].copy()

    if isinstance(frame, pd.Series):
        frame = frame.to_frame()

    frame.columns = [str(col).upper() for col in frame.columns]
    frame.index = pd.to_datetime(frame.index)

    return frame.apply(pd.to_numeric, errors='coerce')


def download_market_data(tickers: list[str], start: str, end: str | None = None):
    tickers = clean_tickers(tickers)
    symbols = clean_tickers(tickers + BENCHMARKS)

    raw = yf.download(
        symbols,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        group_by='column',
        threads=True,
        repair=True,
    )

    close = _extract_field(raw, 'Close')
    volume = _extract_field(raw, 'Volume')

    if close.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.Series(dtype=float), pd.Series(dtype=float), tickers

    valid = [
        ticker for ticker in tickers
        if ticker in close.columns and close[ticker].dropna().shape[0] >= 20
    ]

    invalid = sorted(set(tickers) - set(valid))

    prices = close.reindex(columns=valid)
    prices = prices.ffill(limit=5).dropna(how='all')
    prices = prices.loc[:, prices.notna().sum() >= 20]

    invalid = sorted(set(invalid).union(set(valid) - set(prices.columns)))

    benchmarks = [benchmark for benchmark in BENCHMARKS if benchmark in close.columns]
    benchmark_prices = close.reindex(columns=benchmarks)
    benchmark_returns_frame = benchmark_prices.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)

    if benchmark_returns_frame.empty or benchmark_returns_frame.dropna(how='all').empty:
        benchmark_returns = pd.Series(0.0, index=prices.index, name='Benchmark')
        benchmark = pd.Series(1.0, index=prices.index, name='Benchmark')
    else:
        benchmark_returns = benchmark_returns_frame.mean(axis=1, skipna=True).dropna()
        common_index = prices.index.intersection(benchmark_returns.index)

        prices = prices.loc[common_index]
        benchmark_returns = benchmark_returns.loc[common_index]
        benchmark = (1 + benchmark_returns).cumprod()
        benchmark.name = 'Benchmark'

    volume = volume.reindex(columns=prices.columns)
    volume = volume.reindex(prices.index)

    prices.columns = prices.columns.astype(str)
    volume.columns = volume.columns.astype(str)

    return prices, volume, benchmark, benchmark_returns, invalid


async def _fetch_single_metadata(ticker: str, cadusd_rate: float) -> dict:
    ticker = str(ticker).upper()
    fallback_sector = DEFAULT_SECTOR_FALLBACK.get(ticker, 'Unclassified')

    try:
        obj = yf.Ticker(ticker)

        try:
            info = await asyncio.to_thread(obj.get_info)
        except Exception:
            try:
                info = await asyncio.to_thread(lambda: obj.info)
            except Exception:
                info = {}

        if not isinstance(info, dict):
            info = {}

        market_cap_raw = info.get('marketCap', np.nan)

        if isinstance(market_cap_raw, (int, float)) and not pd.isna(market_cap_raw) and market_cap_raw > 0:
            market_cap_cad = market_cap_raw if ticker.endswith('.TO') else market_cap_raw / cadusd_rate
        else:
            market_cap_cad = np.nan

        sector = info.get('sector') or fallback_sector
        industry = info.get('industry') or 'Unknown'

        if sector == 'Other':
            sector = fallback_sector

        return {
            'Ticker': ticker,
            'Sector': str(sector),
            'Industry': str(industry),
            'MarketCapCAD': market_cap_cad,
            'SmallCap': bool(market_cap_cad < SMALL_CAP_CAD) if not pd.isna(market_cap_cad) else False,
            'LargeCap': bool(market_cap_cad > LARGE_CAP_CAD) if not pd.isna(market_cap_cad) else False,
        }

    except Exception:
        return {
            'Ticker': ticker,
            'Sector': fallback_sector,
            'Industry': 'Unknown',
            'MarketCapCAD': np.nan,
            'SmallCap': False,
            'LargeCap': False,
        }


async def fetch_metadata(tickers: list[str], cadusd_rate: float) -> pd.DataFrame:
    tickers = [str(ticker).upper() for ticker in tickers]
    rows = await asyncio.gather(*[_fetch_single_metadata(ticker, cadusd_rate) for ticker in tickers])

    frame = pd.DataFrame(rows)

    for ticker in tickers:
        if ticker not in set(frame['Ticker']):
            frame.loc[len(frame)] = [ticker, 'Other', 'Unknown', np.nan, False, False]

    frame['Ticker'] = frame['Ticker'].astype(str)

    return frame.set_index('Ticker')


def build_metrics(
    prices: pd.DataFrame,
    volume: pd.DataFrame,
    benchmark_returns: pd.Series,
    metadata: pd.DataFrame,
) -> pd.DataFrame:
    if prices.empty:
        return pd.DataFrame()

    prices = prices.copy()
    volume = volume.copy()

    prices.columns = prices.columns.astype(str)
    volume.columns = volume.columns.astype(str)

    returns = prices.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan).dropna(how='all')

    rows: list[dict] = []

    for ticker in prices.columns:
        r = pd.to_numeric(returns[str(ticker)], errors='coerce').dropna()

        if r.empty:
            continue

        b = pd.to_numeric(benchmark_returns.reindex(r.index), errors='coerce').dropna()
        idx = r.index.intersection(b.index)

        if len(idx) >= 2:
            r_aligned = pd.Series(r.loc[idx], dtype=float)
            b_aligned = pd.Series(b.loc[idx], dtype=float)
        else:
            r_aligned = r
            b_aligned = pd.Series(dtype=float)

        std = float(r_aligned.std()) if len(r_aligned) >= 2 else np.nan

        if len(r_aligned) >= 2 and len(b_aligned) >= 2:
            cov = float(r_aligned.cov(b_aligned))
            variance_b = float(b_aligned.var())
            beta = cov / variance_b if variance_b > 0 else np.nan
            corr = float(r_aligned.corr(b_aligned)) if r_aligned.std() > 0 and b_aligned.std() > 0 else 0.0
            weekly = weekly_volatility(r_aligned)

            alpha = r_aligned.mean() - beta * b_aligned.mean() if not np.isnan(beta) else np.nan
            residuals = r_aligned - (alpha + beta * b_aligned) if not np.isnan(alpha) else pd.Series(dtype=float)
            idio = float(residuals.std()) if not residuals.empty else std
        else:
            cov = np.nan
            beta = 1.0
            corr = 0.0
            weekly = weekly_volatility(r_aligned)
            idio = std

        rows.append({
            'Ticker': str(ticker),
            'AvgVolume': average_volume(volume[str(ticker)]) if str(ticker) in volume.columns else np.nan,
            'DailyVolatility': std,
            'Covariance': cov,
            'Beta': beta,
            'Correlation': corr,
            'WeeklyVolatility': weekly,
            'IdiosyncraticVolatility': idio,
        })

    metrics = pd.DataFrame(rows)

    if metrics.empty:
        return pd.DataFrame()

    metadata = metadata.reset_index().copy() if not metadata.empty else pd.DataFrame()
    if metadata.empty:
        metadata = pd.DataFrame({'Ticker': metrics['Ticker']})

    metadata['Ticker'] = metadata['Ticker'].astype(str)
    joined = metrics.merge(metadata, on='Ticker', how='left')

    joined['Sector'] = joined.apply(
        lambda row: (
            DEFAULT_SECTOR_FALLBACK.get(str(row['Ticker']).upper(), 'Unclassified')
            if pd.isna(row['Sector']) or str(row['Sector']).strip() in ['', 'Other']
            else str(row['Sector'])
        ),
        axis=1,
    )

    joined['Industry'] = joined['Industry'].fillna('Unknown').replace('', 'Unknown')

    joined['Sector'] = joined['Sector'].fillna('Other').replace('', 'Other')
    joined['Industry'] = joined['Industry'].fillna('Unknown').replace('', 'Unknown')
    joined['AvgVolume'] = joined['AvgVolume'].fillna(0.0)
    joined['Correlation'] = joined['Correlation'].fillna(0.0)
    joined['Beta'] = joined['Beta'].fillna(1.0)

    joined = joined.sort_values(['Correlation', 'AvgVolume'], ascending=[False, False])

    return joined.set_index('Ticker')
