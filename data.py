from __future__ import annotations

import asyncio

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

    recent = values.tail(60)
    return float(recent.mean()) if not recent.empty else float(values.mean())


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

    tickers_upper = [ticker.upper() for ticker in tickers]
    benchmarks_upper = [ticker.upper() for ticker in BENCHMARKS]

    valid = [
        ticker for ticker in tickers_upper
        if ticker in close.columns and close[ticker].dropna().shape[0] >= 20
    ]

    invalid = sorted(set(tickers_upper) - set(valid))

    benchmark_prices = close.reindex(columns=[b for b in benchmarks_upper if b in close.columns])
    benchmark_returns_frame = benchmark_prices.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)

    if benchmark_returns_frame.empty or benchmark_returns_frame.dropna(how='all').empty:
        benchmark_returns = pd.Series(dtype=float)
        benchmark = pd.Series(dtype=float)
    else:
        benchmark_returns = benchmark_returns_frame.mean(axis=1, skipna=True).dropna()
        benchmark = (1 + benchmark_returns).cumprod()
        benchmark.name = 'Benchmark'

    prices = close.reindex(columns=valid)
    prices = prices.dropna(how='all', axis=1)
    prices = prices.ffill(limit=5).dropna(how='all')

    if not benchmark_returns.empty:
        common_index = prices.index.intersection(benchmark_returns.index)
        prices = prices.loc[common_index]
        benchmark_returns = benchmark_returns.loc[common_index]
        benchmark = benchmark.loc[common_index]

    prices = prices.loc[:, prices.notna().sum() >= 20]
    invalid = sorted(set(invalid).union(set(valid) - set(prices.columns)))

    volume = volume.reindex(columns=prices.columns)
    volume = volume.reindex(prices.index)

    prices.columns = prices.columns.astype(str)
    volume.columns = volume.columns.astype(str)

    return prices, volume, benchmark, benchmark_returns, invalid


async def _fetch_single_metadata(ticker: str, cadusd_rate: float) -> dict:
    fallback = {
        'Ticker': str(ticker),
        'Sector': 'Other',
        'Industry': 'Unknown',
        'MarketCapCAD': np.nan,
        'SmallCap': False,
        'LargeCap': False,
    }

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

        sector = info.get('sector') or 'Other'
        industry = info.get('industry') or 'Unknown'

        return {
            'Ticker': str(ticker),
            'Sector': str(sector),
            'Industry': str(industry),
            'MarketCapCAD': market_cap_cad,
            'SmallCap': bool(market_cap_cad < SMALL_CAP_CAD) if not pd.isna(market_cap_cad) else False,
            'LargeCap': bool(market_cap_cad > LARGE_CAP_CAD) if not pd.isna(market_cap_cad) else False,
        }

    except Exception:
        return fallback


async def fetch_metadata(tickers: list[str], cadusd_rate: float) -> pd.DataFrame:
    tickers = [str(ticker).upper() for ticker in tickers]
    final_rows: dict[str, dict] = {}

    for _ in range(3):
        missing = [
            ticker for ticker in tickers
            if ticker not in final_rows
            or final_rows[ticker].get('Sector') in [None, '', 'Other']
            or final_rows[ticker].get('Industry') in [None, '', 'Unknown']
        ]

        if not missing:
            break

        rows = await asyncio.gather(*[_fetch_single_metadata(ticker, cadusd_rate) for ticker in missing])

        for row in rows:
            ticker = str(row['Ticker']).upper()

            if ticker not in final_rows:
                final_rows[ticker] = row
            elif row.get('Sector') != 'Other' or row.get('Industry') != 'Unknown':
                final_rows[ticker] = row

        await asyncio.sleep(0.5)

    for ticker in tickers:
        final_rows.setdefault(ticker, {
            'Ticker': ticker,
            'Sector': 'Other',
            'Industry': 'Unknown',
            'MarketCapCAD': np.nan,
            'SmallCap': False,
            'LargeCap': False,
        })

    frame = pd.DataFrame(final_rows.values())
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
        if ticker not in returns.columns:
            continue

        r = pd.to_numeric(returns[str(ticker)], errors='coerce').dropna()

        if benchmark_returns.empty:
            b = pd.Series(dtype=float)
        else:
            b = pd.to_numeric(benchmark_returns.reindex(r.index), errors='coerce').dropna()

        idx = r.index.intersection(b.index)
        r = pd.Series(r.loc[idx], dtype=float) if len(idx) else r
        b = pd.Series(b.loc[idx], dtype=float) if len(idx) else b

        if len(r) < 2:
            std = cov = beta = corr = weekly = idio = np.nan
        elif len(b) < 2:
            std = float(r.std())
            cov = np.nan
            beta = np.nan
            corr = 0.0
            weekly = weekly_volatility(r)
            idio = std
        else:
            std = float(r.std())
            cov = float(r.cov(b))
            variance_b = float(b.var())
            beta = cov / variance_b if variance_b > 0 else np.nan
            corr = float(r.corr(b)) if r.std() > 0 and b.std() > 0 else 0.0
            weekly = weekly_volatility(r)

            alpha = r.mean() - beta * b.mean() if not np.isnan(beta) else np.nan
            residuals = r - (alpha + beta * b) if not np.isnan(alpha) else pd.Series(dtype=float)
            idio = float(residuals.std()) if not residuals.empty else np.nan

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

    metrics['Ticker'] = metrics['Ticker'].astype(str)

    if metadata.empty:
        metadata = pd.DataFrame(index=metrics['Ticker'])
        metadata['Sector'] = 'Other'
        metadata['Industry'] = 'Unknown'
        metadata['MarketCapCAD'] = np.nan
        metadata['SmallCap'] = False
        metadata['LargeCap'] = False

    metadata = metadata.reset_index().copy()
    metadata['Ticker'] = metadata['Ticker'].astype(str)

    joined = metrics.merge(metadata, on='Ticker', how='left')

    joined['Sector'] = joined['Sector'].fillna('Other').replace('', 'Other')
    joined['Industry'] = joined['Industry'].fillna('Unknown').replace('', 'Unknown')
    joined['AvgVolume'] = joined['AvgVolume'].fillna(0.0)
    joined['Correlation'] = joined['Correlation'].fillna(0.0)
    joined['Beta'] = joined['Beta'].fillna(1.0)
    joined['DailyVolatility'] = joined['DailyVolatility'].fillna(joined['DailyVolatility'].median())
    joined['WeeklyVolatility'] = joined['WeeklyVolatility'].fillna(joined['WeeklyVolatility'].median())
    joined['IdiosyncraticVolatility'] = joined['IdiosyncraticVolatility'].fillna(joined['DailyVolatility'])

    joined = joined.sort_values(['Correlation', 'AvgVolume'], ascending=[False, False])

    return joined.set_index('Ticker')
