DEFAULT_TICKERS = [
    'AAPL', 'MSFT', 'NVDA', 'AMZN', 'GOOGL', 'META', 'JPM', 'V', 'XOM', 'COST',
    'UNH', 'HD', 'PG', 'KO', 'PEP', 'CAT', 'NEE', 'LIN', 'AMT', 'RY.TO',
    'TD.TO', 'BNS.TO', 'ENB.TO', 'CNQ.TO', 'CNR.TO', 'CP.TO', 'SHOP.TO',
    'BAM.TO', 'MFC.TO', 'FTS.TO', 'MCD'
]

DEFAULT_SECTOR_FALLBACK = {
    'AAPL': 'Technology',
    'MSFT': 'Technology',
    'NVDA': 'Technology',
    'AMZN': 'Consumer Cyclical',
    'GOOGL': 'Communication Services',
    'META': 'Communication Services',
    'JPM': 'Financial Services',
    'V': 'Financial Services',
    'XOM': 'Energy',
    'COST': 'Consumer Defensive',
    'UNH': 'Healthcare',
    'HD': 'Consumer Cyclical',
    'PG': 'Consumer Defensive',
    'KO': 'Consumer Defensive',
    'PEP': 'Consumer Defensive',
    'CAT': 'Industrials',
    'NEE': 'Utilities',
    'LIN': 'Basic Materials',
    'AMT': 'Real Estate',
    'RY.TO': 'Financial Services',
    'TD.TO': 'Financial Services',
    'BNS.TO': 'Financial Services',
    'ENB.TO': 'Energy',
    'CNQ.TO': 'Energy',
    'CNR.TO': 'Industrials',
    'CP.TO': 'Industrials',
    'SHOP.TO': 'Technology',
    'BAM.TO': 'Financial Services',
    'MFC.TO': 'Financial Services',
    'FTS.TO': 'Utilities',
}

BENCHMARKS = ['^GSPC', '^GSPTSE']
PORTFOLIO_VALUE_CAD = 1_000_000
MIN_WEIGHT = 0.02
MAX_WEIGHT = 0.15
MIN_AVG_VOLUME = 5_000
MIN_CORRELATION = 0.20
SMALL_CAP_CAD = 2_000_000_000
LARGE_CAP_CAD = 10_000_000_000
DEFAULT_START = '2024-01-01'
FX_FALLBACK = 0.73
