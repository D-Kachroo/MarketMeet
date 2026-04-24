DEFAULT_TICKERS = [
    'AAPL', 'ABBV', 'AMZN', 'AXP', 'BK', 'CMCSA', 'COST', 'CSCO', 'DUOL',
    'GOOG', 'GM', 'LOW', 'ORCL', 'PEP', 'SHOP', 'SLB', 'SPG', 'TD',
    'SU.TO', 'RY.TO', 'TD.TO', 'SHOP.TO', 'FTG.TO', 'AIM.TO', 'SAP.TO',
    'AW.TO', 'EXE.TO', 'AUST'
]

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

SECTOR_FALLBACKS = {
    'AAPL': 'Technology',
    'ABBV': 'Healthcare',
    'AMZN': 'Consumer Cyclical',
    'AXP': 'Financial Services',
    'BK': 'Financial Services',
    'CMCSA': 'Communication Services',
    'COST': 'Consumer Defensive',
    'CSCO': 'Technology',
    'DUOL': 'Technology',
    'GOOG': 'Communication Services',
    'GM': 'Consumer Cyclical',
    'LOW': 'Consumer Cyclical',
    'ORCL': 'Technology',
    'PEP': 'Consumer Defensive',
    'SHOP': 'Technology',
    'SHOP.TO': 'Technology',
    'SLB': 'Energy',
    'SPG': 'Real Estate',
    'TD': 'Financial Services',
    'TD.TO': 'Financial Services',
    'SU.TO': 'Energy',
    'RY.TO': 'Financial Services',
    'FTG.TO': 'Industrials',
    'AIM.TO': 'Basic Materials',
    'SAP.TO': 'Consumer Defensive',
    'AW.TO': 'Consumer Cyclical',
    'EXE.TO': 'Healthcare',
    'AUST': 'Basic Materials',
}
