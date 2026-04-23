# MarketMeet

MarketMeet is a Streamlit web app that builds a CAD equity portfolio designed to track the average of the S&P 500 and S&P/TSX Composite while controlling volatility and tracking error.

## Features

- live market data from Yahoo Finance
- benchmark-aware portfolio optimization
- sector diversification logic
- portfolio weights, sector mix, and cumulative growth charts
- downloadable CSV output

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy

1. Push this folder to GitHub.
2. Create a new app in Streamlit Community Cloud.
3. Point the app to `app.py`.
4. Deploy.

## Notes

The original notebook included assignment-specific markdown, print statements, and notebook-only output blocks. This version removes that clutter and packages the core logic into a cleaner app structure.
