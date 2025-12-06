# FinanceDatabase Frontend (Streamlit)

An interactive UI to browse the FinanceDatabase locally, slice by filters, and run quick Monte Carlo simulations or toy "next move" signals using Yahoo Finance prices.

## Quickstart

```bash
cd examples/frontend
python -m venv .venv && source .venv/bin/activate   # optional, but recommended
pip install -e ../..                                # install financedatabase and dependencies
pip install -r requirements.txt                     # install Streamlit + Plotly
streamlit run app.py
```

## What You Get
- Asset browser for equities, ETFs, funds, cryptos, indices, and currencies with country/sector/category filters.
- CSV export for the current selection.
- Monte Carlo price path simulation (geometric Brownian motion) driven by `yfinance` history, plus a lightweight probability tilt/volatility readout.

## Notes
- Results are capped for display (default 400, configurable up to 2000) to keep the UI responsive.
- "Predictions" are intentionally simple momentum/volatility tilts for exploration, not trading advice. Use your own models for production.
