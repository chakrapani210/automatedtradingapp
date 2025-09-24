# Automated Trading App

A fully managed, modular automated trading application supporting stocks and options, with backtesting, risk management, and live trading.

## Features
- Data acquisition (yfinance)
- Strategy development (TA-Lib, pandas, custom logic)
- Backtesting (Backtrader, PyPortfolioOpt, options pricing with Optopsy/QuantLib)
- Risk management (PyPortfolioOpt, custom rules)
- Live trading (Backtrader live broker API integration)

## Setup
1. Install dependencies:
   ```
pip install -r requirements.txt
   ```
2. Run the app:
   ```
python main.py
   ```

## Structure
- `data/`: Data acquisition modules
- `strategies/`: Trading strategies
- `backtest/`: Backtesting engine
- `portfolio/`: Risk/account management
- `live/`: Live trading integration
- `utils/`: Utilities
