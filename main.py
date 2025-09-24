from data.data_acquisition import DataAcquisition
from strategies.ta_strategy import TALibStrategy
from strategies.multi_asset_strategy import MultiAssetStrategy
from backtest.backtest_engine import BacktestEngine
from portfolio.risk_management import RiskManager

from utils.logger import get_logger
from utils.performance import analyze_performance, plot_equity_curve
import pandas as pd
import backtrader as bt



logger = get_logger(__name__)

def main():
    # Multi-asset example
    from utils.app_config import AppConfig
    # Load all configuration at the start
    config = AppConfig.load_from_yaml()
    tickers = config.tickers
    start_date = config.start_date
    end_date = config.end_date
    initial_cash = config.initial_cash
    risk_free_rate = config.risk_free_rate
    data = DataAcquisition.get_stock_data(tickers, start_date, end_date)
    logger.info(f'Downloaded data for {tickers}')
    print('DEBUG: data.columns:', data.columns)
    # Use SMA window from config if present, else default to 5
    sma_window = getattr(config, 'sma_window', 5)
    if hasattr(config, 'sma_window'):
        sma_window = config.sma_window
    else:
        try:
            import yaml
            with open('config.yaml', 'r') as f:
                ydata = yaml.safe_load(f)
            sma_window = ydata.get('sma_window', 5)
        except Exception:
            sma_window = 5
    # Generate signals for each asset
    signals = {}
    explanations = {}
    for ticker in tickers:
        # Robustly extract per-ticker data from MultiIndex or single-index DataFrame
        if isinstance(data.columns, pd.MultiIndex):
            ticker_cols = [col for col in data.columns if (len(col) > 1 and col[0] == ticker)]
            if ticker_cols:
                asset_data = data[ticker_cols].copy()
                asset_data.columns = [col[1] for col in ticker_cols]  # Flatten to ['Open', 'High', ...]
            else:
                logger.warning(f'No columns found for ticker {ticker}')
                continue
        else:
            asset_data = data
        strategy = TALibStrategy()
        # After generating sig for each ticker
        sig = strategy.generate_signals(asset_data, sma_window=sma_window)
        sig = sig.reindex(asset_data.index)  # Align signal index to data index
        signals[ticker] = sig
        # Generate explanations for signals
        if hasattr(strategy, 'explain_signals'):
            asset_data['SMA'] = asset_data['Close'].rolling(sma_window).mean()
            explanations[ticker] = strategy.explain_signals(asset_data, sig)
            #for exp in explanations[ticker]:
                #logger.info(f"{ticker} {exp['date']}: Signal={exp['signal']}, Close={exp['close']:.2f}, SMA={exp['sma']:.2f}, Reason={exp['reason']}")
    logger.info('Generated trading signals for all assets')

    # Prepare price data for risk manager (Close prices for all assets)
    if isinstance(data.columns, pd.MultiIndex):
        close_prices = pd.concat([
            data[(ticker, 'Close')] for ticker in tickers
        ], axis=1)
    else:
        close_prices = pd.concat([
            data['Close'] for _ in tickers
        ], axis=1)
    close_prices.columns = tickers

    # Risk management
    risk_manager = RiskManager(close_prices, risk_free_rate=risk_free_rate)
    weights = risk_manager.optimize_portfolio()
    logger.info(f'Optimized portfolio weights: {weights}')



    # Prepare Backtrader data feeds
    data_dict = {}
    for ticker in tickers:
        if isinstance(data.columns, pd.MultiIndex):
            ticker_cols = [col for col in data.columns if (len(col) > 1 and col[0] == ticker)]
            if ticker_cols:
                df = data[ticker_cols].copy()
                df.columns = [col[1] for col in ticker_cols]  # Flatten to ['Open', 'High', ...]
            else:
                logger.warning(f'No columns found for ticker {ticker} for Backtrader feed')
                continue
        else:
            df = data
        bt_df = pd.DataFrame({
            'open': df['Open'],
            'high': df['High'],
            'low': df['Low'],
            'close': df['Close'],
            'volume': df['Volume'],
        }, index=df.index)
        datafeed = bt.feeds.PandasData(dataname=bt_df)
        data_dict[ticker] = datafeed

    # Run backtest
    # Use MultiAssetStrategy directly
    backtest_engine = BacktestEngine(MultiAssetStrategy, data_dict, signals, weights, cash=initial_cash)
    results = backtest_engine.run()
    logger.info('Backtest complete')

    # Performance analysis (simple version)
    strat = results[0]
    # Try to extract portfolio value history from Backtrader
    if hasattr(strat.broker, '_value_history'):
        portfolio_values = pd.Series(strat.broker._value_history, index=bt_df.index[:len(strat.broker._value_history)])
        metrics, drawdown = analyze_performance(portfolio_values)
        logger.info(f'Performance metrics: {metrics}')
        #plot_equity_curve(portfolio_values, drawdown)

    else:
        logger.warning('Portfolio value history not available for analysis.')

    # Log account status info at the end using Backtrader's built-in features
    final_value = strat.broker.getvalue() if hasattr(strat, 'broker') else None
    cash = strat.broker.get_cash() if hasattr(strat, 'broker') else None
    positions = {}
    unrealized = {}
    realized = 0.0
    for data in strat.datas:
        pos = strat.getposition(data)
        if pos.size != 0:
            positions[data._name] = pos.size
            # Calculate unrealized P&L for each open position
            unrealized[data._name] = (data.close[0] - pos.price) * pos.size
            # Calculate realized P&L from closed trades (if needed, accumulate from trade notifications)
    logger.info(f'Account Status: Portfolio Value = {final_value}, Cash = {cash}, Open Positions = {positions}')
    logger.info(f'Unrealized P&L: {unrealized}')

if __name__ == '__main__':
    main()
