from data.data_acquisition import DataAcquisition
from strategies.ta_strategy import TALibStrategy
from backtest.backtest_engine import BacktestEngine
from portfolio.risk_management import RiskManager
from portfolio.account_manager import AccountManager
from utils.logger import get_logger

logger = get_logger(__name__)

def main():
    # Example: Download data
    data = DataAcquisition.get_stock_data('AAPL', '2022-01-01', '2023-01-01')
    logger.info('Downloaded data for AAPL')

    # Example: Generate signals
    strategy = TALibStrategy()
    signals = strategy.generate_signals(data)
    logger.info('Generated trading signals')

    # Example: Risk management
    risk_manager = RiskManager(data[['Close']])
    weights = risk_manager.optimize_portfolio()
    logger.info(f'Optimized portfolio weights: {weights}')

    # Example: Backtesting (stub, needs Backtrader datafeed)
    # backtest_engine = BacktestEngine(TALibStrategy, data)
    # results = backtest_engine.run()
    # logger.info('Backtest complete')

if __name__ == '__main__':
    main()
