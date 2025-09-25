import sys
from pathlib import Path
import os

# Add the project root directory to the Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple

from data.data_acquisition import DataAcquisition
from strategies.volume_analysis import VolumeAnalysis
from strategies.enhanced_volume import EnhancedVolumeAnalysis
from position_sizing.volume_sizer import VolumeBasedPositionSizer
from utils.app_config import AppConfig

class VolumeStrategyBacktest:
    def __init__(self, initial_capital: float = 100000):
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.positions = {}
        self.trades = []
        self.va = VolumeAnalysis()
        self.eva = EnhancedVolumeAnalysis()
        
        # Load config from project root
        config_path = Path(__file__).resolve().parent.parent / "config.yaml"
        self.config = AppConfig.load_from_yaml(config_path)
        self.sizer = VolumeBasedPositionSizer(self.config)
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate trading signals based on volume analysis"""
        signals = pd.DataFrame(index=data.index)
        signals['position'] = 0  # 1 for long, -1 for short, 0 for neutral
        
        # Calculate volume indicators
        klinger = self.eva.calculate_klinger_oscillator(data)
        force_idx = self.eva.volume_force_index(data)
        
        # Calculate VWAP
        daily_pv = data['Close'] * data['Volume']
        cumulative_pv = daily_pv.cumsum()
        cumulative_volume = data['Volume'].cumsum()
        vwap = cumulative_pv / cumulative_volume
        
        # Calculate volume trend using rolling averages
        volume_sma = data['Volume'].rolling(window=20).mean()
        volume_trend = pd.Series(index=data.index, data=0)
        volume_trend[data['Volume'] > volume_sma * 1.5] = 1
        volume_trend[data['Volume'] < volume_sma * 0.5] = -1
        
        # Support/Resistance levels - recalculated each day
        for i in range(len(data)):
            if i < 20:  # Need enough data for calculations
                continue
                
            historical = data.iloc[:i+1]
            levels = self.eva.volume_based_support_resistance(historical)
            current_price = data['Close'].iloc[i]
            
            # Generate signal based on combined indicators
            price_above_vwap = data['Close'].iloc[i] > vwap.iloc[i]
            volume_signal = (
                1 if klinger.iloc[i] > 0 and force_idx.iloc[i] > 0 and price_above_vwap and volume_trend.iloc[i] > 0
                else -1 if klinger.iloc[i] < 0 and force_idx.iloc[i] < 0 and not price_above_vwap and volume_trend.iloc[i] < 0
                else 0
            )
            
            # Confirm signal with support/resistance
            if volume_signal > 0 and current_price <= levels['support'] * 1.02:  # Within 2% of support
                signals.iloc[i] = 1  # Long signal
            elif volume_signal < 0 and current_price >= levels['resistance'] * 0.98:  # Within 2% of resistance
                signals.iloc[i] = -1  # Short signal
        
        return signals
    
    def calculate_position_size(self, data: pd.DataFrame, signal: int) -> int:
        """Calculate position size based on volume analysis"""
        if signal == 0:
            return 0
            
        size, metrics = self.sizer.calculate_position_size(
            data,
            self.capital,
            signal,  # 1 for long, -1 for short
            risk_percent=0.02  # 2% risk per trade
        )
        
        return size
    
    def run_backtest(self, data: pd.DataFrame) -> Dict:
        """Run backtest on the given data"""
        signals = self.generate_signals(data)
        equity_curve = []
        daily_returns = []
        
        for i in range(len(data)):
            if i < 20:  # Skip initial period needed for calculations
                equity_curve.append(self.capital)
                daily_returns.append(0)
                continue
            
            current_price = data['Close'].iloc[i]
            current_signal = signals.iloc[i]
            
            # Check for exit signals for existing positions
            for symbol, pos in list(self.positions.items()):
                entry_price = pos['entry_price']
                quantity = pos['quantity']
                position_type = pos['type']
                
                # Exit logic based on signals and support/resistance
                if (position_type == 'long' and current_signal < 0) or \
                   (position_type == 'short' and current_signal > 0):
                    # Calculate P&L
                    if position_type == 'long':
                        pnl = (current_price - entry_price) * quantity
                    else:
                        pnl = (entry_price - current_price) * quantity
                    
                    self.capital += pnl
                    del self.positions[symbol]
                    
                    # Record trade
                    self.trades.append({
                        'entry_date': pos['entry_date'],
                        'exit_date': data.index[i],
                        'type': position_type,
                        'entry_price': entry_price,
                        'exit_price': current_price,
                        'quantity': quantity,
                        'pnl': pnl
                    })
            
            # Enter new positions based on signals
            if current_signal != 0 and len(self.positions) == 0:
                position_size = self.calculate_position_size(
                    data.iloc[:i+1],
                    current_signal
                )
                
                if position_size > 0:
                    position_cost = position_size * current_price
                    if position_cost <= self.capital:
                        self.positions['STOCK'] = {
                            'type': 'long' if current_signal > 0 else 'short',
                            'quantity': position_size,
                            'entry_price': current_price,
                            'entry_date': data.index[i]
                        }
                        self.capital -= position_cost
            
            # Calculate daily equity
            total_equity = self.capital
            for pos in self.positions.values():
                position_value = pos['quantity'] * current_price
                if pos['type'] == 'long':
                    total_equity += position_value
                else:
                    total_equity += 2 * self.capital - position_value  # For short positions
            
            equity_curve.append(total_equity)
            daily_return = (total_equity - equity_curve[-2]) / equity_curve[-2] if i > 0 else 0
            daily_returns.append(daily_return)
        
        # Calculate performance metrics
        total_return = (equity_curve[-1] - self.initial_capital) / self.initial_capital
        daily_returns_series = pd.Series(daily_returns)
        sharpe_ratio = np.sqrt(252) * daily_returns_series.mean() / daily_returns_series.std()
        max_drawdown = self.calculate_max_drawdown(equity_curve)
        
        return {
            'equity_curve': equity_curve,
            'trades': self.trades,
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'daily_returns': daily_returns
        }
    
    @staticmethod
    def calculate_max_drawdown(equity_curve: List[float]) -> float:
        """Calculate the maximum drawdown from peak equity"""
        peak = equity_curve[0]
        max_drawdown = 0
        
        for value in equity_curve:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak
            max_drawdown = max(max_drawdown, drawdown)
        
        return max_drawdown

def run_volume_strategy_backtest():
    """Run backtesting for volume-based trading strategy"""
    print("\n=== Starting Volume Strategy Backtest ===\n")
    
    # Ensure config file exists
    config_path = Path(project_root) / "config.yaml"
    if not config_path.exists():
        print("Error: config.yaml not found. Please create the configuration file first.")
        return
    
    # Create output directory with absolute path
    output_dir = project_root / "backtest_results"
    output_dir.mkdir(exist_ok=True)
    print(f"Created output directory: {output_dir}")
    
    # Create test directory with timestamp
    test_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    test_dir = output_dir / f"backtest_{test_time}"
    test_dir.mkdir(exist_ok=True)
    print(f"Created test directory: {test_dir}")
    
    # Create test directory with timestamp
    test_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    test_dir = output_dir / f"backtest_{test_time}"
    test_dir.mkdir(exist_ok=True)
    
    # Load test data
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365 * 2)  # 2 years of data
    tickers = ['AAPL', 'MSFT', 'GOOGL']
    
    backtest_results = {}
    
    with open(test_dir / "backtest_report.txt", "w", encoding='utf-8') as f:
        f.write("=== Volume Strategy Backtest Report ===\n\n")
        f.write(f"Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}\n\n")
        
        for ticker in tickers:
            print(f"\nBacktesting {ticker}...")
            f.write(f"\n{ticker} Results:\n")
            f.write("-" * (len(ticker) + 9) + "\n")
            
            try:
                # Get historical data
                data = DataAcquisition.get_stock_data(
                    [ticker],
                    start_date.strftime('%Y-%m-%d'),
                    end_date.strftime('%Y-%m-%d')
                )
                
                if isinstance(data.columns, pd.MultiIndex):
                    data = data[ticker]
                
                # Run backtest
                backtest = VolumeStrategyBacktest(initial_capital=100000)
                results = backtest.run_backtest(data)
                
                # Save results
                backtest_results[ticker] = results
                
                # Write performance metrics
                f.write(f"\nPerformance Metrics:\n")
                f.write(f"Total Return: {results['total_return']:.2%}\n")
                f.write(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}\n")
                f.write(f"Max Drawdown: {results['max_drawdown']:.2%}\n")
                
                # Trade statistics
                trades_df = pd.DataFrame(results['trades'])
                if not trades_df.empty:
                    win_rate = len(trades_df[trades_df['pnl'] > 0]) / len(trades_df)
                    avg_win = trades_df[trades_df['pnl'] > 0]['pnl'].mean() if len(trades_df[trades_df['pnl'] > 0]) > 0 else 0
                    avg_loss = trades_df[trades_df['pnl'] < 0]['pnl'].mean() if len(trades_df[trades_df['pnl'] < 0]) > 0 else 0
                    
                    f.write(f"\nTrade Statistics:\n")
                    f.write(f"Number of Trades: {len(trades_df)}\n")
                    f.write(f"Win Rate: {win_rate:.2%}\n")
                    f.write(f"Average Win: ${avg_win:.2f}\n")
                    f.write(f"Average Loss: ${avg_loss:.2f}\n")
                    
                    # Save trade log
                    trades_df.to_csv(test_dir / f"{ticker}_trades.csv", index=False)
                    print(f"Saved trade log: {test_dir / f'{ticker}_trades.csv'}")
                
                # Save equity curve
                pd.DataFrame({
                    'Date': data.index,
                    'Equity': results['equity_curve']
                }).to_csv(test_dir / f"{ticker}_equity_curve.csv", index=False)
                print(f"Saved equity curve: {test_dir / f'{ticker}_equity_curve.csv'}")
                
                print(f"✓ {ticker} backtest completed successfully")
                
            except Exception as e:
                error_msg = f"✗ Error in {ticker} backtest: {str(e)}"
                print(error_msg)
                f.write(f"\n{error_msg}\n")
                continue
        
        # Comparative analysis
        if backtest_results:
            f.write("\nComparative Analysis:\n")
            f.write("-------------------\n")
            
            summary_data = {
                ticker: {
                    'Total Return': results['total_return'],
                    'Sharpe Ratio': results['sharpe_ratio'],
                    'Max Drawdown': results['max_drawdown'],
                    'Trades': len(results['trades'])
                }
                for ticker, results in backtest_results.items()
            }
            
            summary_df = pd.DataFrame(summary_data).round(4)
            f.write("\nStrategy Performance Summary:\n")
            f.write(summary_df.to_string())
            
            # Save summary to CSV
            summary_df.to_csv(test_dir / "strategy_summary.csv")
    
    print("\n=== Backtest Complete ===")
    print(f"Results saved in: {test_dir}")

if __name__ == "__main__":
    import traceback
    try:
        run_volume_strategy_backtest()
    except Exception as e:
        print("\nError during backtesting:")
        print(str(e))
        print("\nTraceback:")
        traceback.print_exc()
        sys.exit(1)