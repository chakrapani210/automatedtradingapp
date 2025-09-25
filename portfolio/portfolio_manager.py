import pandas as pd
import numpy as np
from typing import Dict, List, Any
from strategies.long_term_strategy import LongTermValueStrategy
from strategies.short_term_strategy import ShortTermTechnicalStrategy
from strategies.options_strategy import OptionsStrategy

class PortfolioManager:
    def __init__(self, config: Any):
        self.config = config
        self.strategies = {
            'long_term': LongTermValueStrategy(),
            'short_term': ShortTermTechnicalStrategy(),
            'options': OptionsStrategy()
        }
        self.positions = {}
        self.portfolio_value_history = []
        self.last_rebalance = None

    def calculate_portfolio_weights(self) -> Dict[str, float]:
        """Get current portfolio allocation weights"""
        return {
            'long_term': self.config.portfolio_allocation['long_term'],
            'short_term': self.config.portfolio_allocation['short_term'],
            'options': self.config.portfolio_allocation['options']
        }

    def calculate_var(self, returns: pd.Series, confidence: float = 0.95) -> float:
        """Calculate Value at Risk"""
        return np.percentile(returns, (1 - confidence) * 100)

    def check_correlation(self, positions: Dict[str, pd.Series]) -> bool:
        """Check if positions are within correlation limits"""
        if len(positions) < 2:
            return True
            
        returns = pd.DataFrame(positions)
        corr_matrix = returns.corr()
        
        # Check if any correlation exceeds threshold
        threshold = self.config.risk_management['correlation_threshold']
        high_corr = (corr_matrix.abs() > threshold).sum().sum()
        
        return high_corr <= len(positions)  # Allow some correlation

    def check_portfolio_risk(self, positions: Dict[str, pd.Series]) -> bool:
        """Check if portfolio risk is within limits"""
        if not positions:
            return True
            
        # Calculate portfolio returns
        portfolio_returns = pd.DataFrame(positions).sum(axis=1)
        
        # Check VaR
        var = self.calculate_var(portfolio_returns)
        var_limit = self.config.risk_management['portfolio_var_limit']
        if abs(var) > var_limit:
            return False
            
        # Check drawdown
        max_dd = self.config.risk_management['max_portfolio_drawdown']
        current_dd = (portfolio_returns.cummax() - portfolio_returns).max()
        if current_dd > max_dd:
            return False
            
        return True

    def execute_trades(self, date: pd.Timestamp, data: Dict[str, pd.DataFrame]) -> None:
        """Execute trades for each strategy"""
        portfolio_value = sum(self.positions.values()) if self.positions else self.config.initial_cash
        weights = self.calculate_portfolio_weights()
        
        for strategy_name, strategy in self.strategies.items():
            # Check if rebalancing is needed
            if strategy.should_rebalance(date, self.last_rebalance, self.config):
                allocation = portfolio_value * weights[strategy_name]
                
                for ticker, ticker_data in data.items():
                    # Generate signals
                    signals = strategy.generate_signals(ticker_data, self.config)
                    
                    if signals.iloc[-1] != 0:  # If we have a signal
                        # Calculate position size
                        size = strategy.calculate_position_size(ticker_data, allocation, self.config)
                        
                        # Get stop loss
                        entry_price = ticker_data['Close'].iloc[-1]
                        stop_loss = strategy.get_stop_loss(
                            ticker_data,
                            entry_price,
                            'long' if signals.iloc[-1] > 0 else 'short',
                            self.config
                        )
                        
                        # Update positions
                        self.positions[f"{ticker}_{strategy_name}"] = size * signals.iloc[-1]
                        
                        print(f"Executed trade: {ticker} {strategy_name}")
                        print(f"Size: {size}, Stop Loss: {stop_loss}")
                
                self.last_rebalance = date

    def update_portfolio_value(self, date: pd.Timestamp, data: Dict[str, pd.DataFrame]) -> None:
        """Update portfolio value and check risk limits"""
        total_value = 0
        
        for position_id, size in self.positions.items():
            ticker = position_id.split('_')[0]
            if ticker in data:
                price = data[ticker]['Close'].iloc[-1]
                total_value += size * price
        
        self.portfolio_value_history.append({
            'date': date,
            'value': total_value
        })
        
        # Check if we need to reduce risk
        if not self.check_portfolio_risk(self.positions):
            print("Portfolio risk limits exceeded - reducing positions")
            self.reduce_risk()

    def reduce_risk(self) -> None:
        """Reduce portfolio risk by scaling down positions"""
        scale_factor = 0.75  # Reduce positions by 25%
        for position_id in self.positions:
            self.positions[position_id] *= scale_factor

    def run(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Run the portfolio manager"""
        dates = pd.DatetimeIndex(next(iter(data.values())).index)
        
        # Initialize a Series to store portfolio values
        portfolio_values = pd.Series(index=dates, dtype='float64')
        
        for date in dates:
            print(f"\nProcessing date: {date}")
            
            # Execute trades for each strategy
            self.execute_trades(date, data)
            
            # Update portfolio value and check risk
            self.update_portfolio_value(date, data)
            portfolio_values[date] = self.portfolio_value_history[-1] if self.portfolio_value_history else self.config.initial_cash
        
        # Return portfolio values as a Series
        return portfolio_values