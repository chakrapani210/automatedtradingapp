import logging
import os
from datetime import datetime
from typing import Optional
import pandas as pd

class TradingLogger:
    """Enhanced logging system for the automated trading application"""
    
    CATEGORIES = {
        'STRATEGY': 25,  # Between INFO and WARNING
        'TRADE': 21,     # Between INFO and WARNING
        'SIGNAL': 22,    # Between INFO and WARNING
        'POSITION': 23,  # Between INFO and WARNING
        'RISK': 24,     # Between INFO and WARNING
        'PERFORMANCE': 26, # Between INFO and WARNING
        'DATA': 27      # Between INFO and WARNING
    }
    
    def __init__(self, name: str, log_dir: str = "logs", console_level: str = "INFO"):
        """
        Initialize the logger with custom levels and multiple handlers
        
        Args:
            name: Name of the logger/strategy
            log_dir: Directory to store log files
            console_level: Minimum level for console output
        """
        # Register custom levels
        for category, level in self.CATEGORIES.items():
            logging.addLevelName(level, category)
            
        # Create logs directory structure
        self.base_dir = log_dir
        self.strategy_dir = os.path.join(log_dir, name)
        self.create_log_directories()
        
        # Initialize logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)
        
        # Remove existing handlers if any
        self.logger.handlers = []
        
        # Create handlers
        self.setup_handlers(name, console_level)
        
    def create_log_directories(self):
        """Create necessary log directories"""
        os.makedirs(self.base_dir, exist_ok=True)
        os.makedirs(self.strategy_dir, exist_ok=True)
        os.makedirs(os.path.join(self.strategy_dir, 'trades'), exist_ok=True)
        os.makedirs(os.path.join(self.strategy_dir, 'performance'), exist_ok=True)
        
    def setup_handlers(self, name: str, console_level: str):
        """Set up file and console handlers with appropriate formatting"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Console handler
        console = logging.StreamHandler()
        console.setLevel(getattr(logging, console_level))
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console.setFormatter(console_formatter)
        
        # Main file handler
        main_log = os.path.join(self.strategy_dir, f'{timestamp}_main.log')
        file_handler = logging.FileHandler(main_log)
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        
        # Trade-specific file handler
        trade_log = os.path.join(self.strategy_dir, 'trades', f'{timestamp}_trades.log')
        trade_handler = logging.FileHandler(trade_log)
        trade_handler.setLevel(self.CATEGORIES['TRADE'])
        trade_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        )
        trade_handler.setFormatter(trade_formatter)
        
        # Performance-specific file handler
        perf_log = os.path.join(self.strategy_dir, 'performance', f'{timestamp}_performance.log')
        perf_handler = logging.FileHandler(perf_log)
        perf_handler.setLevel(self.CATEGORIES['PERFORMANCE'])
        perf_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        )
        perf_handler.setFormatter(perf_formatter)
        
        # Add all handlers
        self.logger.addHandler(console)
        self.logger.addHandler(file_handler)
        self.logger.addHandler(trade_handler)
        self.logger.addHandler(perf_handler)
        
    def strategy(self, msg: str):
        """Log strategy-related messages"""
        self.logger.log(self.CATEGORIES['STRATEGY'], msg)
        
    def trade(self, ticker: str, action: str, quantity: int, price: float, value: float,
             pnl: Optional[float] = None):
        """Log trade execution details"""
        msg = (f"TRADE - {ticker}: {action} {quantity} @ ${price:.2f} "
               f"(Value: ${value:.2f})")
        if pnl is not None:
            msg += f" P&L: ${pnl:.2f}"
        self.logger.log(self.CATEGORIES['TRADE'], msg)
        
    def signal(self, ticker: str, signal_type: str, value: float, params: dict = None):
        """Log trading signals"""
        msg = f"SIGNAL - {ticker}: {signal_type} = {value}"
        if params:
            msg += f" | Parameters: {params}"
        self.logger.log(self.CATEGORIES['SIGNAL'], msg)
        
    def position(self, ticker: str, quantity: int, avg_price: float, 
                current_price: float, market_value: float, unrealized_pnl: float):
        """Log position updates"""
        msg = (f"POSITION - {ticker}: {quantity} shares @ ${avg_price:.2f} "
               f"(Current: ${current_price:.2f}, Value: ${market_value:.2f}, "
               f"Unrealized P&L: ${unrealized_pnl:.2f})")
        self.logger.log(self.CATEGORIES['POSITION'], msg)
        
    def risk(self, msg: str):
        """Log risk management messages"""
        self.logger.log(self.CATEGORIES['RISK'], msg)
        
    def performance(self, metrics: dict):
        """Log performance metrics"""
        msg = "PERFORMANCE UPDATE:\n"
        for key, value in metrics.items():
            if isinstance(value, float):
                msg += f"  {key}: {value:.4f}\n"
            else:
                msg += f"  {key}: {value}\n"
        self.logger.log(self.CATEGORIES['PERFORMANCE'], msg)
        
    def data(self, msg: str):
        """Log data-related messages"""
        self.logger.log(self.CATEGORIES['DATA'], msg)
        
    def debug(self, msg: str):
        """Log debug messages"""
        self.logger.debug(msg)
        
    def info(self, msg: str):
        """Log general information messages"""
        self.logger.info(msg)
        
    def warning(self, msg: str):
        """Log warning messages"""
        self.logger.warning(msg)
        
    def error(self, msg: str):
        """Log error messages"""
        self.logger.error(msg)
        
    def critical(self, msg: str):
        """Log critical messages"""
        self.logger.critical(msg)