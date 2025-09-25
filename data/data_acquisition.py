import yfinance as yf
import pandas as pd
import backtrader as bt

class DataAcquisition:
    @staticmethod
    def get_stock_data(tickers, start, end, interval='1d'):
        """Get stock data and convert to backtrader-compatible format"""
        if isinstance(tickers, str):
            tickers = [tickers]
            
        print("\nFetching data for tickers:", tickers)
        data_feeds = {}
        for ticker in tickers:
            # Download data using yfinance
            df = yf.download(ticker, start=start, end=end, interval=interval, auto_adjust=True)
            
            # Ensure column names are correct format for backtrader
            if isinstance(df.columns, pd.MultiIndex):
                # Select the data for this ticker and flatten the column names
                ticker_data = pd.DataFrame()
                ticker_data['open'] = df[(ticker, 'Open')] if (ticker, 'Open') in df.columns else df['Open']
                ticker_data['high'] = df[(ticker, 'High')] if (ticker, 'High') in df.columns else df['High']
                ticker_data['low'] = df[(ticker, 'Low')] if (ticker, 'Low') in df.columns else df['Low']
                ticker_data['close'] = df[(ticker, 'Close')] if (ticker, 'Close') in df.columns else df['Close']
                ticker_data['volume'] = df[(ticker, 'Volume')] if (ticker, 'Volume') in df.columns else df['Volume']
                ticker_data['openinterest'] = 0  # Not available from Yahoo
            else:
                # Rename columns to lowercase for backtrader
                ticker_data = df.copy()
                ticker_data.columns = ticker_data.columns.str.lower()
                if 'openinterest' not in ticker_data.columns:
                    ticker_data['openinterest'] = 0
            
            print(f"\nData shape for {ticker}: {ticker_data.shape}")
            print(f"First few rows of {ticker} data:")
            print(ticker_data.head())
            
            # Convert to backtrader format
            data_feeds[ticker] = bt.feeds.PandasData(
                dataname=ticker_data,
                datetime=None,  # Use df index
                open='open',
                high='high',
                low='low',
                close='close',
                volume='volume',
                openinterest='openinterest'
            )
            
        return data_feeds

    @staticmethod
    def get_options_chain(ticker, date=None):
        stock = yf.Ticker(ticker)
        if date:
            return stock.option_chain(date)
        else:
            return stock.options
