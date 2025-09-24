import yfinance as yf
import pandas as pd

class DataAcquisition:
    @staticmethod
    def get_stock_data(tickers, start, end, interval='1d'):
        if isinstance(tickers, str):
            tickers = [tickers]
        data = yf.download(tickers, start=start, end=end, interval=interval, group_by='ticker', auto_adjust=True)
        return data

    @staticmethod
    def get_options_chain(ticker, date=None):
        stock = yf.Ticker(ticker)
        if date:
            return stock.option_chain(date)
        else:
            return stock.options
