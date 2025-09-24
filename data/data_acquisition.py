import yfinance as yf
import pandas as pd

class DataAcquisition:
    @staticmethod
    def get_stock_data(ticker, start, end, interval='1d'):
        data = yf.download(ticker, start=start, end=end, interval=interval)
        return data

    @staticmethod
    def get_options_chain(ticker, date=None):
        stock = yf.Ticker(ticker)
        if date:
            return stock.option_chain(date)
        else:
            return stock.options
