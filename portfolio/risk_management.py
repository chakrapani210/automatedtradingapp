from pypfopt import EfficientFrontier, risk_models, expected_returns
import pandas as pd

class RiskManager:
    def __init__(self, price_data):
        self.price_data = price_data

    def optimize_portfolio(self):
        mu = expected_returns.mean_historical_return(self.price_data)
        S = risk_models.sample_cov(self.price_data)
        ef = EfficientFrontier(mu, S)
        weights = ef.max_sharpe()
        cleaned_weights = ef.clean_weights()
        return cleaned_weights
