from pypfopt import EfficientFrontier, risk_models, expected_returns
import pandas as pd

class RiskManager:
    def __init__(self, price_data, risk_free_rate=0.0):
        self.price_data = price_data
        self.risk_free_rate = risk_free_rate

    def optimize_portfolio(self):
        mu = expected_returns.mean_historical_return(self.price_data)
        S = risk_models.sample_cov(self.price_data)
        ef = EfficientFrontier(mu, S)
        try:
            weights = ef.max_sharpe(risk_free_rate=self.risk_free_rate)
            cleaned_weights = ef.clean_weights()
            return cleaned_weights
        except ValueError as e:
            print(f"Warning: {e}. Using fallback weights.")
            # Fallback: allocate 100% to the first asset
            return {self.price_data.columns[0]: 1.0}
