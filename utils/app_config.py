from dataclasses import dataclass, field
from typing import List
import yaml

@dataclass
class AppConfig:
    tickers: List[str] = field(default_factory=lambda: ['AAPL'])
    start_date: str = '2022-01-01'
    end_date: str = '2023-01-01'
    initial_cash: float = 100000
    risk_free_rate: float = 0.0

    @staticmethod
    def load_from_yaml(path: str = 'config.yaml'):
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        return AppConfig(
            tickers=data.get('tickers', ['AAPL']),
            start_date=data.get('start_date', '2022-01-01'),
            end_date=data.get('end_date', '2023-01-01'),
            initial_cash=data.get('initial_cash', 100000),
            risk_free_rate=data.get('risk_free_rate', 0.0)
        )
