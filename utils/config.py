import os

class Config:
    DATA_PATH = os.getenv('DATA_PATH', './data')
    INITIAL_BALANCE = float(os.getenv('INITIAL_BALANCE', 100000))
