class BaseStrategy:
    def __init__(self, params=None):
        self.params = params or {}

    def generate_signals(self, data):
        raise NotImplementedError("Must implement generate_signals method.")
