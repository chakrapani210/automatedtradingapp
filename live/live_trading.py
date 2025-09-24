import backtrader as bt

class LiveTrading:
    def __init__(self, strategy, broker_api, data):
        self.strategy = strategy
        self.broker_api = broker_api
        self.data = data

    def run(self):
        cerebro = bt.Cerebro()
        cerebro.broker = self.broker_api
        cerebro.addstrategy(self.strategy)
        cerebro.adddata(self.data)
        cerebro.run()
