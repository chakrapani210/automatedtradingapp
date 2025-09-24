class AccountManager:
    def __init__(self, initial_balance=100000):
        self.balance = initial_balance
        self.positions = {}

    def update_balance(self, pnl):
        self.balance += pnl

    def update_position(self, symbol, qty):
        self.positions[symbol] = self.positions.get(symbol, 0) + qty
