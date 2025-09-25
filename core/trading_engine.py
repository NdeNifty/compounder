from typing import Dict, List
import logging
import pandas as pd
import numpy as np
from oandapyV20 import API
from oandapyV20.endpoints.orders import OrderCreate
from oandapyV20.endpoints.trades import TradeClose
from oandapyV20.endpoints.accounts import AccountSummary
import oandapyV20.endpoints.pricing as pricing

class TradingEngine:
    def __init__(self, starting_capital: float, config: Dict):
        """
        Initializes trading engine.

        Args:
            starting_capital (float): Initial capital.
            config (Dict): System config.
        """
        self.capital = starting_capital
        self.config = config
        self.positions: Dict[str, Dict] = {}  # stock: {'shares': qty, 'entry_price': price, 'stop_loss': price, 'take_profit': price}
        self.trades: List[Dict] = []  # Trade history
        self.portfolio_values: List[float] = [starting_capital]
        if config['mode'] == 'live':
            self.api = API(access_token=config['oanda_token'], environment=config['oanda_environment'])

    def predict(self, model, latest_data: pd.Series) -> float:
        """
        Predict next % change.

        Args:
            model: Trained model.
            latest_data (pd.Series): Latest features.

        Returns:
            float: Predicted % change.
        """
        features = latest_data.values.reshape(1, -1)
        if self.config['model_type'] == 'LSTM':
            features = np.reshape(features, (1, features.shape[1], 1))
        return model.predict(features)[0]

    def execute_trade(self, stock: str, action: str, price: float, quantity: float):
        """
        Execute trade (simulation or live).

        Args:
            stock (str): Ticker.
            action (str): 'buy' or 'sell'.
            price (float): Price.
            quantity (float): Shares.
        """
        if self.config['mode'] == 'live':
            # OANDA trade execution (adapt for CFD instrument names, e.g., 'AAPL_USD')
            instrument = f"{stock}_USD"  # Assuming CFD naming; adjust as needed
            units = quantity if action == 'buy' else -quantity
            order = {"order": {"instrument": instrument, "units": str(units), "type": "MARKET"}}
            r = OrderCreate(accountID=self.config['oanda_account_id'], data=order)
            self.api.request(r)
            logging.info(f"Live {action} {quantity} of {stock} at {price}")
        else:
            logging.info(f"Simulated {action} {quantity} of {stock} at {price}")

    def check_positions(self, current_prices: Dict[str, float]):
        """
        Check stop-loss/take-profit for open positions.

        Args:
            current_prices (Dict[str, float]): Latest prices.
        """
        for stock, pos in list(self.positions.items()):
            price = current_prices[stock]
            if price <= pos['stop_loss']:
                self.close_position(stock, price, 'stop_loss')
            elif price >= pos['take_profit']:
                self.close_position(stock, price, 'take_profit')

    def open_position(self, stock: str, price: float, predicted_change: float):
        """
        Open buy position if signal.

        Args:
            stock (str): Ticker.
            price (float): Current price.
            predicted_change (float): Predicted.
        """
        if predicted_change > self.config['buy_threshold'] and stock not in self.positions:
            risk_amount = self.config['risk_per_trade_percent'] * self.capital
            # Simple sizing; for better risk, use ATR or vol for stop distance
            quantity = (risk_amount / (price * self.config['stop_loss_percent'])) * self.config['leverage']
            self.execute_trade(stock, 'buy', price, quantity)
            self.positions[stock] = {
                'shares': quantity,
                'entry_price': price,
                'stop_loss': price * (1 - self.config['stop_loss_percent']),
                'take_profit': price * (1 + self.config['take_profit_percent'])
            }
            self.capital -= quantity * price  # Update capital (for sim; live query balance)

    def close_position(self, stock: str, price: float, reason: str):
        """
        Close position.

        Args:
            stock (str): Ticker.
            price (float): Exit price.
            reason (str): Reason.
        """
        pos = self.positions.pop(stock)
        profit_loss = pos['shares'] * (price - pos['entry_price'])
        self.capital += pos['shares'] * price  # Reinvest
        self.trades.append({
            'stock': stock,
            'action': 'sell',
            'price': price,
            'quantity': pos['shares'],
            'profit_loss': profit_loss,
            'reason': reason
        })
        logging.info(f"Closed {stock} at {price}, P/L: {profit_loss}")
        self.portfolio_values.append(self.capital)
        # For live, query balance to sync: r = AccountSummary(self.config['oanda_account_id']); response = self.api.request(r); self.capital = float(response['account']['balance'])

    def get_latest_price(self, stock: str) -> float:
        """
        Get latest price (sim or live).

        Args:
            stock (str): Ticker.

        Returns:
            float: Price.
        """
        if self.config['mode'] == 'live':
            instrument = f"{stock}_USD"
            params = {"instruments": instrument}
            r = pricing.PricingInfo(accountID=self.config['oanda_account_id'], params=params)
            response = self.api.request(r)
            return float(response['prices'][0]['closeoutAsk'])
        else:
            # For sim/backtest, assume from data
            return 100.0  # Dummy; replace with actual in loop

    def report(self):
        """Generate performance report."""
        returns = pd.Series(self.portfolio_values).pct_change().dropna()
        logging.info(f"Final Portfolio Value: {self.capital}")
        logging.info(f"Win Rate: {win_rate(self.trades)}")
        logging.info(f"Average Daily Return: {returns.mean()}")
        logging.info(f"Sharpe Ratio: {sharpe_ratio(returns.values)}")
        logging.info(f"Max Drawdown: {max_drawdown(self.portfolio_values)}")