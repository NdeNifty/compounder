from typing import Dict
import time
import logging
from core.stock_selector import select_top_stocks
from core.ml_trainer import train_models, create_features

class Scanner:
    def __init__(self, config: Dict, data: Dict, models: Dict, engine: 'TradingEngine'):
        """
        Continuous scanner.

        Args:
            config (Dict): Config.
            data (Dict): Historical data.
            models (Dict): Models.
            engine (TradingEngine): Trading engine.
        """
        self.config = config
        self.data = data
        self.models = models
        self.engine = engine
        self.top_stocks = list(models.keys())

    def update_top_stocks(self):
        """Re-select top stocks and retrain if changed."""
        new_top = select_top_stocks(self.data, self.config['stocks'], top_n=3)
        if set(new_top) != set(self.top_stocks):
            new_stocks = [s for s in new_top if s not in self.top_stocks]
            if new_stocks:
                new_models = train_models(self.data, new_stocks, self.config)
                self.models.update(new_models)
            self.top_stocks = new_top
            logging.info(f"Updated top stocks: {self.top_stocks}")

    def run_loop(self, is_live: bool = False):
        """Core loop for simulation/live."""
        end_time = time.time() + (self.config['session_hours'] * 3600)
        while time.time() < end_time:
            self.update_top_stocks()  # Scan

            # Get latest data/features
            current_prices = {stock: self.engine.get_latest_price(stock) for stock in self.top_stocks}

            self.engine.check_positions(current_prices)  # Check SL/TP

            for stock in self.top_stocks:
                # For sim/live, append latest (dummy for sim)
                latest_df = self.data[stock].tail(1)
                features = create_features(latest_df, self.config).iloc[-1]
                predicted = self.engine.predict(self.models[stock], features)
                self.engine.open_position(stock, current_prices[stock], predicted)

            time.sleep(self.config['interval_minutes'] * 60)

        self.engine.report()

    def run_simulation(self):
        """Run simulation mode."""
        logging.info("Starting simulation...")
        self.run_loop()

    def run_live(self):
        """Run live mode."""
        logging.info("Starting live trading...")
        self.run_loop(is_live=True)

    def run_backtest(self):
        """Run backtest on historical data."""
        logging.info("Starting backtest...")
        # Simulate loop over historical data
        for t in range(len(self.data[self.top_stocks[0]]) - 100):  # Example sliding window
            # Slice data up to t, compute features, predict, trade
            # Omitted detailed impl for brevity; similar to run_loop but historical
            pass
        self.engine.report()