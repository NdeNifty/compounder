import logging
import os
from utils.config_loader import load_config
from utils.logger import setup_logger
from core.instrument_fetcher import get_instruments
from core.data_fetcher import fetch_and_cache_data
from core.stock_selector import select_top_stocks
from core.ml_trainer import train_models
from core.trading_engine import TradingEngine
from core.scanner import Scanner
from oandapyV20 import API

# Suppress TensorFlow oneDNN warning
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

def main():
    """
    Entry point for the automated trading system.
    Fetches all OANDA CFDs in all modes (no cap), scans full universe for top 3.
    Uses full instrument names (e.g., 'AAPL_USD') for OANDA fetches.
    """
    config = load_config('config/config.yaml')
    setup_logger(config['log_level'], config['log_file'])
    logging.info("Starting trading system with config: %s", config)

    api = None
    if config.get('data_source') == 'oanda':
        if not config['oanda_token'] or not config['oanda_account_id']:
            raise ValueError("OANDA credentials missing from .env")
        api = API(access_token=config['oanda_token'], environment=config['oanda_environment'])

    # Always fetch instruments (API if OANDA; fallback for sim/backtest)
    instruments = get_instruments(config, api)
    if not instruments:
        logging.error("No instruments fetched; using expanded config fallback.")
        instruments = config.get('oanda_supported', ['AAPL_USD', 'MSFT_USD'])

    logging.info(f"Using all {len(instruments)} instruments for scanning")

    config['stocks'] = instruments  # Full list with _USD

    # Fetch data: Use full instruments for OANDA; extract base for yfinance/files
    if config.get('data_source') == 'oanda':
        data = fetch_and_cache_data(instruments, config['historical_period'], config['historical_interval'], config, api)
        tickers = [inst.split('_')[0] for inst in instruments]  # Base for selection/models
    else:
        tickers = [inst.split('_')[0] for inst in instruments]
        data = fetch_and_cache_data(tickers, config['historical_period'], config['historical_interval'], config, api)

    # Select top 3 from full universe
    top_stocks = select_top_stocks(data, tickers, top_n=3)
    logging.info(f"Selected top 3 from {len(tickers)}: {top_stocks}")

    if not top_stocks:
        logging.error("No top stocks selected (empty data); aborting.")
        return

    models = train_models(data, top_stocks, config)
    engine = TradingEngine(config['starting_capital'], config)
    scanner = Scanner(config, data, models, engine)

    if config['mode'] == 'backtest':
        logging.info("Running backtest...")
        scanner.run_backtest()
    elif config['mode'] == 'simulation':
        logging.info("Running simulation...")
        scanner.run_simulation()
    elif config['mode'] == 'live':
        logging.info("Running live trading...")
        scanner.run_live()
    else:
        raise ValueError(f"Invalid mode: {config['mode']}")

if __name__ == "__main__":
    main()