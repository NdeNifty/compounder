"""
Instrument Fetcher Module

Fetches all tradable stock CFDs from OANDA v20 /v3/accounts/{id}/instruments.
No cap: Returns full list (~100-150). Caches hourly.

Dependencies: oandapyV20, json, logging, time (for retry).
"""

from typing import List, Dict, Optional
import json
import os
import logging
import time
from datetime import datetime, timedelta
from oandapyV20 import API
from oandapyV20.endpoints import accounts as accounts_endpoints
from oandapyV20.exceptions import V20Error

def fetch_oanda_instruments(config: Dict, api: Optional[API] = None) -> List[str]:
    """
    Fetch all tradable instruments; filter for stock CFDs per v20 docs.

    Args:
        config (Dict): With 'oanda_account_id', etc.
        api (Optional[API]): Client.

    Returns:
        List[str]: Full CFD list (e.g., ['AAPL_USD', ...]).
    """
    if not config.get('mode') == 'live':
        logging.warning("Instrument fetch skipped in non-live mode.")
        return []

    if api is None:
        api = API(access_token=config['oanda_token'], environment=config['oanda_environment'])

    try:
        r = accounts_endpoints.AccountInstruments(accountID=config['oanda_account_id'])
        response = api.request(r)
        instruments = response.get('instruments', [])
        stock_cfds = []

        logging.info("Fetching all OANDA instruments (no cap)...")
        for instr in instruments:
            name = instr['name']  # e.g., 'AAPL_USD'
            instr_type = instr.get('type', '')
            if (instr_type == 'CFD' and
                name.endswith('_USD') and
                not name.startswith(('EUR_', 'USD_', 'GBP_'))):  # Exclude forex
                stock_cfds.append(name)
                logging.info(f"{name}: Added as stock CFD")

        logging.info(f"Fetched all {len(stock_cfds)} stock CFDs")
        return sorted(stock_cfds)

    except V20Error as e:
        if e.status == 429:  # Rate limit
            logging.warning(f"Rate limit hit; retry in 5s: {e}")
            time.sleep(5)
            return fetch_oanda_instruments(config, api)  # Simple retry
        logging.error(f"OANDA API error: {e}")
        return []

def load_cached_instruments(cache_file: str = 'data/instruments.json', cache_hours: int = 1) -> List[str]:
    """
    Load instruments from cache if fresh.

    Args:
        cache_file (str): Path to JSON cache.
        cache_hours (int): Max age in hours.

    Returns:
        List[str]: Instruments or empty if stale/missing.
    """
    if not os.path.exists(cache_file):
        logging.warning(f"No cache found at {cache_file}")
        return []

    try:
        with open(cache_file, 'r') as f:
            data = json.load(f)
        timestamp = datetime.fromisoformat(data['timestamp'])
        if datetime.now() - timestamp < timedelta(hours=cache_hours):
            logging.info(f"Loaded {len(data['instruments'])} instruments from cache")
            return data['instruments']
        else:
            logging.info(f"Cache at {cache_file} is stale")
    except (json.JSONDecodeError, KeyError, ValueError) as e:
        logging.warning(f"Cache invalid: {e}")

    return []

def save_cached_instruments(instruments: List[str], cache_file: str = 'data/instruments.json'):
    """
    Save instruments to JSON cache.

    Args:
        instruments (List[str]): List to cache.
        cache_file (str): Path.
    """
    os.makedirs(os.path.dirname(cache_file), exist_ok=True)
    data = {
        'instruments': instruments,
        'timestamp': datetime.now().isoformat()
    }
    try:
        with open(cache_file, 'w') as f:
            json.dump(data, f, indent=2)
        logging.info(f"Cached {len(instruments)} instruments to {cache_file}")
    except Exception as e:
        logging.error(f"Failed to cache instruments: {e}")

def get_instruments(config: Dict, api: Optional[API] = None, cache_file: str = 'data/instruments.json') -> List[str]:
    """
    Get full instruments: Cache/API for all modes (sim uses fallback).
    """
    cached = load_cached_instruments(cache_file, config.get('instrument_cache_hours', 1))
    if cached:
        logging.info(f"Using cached {len(cached)} instruments")
        return cached

    if config.get('data_source') == 'oanda' and api:
        instruments = fetch_oanda_instruments(config, api)
    else:
        instruments = []  # Trigger fallback

    if instruments:
        save_cached_instruments(instruments, cache_file)
    else:
        instruments = config.get('oanda_supported', ['AAPL_USD', 'MSFT_USD'])
        logging.warning(f"Using fallback {len(instruments)} instruments")

    return instruments