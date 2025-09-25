"""
Data Fetcher Module

Fetches historical OHLCV from OANDA v20 /v3/instruments/{inst}/candles for all CFDs.
Paginated for >5000 candles; parallel via joblib.

Dependencies: pandas, oandapyV20, joblib, logging.
"""

from typing import Dict, List
import os
import pandas as pd
import logging
from datetime import datetime, timedelta
from oandapyV20 import API
from oandapyV20.endpoints import instruments as instruments_endpoints
from oandapyV20.exceptions import V20Error
from joblib import Parallel, delayed
import yfinance as yf  # Fallback

def fetch_oanda_candles(instrument: str, api: API, from_date: str, granularity: str, max_candles: int = 5000) -> pd.DataFrame:
    """
    Fetch paginated candles per v20 docs (ISO DateTime).

    Args:
        instrument (str): e.g., 'AAPL_USD'.
        api (API): Client.
        from_date (str): ISO 'YYYY-MM-DDTHH:MM:SSZ'.
        granularity (str): e.g., 'H1'.
        max_candles (int): Per request.

    Returns:
        pd.DataFrame: OHLCV (index: time).
    """
    all_candles = []
    current_from = from_date
    while True:
        params = {
            'from': current_from,
            'granularity': granularity,
            'count': max_candles
        }
        r = instruments_endpoints.InstrumentsCandles(instrument=instrument, params=params)
        try:
            api.request(r)
            candles = r.response['candles']
            if not candles:
                break
            all_candles.extend(candles)
            if len(candles) < max_candles:
                break
            current_from = candles[-1]['time']  # ISO for next page
            logging.debug(f"{instrument}: Paginated from {current_from}")
        except V20Error as e:
            logging.error(f"{instrument}: Candle error ({e})")
            break

    if not all_candles:
        return pd.DataFrame()

    df_data = []
    for candle in all_candles:
        mid = candle['mid']
        df_data.append({
            'Open': float(mid['o']),
            'High': float(mid['h']),
            'Low': float(mid['l']),
            'Close': float(mid['c']),
            'Volume': int(candle.get('volume', 0))
        })
    df = pd.DataFrame(df_data)
    df.index = pd.to_datetime([c['time'] for c in all_candles])  # Parse ISO
    df.sort_index(inplace=True)
    return df

def fetch_oanda_data(instruments: List[str], config: Dict, api: API) -> Dict[str, pd.DataFrame]:
    """
    Parallel fetch for all instruments.
    """
    # ISO dates
    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=365) if config['historical_period'] == '1y' else end_date
    from_date = start_date.strftime('%Y-%m-%dT%H:%M:%S.000000000Z')  # UTC ISO

    granularity = config.get('oanda_granularity_map', {}).get(config['historical_interval'], 'H1')
    max_candles = config.get('max_candles_per_request', 5000)

    logging.info(f"Fetching OANDA data for all {len(instruments)} instruments...")
    results = Parallel(n_jobs=-1)(delayed(fetch_oanda_candles)(inst, api, from_date, granularity, max_candles) for inst in instruments)

    data = {}
    for inst, df in zip(instruments, results):
        base_ticker = inst.split('_')[0]
        file = f'data/{base_ticker}.csv'
        try:
            if df.empty:
                logging.warning(f"{base_ticker}: Empty data")
                continue
            # Cache check (simple: overwrite if new)
            df.to_csv(file)
            data[base_ticker] = df
            logging.info(f"{base_ticker}: Fetched {len(df)} candles from OANDA")
        except Exception as e:
            logging.error(f"{base_ticker}: Failed ({e})")

    logging.info(f"OANDA data complete: {len(data)} instruments")
    return data

def fetch_and_cache_data(stocks_or_instruments: List[str], period: str, interval: str, config: Dict = None, api: API = None) -> Dict[str, pd.DataFrame]:
    """
    Unified: OANDA (all CFDs) or yfinance fallback.
    """
    source = config.get('data_source', 'yfinance') if config else 'yfinance'
    os.makedirs('data', exist_ok=True)

    if source == 'oanda' and api:
        return fetch_oanda_data(stocks_or_instruments, config, api)
    else:
        # yfinance fallback (unchanged)
        data = {}
        logging.info(f"Yfinance fallback for {len(stocks_or_instruments)} stocks...")
        for stock in stocks_or_instruments:
            file = f'data/{stock}.csv'
            try:
                if os.path.exists(file):
                    df = pd.read_csv(file, index_col=0, parse_dates=True)
                    logging.info(f"{stock}: Loaded from cache")
                else:
                    df = yf.download(stock, period=period, interval=interval, progress=False, auto_adjust=False)
                    if df.empty:
                        continue
                    df.to_csv(file)
                    logging.info(f"{stock}: Fetched and cached")
                data[stock] = df
            except Exception as e:
                logging.error(f"{stock}: Failed ({e})")
        return data