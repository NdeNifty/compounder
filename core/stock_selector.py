from typing import List, Dict
import pandas as pd
from utils.metrics import calculate_score

def select_top_stocks(data: Dict[str, pd.DataFrame], stocks: List[str], top_n: int = 3) -> List[str]:
    """
    Select top stocks by score.

    Args:
        data (Dict[str, pd.DataFrame]): Historical data.
        stocks (List[str]): All stocks.
        top_n (int): Number to select.

    Returns:
        List[str]: Top stock tickers.
    """
    scores = {stock: calculate_score(df.tail(100)) for stock, df in data.items()}  # Last 100 periods for recent score
    sorted_stocks = sorted(scores, key=scores.get, reverse=True)
    return sorted_stocks[:top_n]