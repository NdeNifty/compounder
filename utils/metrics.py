from typing import Dict
import numpy as np
import pandas as pd

def calculate_score(df: pd.DataFrame) -> float:
    """
    Compute return/volatility score.

    Args:
        df (pd.DataFrame): OHLCV data with 'Close' column.

    Returns:
        float: Score (mean return / volatility) or 0.0 if invalid.
    """
    if df.empty or 'Close' not in df.columns:
        logging.warning("Invalid DataFrame for score calculation")
        return 0.0

    returns = df['Close'].pct_change().mean()  # Scalar mean return
    volatility = df['Close'].pct_change().std()  # Scalar volatility
    return returns / volatility if volatility != 0 else 0.0

def sharpe_ratio(returns: np.ndarray, risk_free_rate: float = 0.0) -> float:
    """
    Calculate Sharpe ratio.

    Args:
        returns (np.ndarray): Array of returns.
        risk_free_rate (float): Risk-free rate.

    Returns:
        float: Sharpe ratio.
    """
    excess_returns = returns - risk_free_rate
    return np.mean(excess_returns) / np.std(excess_returns) if np.std(excess_returns) != 0 else 0.0

def win_rate(trades: list) -> float:
    """
    Calculate win rate from trade list.

    Args:
        trades (list): List of trade dicts with 'profit_loss' key.

    Returns:
        float: Win rate (fraction of profitable trades).
    """
    if not trades:
        return 0.0
    profits = [t['profit_loss'] for t in trades]
    return sum(p > 0 for p in profits) / len(profits)

def max_drawdown(portfolio_values: list) -> float:
    """
    Calculate max drawdown.

    Args:
        portfolio_values (list): List of portfolio values.

    Returns:
        float: Maximum drawdown (negative, as fraction).
    """
    arr = np.array(portfolio_values)
    peak = np.maximum.accumulate(arr)
    drawdown = (arr - peak) / peak
    return drawdown.min()