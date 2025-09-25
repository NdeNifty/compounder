import logging

def setup_logger(level: str, file: str):
    """
    Set up logging to file and console.

    Args:
        level (str): Logging level (e.g., INFO).
        file (str): Log file path.
    """
    logging.basicConfig(
        level=logging.getLevelName(level),
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.FileHandler(file), logging.StreamHandler()]
    )