from typing import Dict
import yaml
import os
from dotenv import load_dotenv

def load_config(file_path: str) -> Dict:
    """
    Load configuration from YAML file, overlaying OANDA creds from .env.

    Args:
        file_path (str): Path to config file.

    Returns:
        Dict: Parsed config with env vars merged.
    """
    load_dotenv()  # Load .env file
    
    with open(file_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Overlay OANDA from env
    config['oanda_token'] = os.getenv('OANDA_TOKEN', config.get('oanda_token', ''))
    config['oanda_account_id'] = os.getenv('OANDA_ACCOUNT_ID', config.get('oanda_account_id', ''))
    config['oanda_environment'] = os.getenv('OANDA_ENV', config.get('oanda_environment', 'practice'))
    
    return config