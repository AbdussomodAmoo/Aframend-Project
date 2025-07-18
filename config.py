import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    QAIHUB_API_KEY = os.getenv('QAIHUB_API_KEY')
    STREAMLIT_PORT = int(os.getenv('STREAMLIT_SERVER_PORT', 8501))
    STREAMLIT_ADDRESS = os.getenv('STREAMLIT_SERVER_ADDRESS', 'localhost')

    # Data paths
    DATA_DIR = 'data'
    RESULTS_DIR = 'results'

    # Analysis settings
    RISK_THRESHOLDS = {
        'high': 0.7,
        'medium': 0.4,
        'low': 0.1
    }