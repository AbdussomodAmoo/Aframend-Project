import sys
import pandas as pd
from pathlib import Path
sys.path.append(str(Path(__file__).parent))
from toxicity_pipeline import AfrcanPhytochemicalToxicityPredictor
from config import Config
from toxicity_pipeline import PlantAgent
from toxicity_pipeline import FilterAgent
import streamlit as st

@st.cache_data
def load_data():
    url = "https://drive.google.com/file/d/1CUpGo3PuGQOTFOLyvQNG3ZKcK0oNU57Y/view?usp=sharing"
    return pd.read_csv(url)

filtered_afrodb_df = load_data()