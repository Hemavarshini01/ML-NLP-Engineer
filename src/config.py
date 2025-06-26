# src/config.py

DATA_PATH = "data/reviews.csv"
MODEL_SAVE_PATH = "models/"
REPORTS_PATH = "reports/"
PROCESSED_DATA_PATH = "data/processed_reviews.csv"
LANGUAGES = ["de", "en", "es", "fr", "ja", "zh"]
SENTIMENT_MAP = {1: "Negative", 2: "Negative", 3: "Neutral", 4: "Positive", 5: "Positive"}
MAX_SEQ_LENGTH = 256
