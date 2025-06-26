# src/data_preprocessing.py

import pandas as pd
import re
import os
from argostranslate import package, translate
from src.config import DATA_PATH, PROCESSED_DATA_PATH, SENTIMENT_MAP

def load_and_clean_data():
    df = pd.read_csv(DATA_PATH)
    # Drop rows with missing review_body or stars
    df = df.dropna(subset=['review_body', 'stars'])
    # Fill missing review_title with empty string
    df['review_title'] = df['review_title'].fillna("")
    # Remove duplicates
    df = df.drop_duplicates()
    return df

def preprocess_text(text):
    # Remove punctuation, emojis, special characters
    text = re.sub(r'[^\w\s]', '', str(text))
    # Remove emojis
    emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           "]+", flags=re.UNICODE)
    text = emoji_pattern.sub(r'', text)
    return text.strip()

def convert_stars_to_sentiment(star):
    try:
        star = int(star)
        return SENTIMENT_MAP.get(star, "Unknown")
    except:
        return "Unknown"

def translate_text(text, src_lang):
    if src_lang == 'en':
        return text
    try:
        installed_languages = translate.get_installed_languages()
        from_lang = next((l for l in installed_languages if l.code == src_lang), None)
        to_lang = next((l for l in installed_languages if l.code == 'en'), None)
        if from_lang and to_lang:
            translation = from_lang.get_translation(to_lang)
            return translation.translate(text)
        else:
            return text  # fallback: return original
    except Exception as e:
        print(f"Translation error: {e}")
        return text

def combine_title_body(title, body):
    return f"[CLS] {title} [SEP] {body} [SEP]"

def main():
    df = load_and_clean_data()
    # Clean text
    df['review_body'] = df['review_body'].apply(preprocess_text)
    df['review_title'] = df['review_title'].apply(preprocess_text)
    # Sentiment label
    df['sentiment'] = df['stars'].apply(convert_stars_to_sentiment)
    # Translate non-English reviews
    df['translated_body'] = df.apply(lambda row: translate_text(row['review_body'], row['language']), axis=1)
    df['translated_title'] = df.apply(lambda row: translate_text(row['review_title'], row['language']), axis=1)
    # Combine title and body
    df['text_for_model'] = df.apply(lambda row: combine_title_body(row['translated_title'], row['translated_body']), axis=1)
    # Save processed data
    os.makedirs(os.path.dirname(PROCESSED_DATA_PATH), exist_ok=True)
    df.to_csv(PROCESSED_DATA_PATH, index=False)
    print(f"Processed data saved to {PROCESSED_DATA_PATH}")

if __name__ == "__main__":
    main()
