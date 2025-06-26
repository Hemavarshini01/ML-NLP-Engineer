# src/model_utils.py
from transformers import XLMRobertaTokenizer, XLMRobertaForSequenceClassification
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch
import os
from src.config import MODEL_SAVE_PATH

def load_tokenizer(model_type="xlm-roberta"):
    """Load appropriate tokenizer based on model choice"""
    if model_type == "distilbert":
        return DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    else:  # Default to XLM-RoBERTa
        return XLMRobertaTokenizer.from_pretrained('xlm-roberta-base')

def tokenize_data(tokenizer, texts, max_length=256):
    """Tokenize text data with padding and truncation"""
    return tokenizer(
        texts,
        padding='max_length',
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )

def load_pretrained_model(model_type="xlm-roberta", num_labels=3):
    """Load pre-trained model with CPU configuration"""
    if model_type == "distilbert":
        model = DistilBertForSequenceClassification.from_pretrained(
            'distilbert-base-uncased',
            num_labels=num_labels
        )
    else:  # Default to XLM-RoBERTa
        model = XLMRobertaForSequenceClassification.from_pretrained(
            'xlm-roberta-base',
            num_labels=num_labels
        )
    
    # Explicit CPU configuration
    model = model.to(torch.device("cpu"))
    return model

def save_model(model, tokenizer, model_name="sentiment_model"):
    """Save model and tokenizer to specified path"""
    output_dir = os.path.join(MODEL_SAVE_PATH, model_name)
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Model saved to {output_dir}")

def load_model(model_path):
    """Load saved model and tokenizer from path"""
    if "distilbert" in model_path.lower():
        model = DistilBertForSequenceClassification.from_pretrained(model_path)
        tokenizer = DistilBertTokenizer.from_pretrained(model_path)
    else:
        model = XLMRobertaForSequenceClassification.from_pretrained(model_path)
        tokenizer = XLMRobertaTokenizer.from_pretrained(model_path)
    
    model = model.to(torch.device("cpu"))
    return model, tokenizer
