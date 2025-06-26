# src/train_model.py
import pandas as pd
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score
from tqdm import tqdm
import json
import os
import time
from src.config import PROCESSED_DATA_PATH, REPORTS_PATH, MODEL_SAVE_PATH
from src.model_utils import load_tokenizer, load_pretrained_model
from transformers import DistilBertForSequenceClassification
from torch.quantization import quantize_dynamic
import warnings

# Suppress unnecessary warnings
warnings.filterwarnings("ignore", category=UserWarning, module='torch.quantization')
warnings.filterwarnings("ignore", category=UserWarning, module='transformers')

class SentimentDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
        
    def __getitem__(self, idx):
        return {
            'input_ids': self.encodings['input_ids'][idx],
            'attention_mask': self.encodings['attention_mask'][idx],
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }
        
    def __len__(self):
        return len(self.labels)

def save_quantized_model(model, tokenizer, model_name):
    """Save quantized model manually"""
    output_dir = os.path.join(MODEL_SAVE_PATH, model_name)
    os.makedirs(output_dir, exist_ok=True)
    
    # Save model state dict
    torch.save(model.state_dict(), os.path.join(output_dir, "pytorch_model.bin"))
    
    # Save config
    model.config.save_pretrained(output_dir)
    
    # Save tokenizer
    tokenizer.save_pretrained(output_dir)
    
    print(f"Quantized model saved to {output_dir}")

def compute_class_weights(labels):
    """Calculate class weights for imbalanced datasets"""
    class_counts = np.bincount(labels)
    total_samples = len(labels)
    return torch.tensor([
        total_samples / (3 * count) if count > 0 else 1.0
        for count in class_counts
    ], dtype=torch.float)

def train_sentiment_model(model_type="distilbert", epochs=3, batch_size=8, sample_size=1500):
    """Optimized training function with class weights and faster settings"""
    # Load and sample data
    df = pd.read_csv(PROCESSED_DATA_PATH)
    
    # Create balanced sample
    n_neg = int(sample_size * 0.4)
    n_neu = int(sample_size * 0.2)
    n_pos = sample_size - n_neg - n_neu
    
    df_sample = pd.concat([
        df[df['sentiment'] == 'Negative'].sample(n_neg, random_state=42),
        df[df['sentiment'] == 'Neutral'].sample(n_neu, random_state=42),
        df[df['sentiment'] == 'Positive'].sample(n_pos, random_state=42)
    ])
    
    texts = df_sample['text_for_model'].tolist()
    labels = df_sample['sentiment'].astype('category').cat.codes.values
    
    print(f"Using {len(texts)} samples")
    print(f"Label distribution: {np.bincount(labels)}")
    
    # Split data
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=42
    )
    
    # Initialize tokenizer and model
    tokenizer = load_tokenizer(model_type)
    model = load_pretrained_model(model_type, num_labels=3)
    
    # Tokenize with shorter sequences
    print("Tokenizing...")
    train_encodings = tokenizer(
        train_texts,
        padding='max_length',
        truncation=True,
        max_length=32,  # Reduced to 32 for speed
        return_tensors="pt"
    )
    val_encodings = tokenizer(
        val_texts,
        padding='max_length',
        truncation=True,
        max_length=32,
        return_tensors="pt"
    )
    
    # Create datasets
    train_dataset = SentimentDataset(train_encodings, train_labels)
    val_dataset = SentimentDataset(val_encodings, val_labels)
    
    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Optimizer and class-weighted loss
    optimizer = AdamW(model.parameters(), lr=2e-5)
    class_weights = compute_class_weights(labels)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    
    # Training loop
    for epoch in range(epochs):
        start_time = time.time()
        model.train()
        total_loss = 0
        
        # Progress bar with estimated time
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        for batch in progress_bar:
            optimizer.zero_grad()
            
            # Get model outputs
            outputs = model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask']
            )
            
            # Calculate loss with class weights
            loss = criterion(outputs.logits, batch['labels'])
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix({"loss": loss.item()})
        
        avg_loss = total_loss / len(train_loader)
        epoch_time = time.time() - start_time
        
        # Validation
        model.eval()
        val_preds, val_true = [], []
        with torch.no_grad():
            for batch in val_loader:
                outputs = model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask']
                )
                preds = torch.argmax(outputs.logits, dim=1)
                val_preds.extend(preds.numpy())
                val_true.extend(batch['labels'].numpy())
        
        f1 = f1_score(val_true, val_preds, average='weighted')
        print(f"Epoch {epoch+1} | Loss: {avg_loss:.4f} | F1: {f1:.4f} | Time: {epoch_time:.1f}s")
    
     # ✅ Save the unquantized model for evaluation
    from src.model_utils import save_model
    save_model(model, tokenizer, f"{model_type}_sentiment")
    
    # Apply quantization AFTER training
    quantized_model = quantize_dynamic(
        model, {torch.nn.Linear}, dtype=torch.qint8
    )
    
    # Save quantized model with custom method
    save_quantized_model(quantized_model, tokenizer, f"{model_type}_quantized_sentiment")
    
    # Save metrics
    metrics = {
        "f1_score": f1,
        "precision": precision_score(val_true, val_preds, average='weighted'),
        "recall": recall_score(val_true, val_preds, average='weighted')
    }
    os.makedirs(REPORTS_PATH, exist_ok=True)
    with open(os.path.join(REPORTS_PATH, "evaluation_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    
    print("Training complete. Quantized model saved.")

if __name__ == "__main__":
    train_sentiment_model(
        model_type="distilbert",
        batch_size=8,
        epochs=3,            # Increased epochs for better learning
        sample_size=1500      # Reduced sample size for speed
    )
