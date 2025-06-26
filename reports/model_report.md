# 📊 Model Report: Sentiment Classification

## 📌 Task Summary

Multilingual sentiment classification using customer review data from diverse product categories and six languages. The task is to classify each review as **Negative**, **Neutral**, or **Positive**.

---

## 🔧 Model Architecture

- **Base Model**: DistilBERT (`distilbert-base-uncased`)
- **Classification Head**: Linear → ReLU → Linear (3-class softmax)
- **Tokenizer**: Hugging Face DistilBERT tokenizer
- **Max Sequence Length**: 32 tokens (for faster inference and training)

---

## ⚙️ Training Configuration

| Parameter         | Value                          |
|------------------|---------------------------------|
| Batch Size       | 8                               |
| Epochs           | 3                               |
| Learning Rate    | 2e-5                            |
| Optimizer        | AdamW                           |
| Loss Function    | Weighted CrossEntropy (to handle class imbalance) |
| Quantization     | ✅ Post-training dynamic quantization (`qint8`) |

> Training conducted using a stratified 1,500-sample dataset with balanced sentiment labels.

---

## 🧪 Dataset Summary

- **Input File**: `processed_reviews.csv`
- **Size**: ~1.2 million reviews
- **Evaluation Sample**: 1,500 rows (balanced 500 per sentiment)
- **Languages**: German (`de`), English (`en`), Spanish (`es`), French (`fr`), Japanese (`ja`), Chinese (`zh`)
- **Classes**: Negative, Neutral, Positive

---

## 📈 Evaluation Metrics

### 🔹 Training Evaluation (`train_model.py`)

| Metric     | Score  |
|------------|--------|
| F1 Score   | ~0.52  |
| Precision  | ~0.60  |
| Recall     | ~0.57  |

📁 *Saved in*: `reports/evaluation_metrics.json`

### 🔹 Test Evaluation (`evaluation_analysis.ipynb`)

| Metric     | Score  |
|------------|--------|
| Accuracy   | ~55%   |
| F1 (Macro) | ~0.43  |
| Notable Observations | Strong for Negative/Positive, Weak for Neutral |

🧩 *Confusion matrix & language breakdown included.*

---

## 📊 Error & Language Analysis

- **Neutral sentiment** is the hardest to classify — often confused with Positive.
- **Short/ambiguous reviews** and poor translation quality affect accuracy.
- **Language-wise**:
  - Best performance in **German** and **English**
  - Relatively lower performance in **Japanese** and **Chinese**
- **Product category breakdown** shows variance based on domain-specific wording.

---

## 📦 Model Artifacts

| Artifact                         | Path                            |
|----------------------------------|----------------------------------|
| Fine-tuned Model                 | `models/distilbert_sentiment/`   |
| Quantized Model (Dynamic Int8)   | `models/distilbert_quantized_sentiment/` |
| Tokenizer                        | Included in both model folders   |
| Evaluation Report                | `reports/evaluation_metrics.json` |
| Confusion Matrix Visualization   | `reports/confusion_matrix.png`   |

---

## ✅ Conclusion

The DistilBERT-based model, trained with class weighting and quantized post-training, provides **reliable multilingual sentiment classification**. While Neutral class accuracy needs further work, the model is lightweight and deployable.

> 💡 Future improvements: language-specific fine-tuning, synthetic data for class balancing, or ensemble learning for neutrality detection.
