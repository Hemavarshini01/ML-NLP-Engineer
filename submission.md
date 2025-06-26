# 🚀 Submission Report: Multilingual Sentiment Classification

## 👤 Author

- **Name**: Hemavarshini R
- **Challenge**: ML/NLP Engineer Internship Task

---

## 🧠 Objective

Build a robust multilingual text classification pipeline to predict review sentiment (Negative / Neutral / Positive) using Hugging Face Transformers.

---

## 📦 Dataset

- Source: `processed_reviews.csv`
- Size: ~1.2 million reviews
- Languages: German, English, Spanish, French, Japanese, Chinese
- Fields: review_body, review_title, sentiment, language, product_category, etc.

---

## 🧹 Preprocessing

- Combined `review_body` and `review_title` for context
- Normalized text: lowercasing, removing symbols
- Star ratings mapped to 3 sentiment classes
- Balanced sampling for training & evaluation
- Multilingual filtering using language code field

---

## 🤖 Model Design

- **Base**: `distilbert-base-uncased`
- **Tokenizer**: Hugging Face pre-trained
- **Fine-Tuning Head**: Linear classifier for 3 labels
- **Max Sequence Length**: 32
- **Tricks Used**:
  - Class weights to handle imbalance
  - Quantization (for fast inference and low memory)
  - Subsampling with multilingual stratification

---

## ⚙️ Training Summary

| Config        | Value     |
|---------------|-----------|
| Epochs        | 3         |
| Batch Size    | 8         |
| LR            | 2e-5      |
| Optimizer     | AdamW     |
| Loss          | CrossEntropy with class weights |
| Quantization  | Dynamic (qint8) after training |

---

## 📈 Evaluation (Unquantized Model)

- **F1 Score**: ~0.52
- **Accuracy**: 55.4%
- **Confusion Matrix**: Generated per-run
- **Multilingual Breakdown**: Evaluated across 6 languages
- **Misclassification**: Most common for Neutral class

---

## 🛠️ Technical Stack

- **Transformers**: Hugging Face (`transformers`)
- **DL Framework**: PyTorch
- **Data Tools**: Pandas, NumPy
- **Evaluation**: scikit-learn (F1, confusion matrix), ROC curves
- **Visuals**: Matplotlib, Seaborn

---

## 🧪 Advanced Evaluation

- ✅ ROC Curve (per-class) plotted
- ✅ Accuracy by language
- ✅ Accuracy by product category
- ✅ Misclassification samples explored
---

## 🔬 Error Analysis

- Confusions arise mainly between **Neutral vs. Positive**
- Reviews with very short or generic text are prone to misclassification
- Language-specific tokenization quirks affect performance (e.g., Chinese vs. English)

---

## 📁 Deliverables

- `models/`: Fine-tuned and quantized model folders
- `reports/`: Includes `confusion_matrix.png`, `evaluation_metrics.json`, and this report
- `notebooks/`: 
  - `data_exploration.ipynb`: EDA and class distributions  
  - `model_training.ipynb`: Interactive training/testing  
  - `evaluation_analysis.ipynb`: Misclassifications, language/category accuracy
- `src/`: Modular training scripts (`train_model.py`, `model_utils.py`, etc.)
- `main.py`: CLI-based pipeline runner
- `requirements.txt`: All required Python libraries

---

## 🧠 Key Learnings

- DistilBERT offers a great balance of **speed vs. accuracy** for low-resource NLP
- **Post-training quantization** significantly reduces model size with minimal performance loss
- Using **dynamic language-based sampling** helps in fair multilingual evaluation
- Handling large datasets on **low-RAM** machines requires:
  - Chunked sampling
  - Lightweight inference
  - Token length control
  - Efficient I/O using `csv.DictReader`, progress bars, etc.

---

## 🏁 Final Thoughts

This project showcases a complete **NLP model pipeline** — from raw multilingual data to quantized model inference, designed with **practical constraints** in mind.

DistilBERT is an effective transformer baseline, and this project is ready to be scaled or deployed with minimal adaptation.

> 💡 Future work: improve Neutral detection using auxiliary tasks, integrate XLM-R for deeper multilingual understanding, or experiment with LoRA for low-resource fine-tuning.
