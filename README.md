# 🤖 Multilingual Sentiment Classification  
### *ML/NLP Engineer Intern Challenge*

---

## 🎯 Objective

Build a complete **Sentiment classification pipeline** using Hugging Face Transformers. This challenge demonstrates end-to-end proficiency in:

- Text preprocessing and cleaning
- Transformer-based model fine-tuning (`DistilBERT`)
- Performance evaluation using precision, recall, and F1-score
- *(Bonus)*: Extend capabilities to support **6 languages** for multilingual sentiment analysis

---

## 📋 Task Overview

- ✅ Choose a labeled text dataset (e.g.amazon multilingual product review)
- ✅ Clean, normalize, and tokenize using Hugging Face tokenizers
- ✅ Fine-tune a pre-trained transformer (DistilBERT)
- ✅ Evaluate the model on test data using standard metrics
- ✅ Provide reports, visualizations, and insights
- 🌐 Support multilingual sentiment classification

---

## 🗂️ Project Structure

```text
ML-NLP-Engineer/
│
├── train.py                         # Entry-point: Runs end-to-end pipeline
├── requirements.txt                # Required dependencies
|
├── README.md                       # Project documentation (this file)
├── submission.md                   # Summary report for submission
── /data/                            # Paste downloaded dataset here
│
├── /notebooks/
│   ├── data_exploration.ipynb      # EDA and dataset insights
│   ├── model_training.ipynb        # Training experiments
│   └── evaluation_analysis.ipynb   # Error analysis, metrics, and plots
│
├── /src/
│   ├── config.py                   # Central config file (paths, constants)
│   ├── data_preprocessing.py      # Cleaning and tokenization logic
│   ├── model_utils.py             # Model loading and evaluation helpers
│   └── train_model.py             # Core training pipeline
│
├── /models/
│   ├── distilbert_sentiment/              # Saved fine-tuned DistilBERT model
│   └── distilbert_quantized_sentiment/ # Optimized model for inference
│
├── /reports/
│   ├── evaluation_metrics.json     # Precision, recall, F1 scores
│   ├── confusion_matrix.png        # Visual performance evaluation
│   └── model_report.md             # Technical summary and findings





---
## 🚀 Getting Started
### Download dataset from here:
[`reviews.csv`](https://drive.google.com/file/d/1uvPBl2z3mdrECuY80mTyAcVJm3QLD_9q/view?usp=sharing)  

### ✅ Step 1: Environment Setup

```bash
python -m venv venv
source venv/bin/activate    # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 📊 Step 2: Explore the Data

Open Jupyter notebook for exploration:

```bash
python -m src.data_preprocessing

```

```bash
jupyter notebook notebooks/data_exploration.ipynb
```

---

### 🏋️‍♂️ Step 3: Train the Model

Run the training pipeline:

```bash
python -m src.train_model
```

---

### 📈 Step 4: Evaluate Results

Visualize performance and perform error analysis:

```bash
jupyter notebook notebooks/evaluation_analysis.ipynb
```


### ▶️ Optional Step: Run the Entire Pipeline

This command runs preprocessing, model training, and launches the evaluation notebook:

```bash
python train.py
```
---

## 🧠 Key Learnings & Highlights

- Leveraged Hugging Face Transformers for efficient fine-tuning  
- Achieved high accuracy and balanced performance across classes  
- Implemented robust multilingual support  
- Generated meaningful visualizations and reports for insight  

---

## 🛠️ Tools & Libraries

- Python, PyTorch, Hugging Face Transformers  
- scikit-learn, Matplotlib/Seaborn, Jupyter Notebook


---
