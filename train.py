import os

def run_pipeline():
    print("\n🚿 Step 1: Running Data Preprocessing...")
    os.system("python -m src.data_preprocessing")

    print("\n🏋️ Step 2: Training Sentiment Model...")
    os.system("python -m src.train_model")

    print("\n📊 Step 3: Launching Evaluation Notebook (interactive)...")
    os.system("jupyter notebook notebooks/evaluation_analysis.ipynb")

if __name__ == "__main__":
    run_pipeline()
