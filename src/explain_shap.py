# src/explain_shap.py
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import StandardScaler

MODEL_PATH = "models/rf_baseline.joblib"
DATA_CSV = "data/spectra_labeled.csv"
OUTDIR = "models/shap"
os.makedirs(OUTDIR, exist_ok=True)

def load_model(path=MODEL_PATH):
    data = joblib.load(path)
    return data['model'], data['scaler']

def main():
    print("Loading model...")
    model, scaler = load_model()
    print("Loading data...")
    df = pd.read_csv(DATA_CSV)
    X = df.drop(columns=["label"]).values
    y = df["label"].values
    Xs = scaler.transform(X)
    print("Creating SHAP explainer...")
    # Use TreeExplainer for RandomForest
    explainer = shap.TreeExplainer(model)
    # take a small sample (shap can be slow)
    sample_idx = np.random.choice(range(Xs.shape[0]), size=min(200, Xs.shape[0]), replace=False)
    X_sample = Xs[sample_idx]
    print("Computing SHAP values (this may take a bit)...")
    shap_values = explainer.shap_values(X_sample)
    # shap_values is list per class for multiclass
    # summary plot for each class
    for i, class_name in enumerate(model.classes_):
        plt.figure(figsize=(8,6))
        shap.summary_plot(shap_values[i], features=X_sample, show=False)
        plt.title(f"SHAP summary for class: {class_name}")
        plt.savefig(os.path.join(OUTDIR, f"shap_summary_{class_name}.png"), bbox_inches="tight")
        plt.close()
    print("Saved SHAP summary plots to", OUTDIR)

if __name__ == "__main__":
    main()
