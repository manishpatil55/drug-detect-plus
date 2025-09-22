# src/utils.py
import numpy as np
import os
import joblib

def ensure_dir(d):
    if not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

def load_model(path="models/rf_baseline.joblib"):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model not found at {path}. Run training first.")
    data = joblib.load(path)
    return data['model'], data['scaler']
