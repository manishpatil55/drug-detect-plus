# src/train_basic.py
import os
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from utils import ensure_dir

DATA_CSV = "data/spectra_labeled.csv"
MODEL_DIR = "models"
ensure_dir(MODEL_DIR)

def load_data(path=DATA_CSV):
    df = pd.read_csv(path)
    if "label" not in df.columns:
        raise RuntimeError("CSV must contain 'label' column.")
    X = df.drop(columns=["label"]).values
    y = df["label"].values
    return X, y

def main():
    print("Loading data...")
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, stratify=y, random_state=42)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    print("Training RandomForest...")
    clf = RandomForestClassifier(n_estimators=300, random_state=42, class_weight="balanced", n_jobs=-1)
    clf.fit(X_train_s, y_train)
    y_pred = clf.predict(X_test_s)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification report:\n", classification_report(y_test, y_pred))
    print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))
    joblib.dump({"model": clf, "scaler": scaler}, os.path.join(MODEL_DIR, "rf_baseline.joblib"))
    print("Saved model to models/rf_baseline.joblib")

if __name__ == "__main__":
    main()
