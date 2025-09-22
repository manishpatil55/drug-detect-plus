# app_streamlit.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib, os
import matplotlib.pyplot as plt

MODEL_PATH = "models/rf_baseline.joblib"
DATA_PATH = "data/spectra_labeled.csv"
WAVELENGTHS_PATH = "data/wavelengths.npy"

st.set_page_config(page_title="Drug Detect+", layout="wide")
st.title("Drug Detect+ — Local Demo")

# Load model
if os.path.exists(MODEL_PATH):
    saved = joblib.load(MODEL_PATH)
    model = saved['model']
    scaler = saved['scaler']
    st.success("Model loaded.")
else:
    model = None
    scaler = None
    st.warning("No model found. Run training (python src/train_basic.py) first.")

uploaded = st.file_uploader("Upload spectra CSV (columns i_0...i_N and optional 'label')", type=["csv"])
use_sample = st.button("Use generated sample (first 200 rows)")

df = None
if uploaded:
    df = pd.read_csv(uploaded)
elif use_sample and os.path.exists(DATA_PATH):
    df = pd.read_csv(DATA_PATH).sample(n=200, random_state=42)

if df is not None:
    st.subheader("Data preview")
    st.dataframe(df.head())

    # find intensity columns
    cols = [c for c in df.columns if c != "label"]
    if len(cols) == 0:
        st.error("No intensity columns found (columns should be i_0 ... i_N).")
    else:
        # show label distribution if exists
        if "label" in df.columns:
            st.write("Label counts:")
            st.bar_chart(df["label"].value_counts())

        # show spectral overlay for a few samples
        st.subheader("Spectral overlay (first 6 samples)")
        wl = None
        if os.path.exists(WAVELENGTHS_PATH):
            wl = np.load(WAVELENGTHS_PATH)
        else:
            wl = np.arange(len(cols))
        fig, ax = plt.subplots(figsize=(10,4))
        for idx in range(min(6, df.shape[0])):
            spectrum = df.iloc[idx][cols].values.astype(float)
            ax.plot(wl, spectrum, label=f"sample_{idx}")
        ax.set_xlabel("Wavelength (a.u.)" if wl is None else "Wavelength")
        ax.set_ylabel("Normalized intensity")
        ax.legend()
        st.pyplot(fig)

        if model is not None:
            st.subheader("Predictions")
            X = df[cols].values.astype(float)
            Xs = scaler.transform(X)
            preds = model.predict(Xs)
            probs = model.predict_proba(Xs)
            out = pd.DataFrame({"prediction": preds})
            if "label" in df.columns:
                out["true_label"] = df["label"].values
            st.dataframe(out.head(50))

            # show probability distribution for first few samples
            st.subheader("Prediction probabilities (first 10)")
            prob_df = pd.DataFrame(probs, columns=model.classes_)
            st.dataframe(prob_df.head(10))

            # If SHAP images were computed, show one
            shap_img_dir = "models/shap"
            if os.path.exists(shap_img_dir) and len(os.listdir(shap_img_dir))>0:
                st.subheader("SHAP (precomputed)")
                imgs = [os.path.join(shap_img_dir, f) for f in os.listdir(shap_img_dir) if f.endswith(".png")]
                for img in imgs[:3]:
                    st.image(img, caption=os.path.basename(img))
        else:
            st.info("Train the model to enable predictions.")
else:
    st.info("Upload a CSV or press 'Use generated sample' (after running data_prep).")
