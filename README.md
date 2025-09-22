# Drug Detect+ (student prototype)

Drug Detect+ is a software-only prototype for identifying fake and substandard medicines using spectral data (synthetic / public Raman-like or NIR-like spectra).

## Features
- Synthetic dataset generator (data/spectra_labeled.csv)
- Training pipeline (RandomForest baseline)
- SHAP explainability images
- Streamlit demo for local predictions and spectral overlays

## Quick start (Mac)
1. Create venv and install:
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

2. Generate data:
python src/data_prep.py

3. Train:
python src/train_basic.py

4. Run SHAP explain:
python src/explain_shap.py

5. Run demo:
streamlit run app_streamlit.py


## Notes
- This prototype uses synthetic spectra by default. You can replace with real spectra by placing per-spectrum CSVs in `data/raw/` (format to be harmonized).
- This is NOT certified to detect real-world counterfeit medicines. It's a research/learning prototype.
