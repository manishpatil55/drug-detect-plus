# src/data_prep.py
"""
Data preparation:
- If CSV files are placed in data/raw/ they will be attempted to load (format expectations below).
- Otherwise synthetic dataset is generated and saved at data/spectra_labeled.csv
Output:
- data/spectra_labeled.csv : each row has intensity columns i_0 ... i_N and a label column.
- data/wavelengths.npy : the wavelength grid used.
"""
import os, glob, random, math
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

random.seed(42)
np.random.seed(42)

OUTDIR = "data"
RAW_GLOB = os.path.join(OUTDIR, "raw", "*.csv")

def make_wavelengths(n=1401, start=400, stop=1800):
    return np.linspace(start, stop, n)

def gaussian(x, mu, sigma, amp=1.0):
    return amp * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

def synth_base_spectrum(wavelengths, peaks):
    s = np.zeros_like(wavelengths)
    for mu, sigma, amp in peaks:
        s += gaussian(wavelengths, mu, sigma, amp)
    baseline = 0.2 * (1 + 0.0004 * (wavelengths - wavelengths.mean()))
    return s + baseline

def make_random_api(wl):
    peaks = [
        (500 + random.uniform(-12,12), 4+random.uniform(0,8), 1.0+random.uniform(-0.1,0.1)),
        (780 + random.uniform(-15,15), 7+random.uniform(0,12), 0.8+random.uniform(-0.2,0.2)),
        (1200 + random.uniform(-25,25), 9+random.uniform(0,15), 0.6+random.uniform(-0.3,0.3)),
        (1500 + random.uniform(-25,25), 18+random.uniform(0,30), 0.4+random.uniform(-0.2,0.2)),
    ]
    return synth_base_spectrum(wl, peaks)

def add_noise(s, snr_db=30):
    power_signal = np.mean(s**2)
    snr_linear = 10 ** (snr_db/10)
    noise_power = power_signal / snr_linear
    noise = np.random.normal(0, math.sqrt(noise_power), s.shape)
    return s + noise

def snv(s):
    return (s - np.mean(s)) / (np.std(s) + 1e-9)

def generate_dataset(out_csv=os.path.join(OUTDIR, "spectra_labeled.csv"),
                     n_per_class=250):
    os.makedirs(OUTDIR, exist_ok=True)
    wl = make_wavelengths()
    records = []
    api = make_random_api(wl)
    filler = synth_base_spectrum(wl, [(600,80,0.7),(1000,120,0.5)])
    other_api = make_random_api(wl) * 0.9 + 0.1 * api

    def create_sample(base_spec, snr_db=30, smooth=True):
        s = base_spec.copy()
        s = add_noise(s, snr_db=snr_db)
        if smooth:
            s = savgol_filter(s, window_length=11, polyorder=3)
        s = snv(s)
        return s

    # genuine
    for _ in range(n_per_class):
        s = create_sample(api, snr_db=random.uniform(28,36))
        records.append(np.concatenate([s.astype(object), ["genuine"]]))

    # substandard: decrease API fraction at several levels
    frac_list = [0.9, 0.75, 0.5, 0.3]
    for frac in frac_list:
        for _ in range(int(n_per_class * 0.25)):
            mixed = frac*api + (1-frac)*filler
            s = create_sample(mixed, snr_db=random.uniform(20,32))
            records.append(np.concatenate([s.astype(object), ["substandard"]]))

    # counterfeit
    for _ in range(n_per_class):
        mixed = 0.6*other_api + 0.4*filler
        s = create_sample(mixed, snr_db=random.uniform(20,35))
        records.append(np.concatenate([s.astype(object), ["counterfeit"]]))

    df = pd.DataFrame(records)
    n_cols = wl.size
    cols = [f"i_{j}" for j in range(n_cols)] + ["label"]
    df.columns = cols
    # convert intensity columns to float
    for c in cols[:-1]:
        df[c] = df[c].astype(float)
    df.to_csv(out_csv, index=False)
    np.save(os.path.join(OUTDIR, "wavelengths.npy"), wl)
    print(f"Saved dataset to {out_csv} ({len(df)} rows) and wavelengths to data/wavelengths.npy")

def try_load_raw():
    files = glob.glob(RAW_GLOB)
    if not files:
        return None
    # for this project we skip trying to harmonize arbitrary files automatically
    return None

if __name__ == "__main__":
    print("Checking data/raw for CSVs...")
    raw = try_load_raw()
    if raw is None:
        print("No raw input detected or not implemented - generating synthetic dataset.")
        generate_dataset()
    else:
        print("Raw data found - merge pipeline not implemented in this quick script.")
