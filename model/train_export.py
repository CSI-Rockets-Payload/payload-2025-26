import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
import pandas as pd

from cnn1d import CAE1D


ARTIFACT_DIR = Path("artifacts")
ARTIFACT_DIR.mkdir(exist_ok=True)
BASE_DIR = Path.cwd()

def save_artifacts(ae, mu, std, config):
    ae.save(ARTIFACT_DIR / "autoencoder.keras")
    np.savez(ARTIFACT_DIR / "norm_stats.npz", mu=mu, std=std)

    with open(ARTIFACT_DIR / "config.json", "w") as f:
        json.dump(config, f, indent=2)


def main():
    # Reading simulated data from sim

    BASE_DIR = Path.cwd()  # model/
    sim_dir = BASE_DIR.parent / "sim" / "augmentedFlights"

    dfs = []

    for i, csv_file in enumerate(sim_dir.glob("*.csv")):
        d = pd.read_csv(csv_file)
        d["flight_id"] = i  # keep track of which flight
        dfs.append(d)

    df = pd.concat(dfs, ignore_index=True)

    df = df.rename(columns = {"Time (s)":"time", "acceleration Magnitude (m/s²)":"acceleration", "Vz (m/s)":"vz", "Z (m)":"z m"})

    # normalize features
    features = ["acceleration", "vz", "z"]
    X_win = df[features].to_numpy()   # (T, 3)

    # Fit normalization stats
    mu = X_win.mean(axis=0, keepdims=True)
    std = X_win.std(axis=0, keepdims=True) + 1e-8
    print(X_win[0])

    X_norm = (X_win - mu) / std
    df[features] = X_norm
    

    # Make sliding windows

    def make_windows(X, window_len=52, stride=13):
        windows = []
        for start in range(0, len(X) - window_len + 1, stride):
            windows.append(X[start:start + window_len])
        return np.stack(windows)

    all_windows = []

    for flight_id in df["flight_id"].unique():
        flight_df = df[df["flight_id"] == flight_id]

        X_flight = flight_df[["acceleration", "vz", "z"]].values  
        
        windows = make_windows(X_flight, 52, 13)
        all_windows.append(windows)

    X_norm = np.concatenate(all_windows, axis=0)
    print(X_norm[0])

    # Model
    model = CAE1D(window_length=52)
    ae = model.build_cnn1d_autoencoder()


    # Training model 
    history = ae.fit(
        X_norm,
        X_norm,
        epochs=20,
        batch_size=32,
        shuffle=True,
        validation_split=0.1
    )

    config = {
        "window_length": 52,
        "n_channels": 3,
        "score_type": "relative_mse",
        "predict_every_n_samples": 13
    }

    save_artifacts(ae, mu, std, config)

    print("Saved model and normalization stats to artifacts/")


if __name__ == "__main__":
    main()