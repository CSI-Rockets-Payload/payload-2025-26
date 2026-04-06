import json
from pathlib import Path
from collections import deque

import numpy as np
import tensorflow as tf


# run model.predict

# compute relative reconstruction error

# print or publish the score




class OnlineDetector:

    def __init__(self, artifact_dir="artifacts"):
        artifact_dir = Path(artifact_dir)

        # load artifacts/autoencoder.keras
        self.model = tf.keras.models.load_model(artifact_dir / "autoencoder.keras")

        # load norm_stats.npz
        stats = np.load(artifact_dir / "norm_stats.npz")
        self.mu = stats["mu"].astype(np.float32)   # shape (1, 3)
        self.std = stats["std"].astype(np.float32) # shape (1, 3)

        # # load config.json
        with open(artifact_dir / "config.json", "r") as f:
            self.config = json.load(f)

        self.window_length = self.config["window_length"]
        self.n_channels = self.config["n_channels"]
        self.predict_every_n = self.config.get("predict_every_n_samples", 13)

        # maintain a rolling buffer of 52 samples
        self.buffer = deque(maxlen=self.window_length)
        self.sample_count = 0

    def _normalize_window(self, x):
        """
        x shape: (1, window_length, n_channels)
        mu/std shape: (1, n_channels)
        every 13 new samples, normalize the window with saved mu/std
        """
        return (x - self.mu) / self.std

    def _compute_score(self, x_norm):
        """
        x_norm shape: (1, window_length, n_channels)
        Returns scalar relative reconstruction error
        """
        x_hat = self.model.predict(x_norm, verbose=0)

        err = np.mean((x_norm - x_hat) ** 2, axis=(1, 2))
        energy = np.mean(x_norm ** 2, axis=(1, 2)) + 1e-8
        score = err / energy

        return float(score[0])

    def update(self, sample):
        """
        sample: one new telemetry sample, shape (n_channels,)
        Example: [acceleration, vz, z]

        Returns:
            None if not enough data yet or not prediction step
            float anomaly score otherwise
        """
        sample = np.asarray(sample, dtype=np.float32)

        if sample.shape != (self.n_channels,):
            raise ValueError(
                f"Expected sample shape ({self.n_channels},), got {sample.shape}"
            )

        # Add new sample to rolling buffer
        self.buffer.append(sample)
        self.sample_count += 1

        # Need a full window first
        if len(self.buffer) < self.window_length:
            return None

        # Only predict every N new samples
        if self.sample_count % self.predict_every_n != 0:
            return None

        # Build input window
        x = np.array(self.buffer, dtype=np.float32) 
        x = x[None, :, :]                           

        # Normalize with saved stats
        x_norm = self._normalize_window(x)

        # Score
        score = self._compute_score(x_norm)
        return score


def main():
    detector = OnlineDetector("artifacts")

    print("Detector initialized.")
    print(f"Window length: {detector.window_length}")
    print(f"Channels: {detector.n_channels}")
    print(f"Predict every {detector.predict_every_n} samples")

    # Demo loop with fake telemetry
    # Replace this with real telemetry stream later
    for t in range(200):
        sample = np.random.randn(detector.n_channels).astype(np.float32)

        score = detector.update(sample)

        if score is not None:
            print(f"t={t:03d} score={score:.6f}")


if __name__ == "__main__":
    main()