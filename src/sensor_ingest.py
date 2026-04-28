from pathlib import Path
import csv
from typing import Dict, Optional

import pandas as pd


DEFAULT_COLUMN_MAP = {
    "accel": "accel",
    "vz": "vz",
    "z": "z",
    "timestamp": "timestamp",
}


def _find_flight_header_row(csv_path: Path) -> Optional[int]:
    """
    Detect TeleMetrum/TelemPro-style export header row:
    row starts with "Flight,Acc,Acc,...", followed by a row starting with "time,..."
    """
    with csv_path.open("r", newline="") as f:
        reader = csv.reader(f)
        rows = list(reader)

    for i in range(len(rows) - 1):
        row0 = rows[i]
        row1 = rows[i + 1]
        if len(row0) < 2 or len(row1) < 2:
            continue
        if row0[0].strip().lower() == "flight" and row1[0].strip().lower() == "time":
            return i
    return None


def _load_simulation_export(csv_path: Path) -> pd.DataFrame:
    """
    Parse sim/simulation_sensor_data.csv format into standardized columns.
    Uses first 4 telemetry columns from the 'time' header row:
    time, acceleration, velocity, altitude
    """
    flight_header_row = _find_flight_header_row(csv_path)
    if flight_header_row is None:
        raise ValueError(f"Could not find simulation header block in {csv_path}")

    # Row layout:
    # - flight_header_row: group labels ("Flight,Acc,Acc,...")
    # - flight_header_row + 1: field names ("time,acceleration,velocity,altitude,...")
    # - flight_header_row + 2: units row ("sec,G's,feet/sec,feet AGL,...")
    # - flight_header_row + 3+: data rows
    raw = pd.read_csv(
        csv_path,
        header=flight_header_row + 1,
        skiprows=[flight_header_row + 2],
    )

    if raw.shape[1] < 4:
        raise ValueError(f"Expected at least 4 columns in {csv_path}, got {raw.shape[1]}")

    out = pd.DataFrame(
        {
            "timestamp": pd.to_numeric(raw.iloc[:, 0], errors="coerce"),
            "accel": pd.to_numeric(raw.iloc[:, 1], errors="coerce"),
            "vz": pd.to_numeric(raw.iloc[:, 2], errors="coerce"),
            "z": pd.to_numeric(raw.iloc[:, 3], errors="coerce"),
        }
    ).dropna(subset=["timestamp", "accel", "vz", "z"])

    return out.astype(
        {
            "timestamp": "float32",
            "accel": "float32",
            "vz": "float32",
            "z": "float32",
        }
    )


def load_sensor_dataframe(
    csv_path: str | Path,
    column_map: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    """
    Load telemetry CSV and return a standardized DataFrame.

    Standardized columns:
    - accel
    - vz
    - z
    - timestamp (optional; synthesized if missing)
    """
    csv_path = Path(csv_path)
    # Auto-parse your simulation export format.
    if csv_path.name == "simulation_sensor_data.csv":
        return _load_simulation_export(csv_path)

    df = pd.read_csv(csv_path)

    mapping = dict(DEFAULT_COLUMN_MAP)
    if column_map:
        mapping.update(column_map)

    required_inputs = [mapping["accel"], mapping["vz"], mapping["z"]]
    missing = [c for c in required_inputs if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required telemetry columns in {csv_path}: {missing}. "
            f"Available columns: {list(df.columns)}"
        )

    out = pd.DataFrame(
        {
            "accel": df[mapping["accel"]].astype("float32"),
            "vz": df[mapping["vz"]].astype("float32"),
            "z": df[mapping["z"]].astype("float32"),
        }
    )

    timestamp_col = mapping.get("timestamp")
    if timestamp_col and timestamp_col in df.columns:
        out["timestamp"] = df[timestamp_col]
    else:
        out["timestamp"] = out.index.astype("int64")

    return out


class CSVSensorStream:
    """
    Row-by-row CSV reader that yields one detector-ready sample per call.
    """

    def __init__(self, csv_path: str | Path, column_map: Optional[Dict[str, str]] = None):
        self.df = load_sensor_dataframe(csv_path, column_map=column_map)
        self.idx = 0

    def next_sample(self):
        if self.idx >= len(self.df):
            return None
        row = self.df.iloc[self.idx]
        self.idx += 1
        return [float(row["accel"]), float(row["vz"]), float(row["z"])]
