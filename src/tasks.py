import os
import time
from pathlib import Path

from .celery_app import app
from .sensor_ingest import CSVSensorStream
from model.detector import OnlineDetector


_detector = None
_csv_stream = None


def _get_detector() -> OnlineDetector:
    global _detector
    if _detector is None:
        artifact_dir = Path(
            os.getenv("MODEL_ARTIFACT_DIR", "/workspace/model/artifacts")
        )
        _detector = OnlineDetector(artifact_dir=artifact_dir)
    return _detector


def _read_sensor_sample():
    """
    Read one sample from Kate-3.

    Replace this function with sensor integration.
    Expected return shape: [accel, vz, z] (length must match config n_channels).

    """
    global _csv_stream
    csv_path = os.getenv("SENSOR_CSV_PATH")
    if not csv_path:
        return None

    if _csv_stream is None:
        _csv_stream = CSVSensorStream(csv_path)

    return _csv_stream.next_sample()


@app.task(name="src.tasks.score_telemetry_sample")
def score_telemetry_sample(sample, timestamp=None):
    detector = _get_detector()
    score = detector.update(sample)

    result = {
        "timestamp": timestamp if timestamp is not None else time.time(),
        "score_ready": score is not None,
        "score": float(score) if score is not None else None,
    }

    if result["score_ready"]:
        print(f"[detector] score={result['score']:.6f} ts={result['timestamp']:.3f}")

    return result


@app.task(name="src.tasks.poll_sensor_and_score")
def poll_sensor_and_score():
    sample = _read_sensor_sample()
    if sample is None:
        return {"score_ready": False, "reason": "no_new_sample"}
    return score_telemetry_sample(sample=sample)
