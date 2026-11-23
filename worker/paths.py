"""Paths shared by the training and serving entry points."""
from pathlib import Path

MODEL = Path(__file__).resolve().parents[1] / "models" / "iris_logistic.onnx"
