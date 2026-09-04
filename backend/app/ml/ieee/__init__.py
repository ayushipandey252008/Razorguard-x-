"""IEEE-CIS offline evaluation package. Isolated from live scoring and ULB."""

from app.ml.ieee.adapter import ieee_files_present, setup_payload
from app.ml.ieee.constants import LIVE_MODEL_VERSION, SETUP_MESSAGE, TRACK
from app.ml.ieee.pipeline import run_ieee_pipeline

__all__ = [
    "LIVE_MODEL_VERSION",
    "SETUP_MESSAGE",
    "TRACK",
    "ieee_files_present",
    "run_ieee_pipeline",
    "setup_payload",
]
