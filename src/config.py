"""Paths and constants shared by every script."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"          # OOF / test predictions (git-ignored)
SUBMISSION_DIR = ROOT / "submissions"  # files to upload (git-ignored)
EXPERIMENT_DIR = ROOT / "experiments"  # small JSON result per run (tracked)
REPORT_DIR = ROOT / "reports"

COMPETITION = "playground-series-s6e10"
ID = "id"
TARGET = "satisfaction"

SEED = 42
N_FOLDS = 5

CATEGORICAL = ["Gender", "Customer Type", "Type of Travel", "Class"]
SERVICE_RATINGS = [
    "Inflight wifi service",
    "Departure/Arrival time convenient",
    "Ease of Online booking",
    "Gate location",
    "Food and drink",
    "Online boarding",
    "Seat comfort",
    "Inflight entertainment",
    "On-board service",
    "Leg room service",
    "Baggage handling",
    "Checkin service",
    "Cleanliness",
]
DELAYS = ["Departure Delay in Minutes", "Arrival Delay in Minutes"]
NUMERIC = ["Age", "Flight Distance", *DELAYS]
