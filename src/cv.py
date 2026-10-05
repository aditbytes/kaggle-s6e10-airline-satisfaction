"""Cross-validation splits. Every experiment uses the same folds so their scores are comparable."""
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from config import N_FOLDS, SEED


def make_folds(y: pd.Series, n_folds: int = N_FOLDS, seed: int = SEED) -> np.ndarray:
    """Return a fold number (0..n_folds-1) for every row, stratified on the target."""
    folds = np.full(len(y), -1, dtype=np.int8)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for fold, (_, valid_idx) in enumerate(skf.split(np.zeros(len(y)), y)):
        folds[valid_idx] = fold
    return folds
