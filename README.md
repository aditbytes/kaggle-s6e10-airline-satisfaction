# Predicting Airline Satisfaction · Kaggle Playground S6E10

My solution for [Playground Series – Season 6, Episode 10](https://www.kaggle.com/competitions/playground-series-s6e10):
predict whether an airline passenger was **satisfied**, from their trip details and service ratings.

| | |
|---|---|
| **Task** | Binary classification (`satisfaction`: True / False) |
| **Metric** | ROC AUC on the predicted probability |
| **Data** | 699,635 train rows · 299,844 test rows · 21 features |
| **Timeline** | 1 Oct 2026 → 31 Oct 2026 (23:59 UTC) |
| **Limits** | 10 submissions/day · teams up to 3 |

## Key findings from EDA

Full report: [reports/eda.md](reports/eda.md) (regenerate with `python src/eda.py`).

- **Target is balanced enough**: 44.4% satisfied. Folds are still stratified.
- **Strongest single signals**: `Online boarding` (AUC 0.84 on its own), `Class` (0.78), `Inflight entertainment` (0.74), `Seat comfort` (0.73), `Type of Travel` (0.70).
- **Segments matter a lot**: Business class is 72.5% satisfied vs 16.8% in Eco; personal travel is only 9.7% satisfied vs 58.4% for business travel.
- **Weak alone**: delays, `Gender`, `Gate location` (AUC ≈ 0.51). They may still help in interactions.
- **Missing values** only in `Arrival Delay in Minutes` (292 train / 130 test); LightGBM handles NaN natively.
- **No train/test drift**: every feature mean differs by under 2%, so local CV should track the leaderboard.

## How to run

```bash
make setup     # .venv with Python 3.12 + pandas, scikit-learn, LightGBM
make data      # needs the Kaggle CLI and the competition rules accepted on kaggle.com
make eda       # -> reports/eda.md
make smoke     # 30-second sanity check
make train EXP=exp001_lgbm_raw FEATURES=raw
make submit EXP=exp001_lgbm_raw
```

| Path | What's there |
|---|---|
| `src/config.py` | Paths, column groups, seed, number of folds |
| `src/data.py` | Loads CSVs with typed categoricals and a 0/1 target |
| `src/eda.py` | Writes `reports/eda.md` |
| `src/cv.py` | Stratified 5-fold split, identical for every experiment |
| `src/features.py` | Named feature sets (`--features raw`, …) |
| `src/models.py` | LightGBM hyperparameters |
| `src/train.py` | K-fold training → OOF AUC, `outputs/<exp>/`, `experiments/<exp>.json` |
| `src/submit.py` | Builds `submissions/<exp>.csv`; rank-blends several experiments |
| `experiments/` | One JSON per run: scores, params, feature importance (tracked in git) |
