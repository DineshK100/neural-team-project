# Learning the Cliff — Predicting F1 Lap Time Loss From Tire Degradation

Predicts the lap-time penalty (in seconds) caused by tire degradation on a given lap using an LSTM trained on FastF1 timing data from the 2019–2024 F1 seasons (2022 excluded).

---

## Setup

```bash
pip install -r requirements.txt
```

---

## Quickstart (notebook)

Open `f1_tyre_degradation.ipynb` in Jupyter and run all cells top to bottom.

- If `laps.parquet` already exists in the repo, the notebook loads it directly — no download needed.
- If it is missing, the notebook fetches the configured seasons from FastF1 automatically (requires internet; first run takes 30–60 min; subsequent runs use the local cache).

---

## Running the scripts

### 1 · Build the dataset

```bash
# Full dataset (2019–2021, 2023–2024 — uses local cache if available)
python data.py --years 2019 2020 2021 2023 2024 --out laps.parquet

# Quick smoke test (2 races from 2023)
python data.py --years 2023 --limit 2 --out smoke.parquet
```

`data.py` downloads race sessions via FastF1, applies fuel correction, computes the per-stint tire-degradation target (`DegSec`), and saves a cleaned parquet file. Results are cached in `./fastf1_cache`; re-runs are fast.

### 2 · Train

```bash
# Linear baseline
python train.py --data laps.parquet --model linear

# LSTM (2-layer, dropout, early stopping)
python train.py --data laps.parquet --model lstm

# Override defaults
python train.py --data laps.parquet --model lstm --epochs 60 --lr 1e-3 --batch_size 128
```

Prints per-epoch train loss + val MAE/RMSE/R², restores the best checkpoint, then reports held-out 2024 test metrics.

---

## Data

| Field | Description |
|---|---|
| `Compound` | Tire compound (Soft / Medium / Hard) |
| `TyreLife` | Laps on the current set |
| `AirTemp`, `TrackTemp`, `Humidity`, `Rainfall` | Per-lap weather from FastF1 |
| `Driver`, `Circuit` | Identity features (one-hot encoded at train time) |
| `DegSec` | **Target** — fuel-corrected lap time minus the driver's fastest early-stint lap (seconds) |

Seasons: 2019, 2020, 2021, 2023, 2024 · Compounds: Soft / Medium / Hard · Green-flag accurate laps only · In/out laps excluded · `DegSec` range filter: [−1.0, 10.0] s

---

## Model

- **Linear baseline** — single `nn.Linear` over the current lap's features; establishes the floor.
- **LSTM** — 2-layer LSTM (64 hidden units, dropout 0.2), 8-lap sliding window, Adam + weight decay, early stopping on val RMSE.

Split: 2019–2023 train/val (chronological 80/20) · 2024 held-out test

---

## Baseline results (2023-only train, 2024 test)

| Model | MAE | RMSE | R² |
|---|---|---|---|
| Linear | 1.08 s | 1.44 s | −0.06 |
| LSTM | 1.38 s | 1.76 s | −0.58 |

Full 2019–2021 + 2023 results pending.

---

## References

- [FastF1](https://docs.fastf1.dev/)
- [Data-driven pit stop decision support for Formula 1 using deep learning models — Frontiers in AI, 2025](https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2025.1673148/full)
