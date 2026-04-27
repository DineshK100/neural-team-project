"""Train the baseline LSTM (or linear baseline) on the prepared lap dataset.

Usage:
    python train.py --data laps.parquet --model lstm
    python train.py --data laps.parquet --model linear
"""

import argparse

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

from model import LinearBaseline, LSTMRegressor  # type: ignore[import-not-found]

WINDOW = 5
COMPOUNDS = ["SOFT", "MEDIUM", "HARD"]
NUMERIC_FEATURES = ["TyreLife", "AirTemp", "TrackTemp", "Humidity", "Rainfall"]


def encode(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in COMPOUNDS:
        df[f"is_{c}"] = (df["Compound"] == c).astype(float)
    for c in NUMERIC_FEATURES:
        coerced = pd.to_numeric(df[c], errors="coerce")
        df[c] = pd.Series(coerced, index=df.index).fillna(0.0).astype(float)
    return df


def build_identity_vocab(df: pd.DataFrame):
    drivers = sorted(df["Driver"].dropna().unique().tolist())
    circuits = sorted(df["Circuit"].dropna().unique().tolist())
    return drivers, circuits


def add_identity_columns(df: pd.DataFrame, drivers, circuits) -> pd.DataFrame:
    df = df.copy()
    for d in drivers:
        df[f"drv_{d}"] = (df["Driver"] == d).astype(float)
    for c in circuits:
        df[f"crc_{c}"] = (df["Circuit"] == c).astype(float)
    return df


def make_windows(
    df: pd.DataFrame,
    drivers,
    circuits,
    scaler: StandardScaler | None = None,
    fit_scaler: bool = False,
):
    feat_cols = (
        NUMERIC_FEATURES
        + [f"is_{c}" for c in COMPOUNDS]
        + [f"drv_{d}" for d in drivers]
        + [f"crc_{c}" for c in circuits]
    )
    X_list, y_list = [], []
    grouped = (
        df.sort_values(["Year", "Round", "Driver", "Stint", "LapNumber"])
        .groupby(["Year", "Round", "Driver", "Stint"], sort=False)
    )
    for _, g in grouped:
        if len(g) < WINDOW:
            continue
        feats = g[feat_cols].to_numpy(dtype=np.float32)
        targets = g["DegSec"].to_numpy(dtype=np.float32)
        for i in range(WINDOW - 1, len(g)):
            X_list.append(feats[i - WINDOW + 1 : i + 1])
            y_list.append(targets[i])

    if not X_list:
        return None, None, scaler

    X = np.stack(X_list).astype(np.float32)
    y = np.array(y_list, dtype=np.float32)

    n_num = len(NUMERIC_FEATURES)
    n_windows, win_len, _ = X.shape
    flat = X[:, :, :n_num].reshape(-1, n_num)
    if fit_scaler:
        scaler = StandardScaler().fit(flat)
    assert scaler is not None
    flat_scaled = scaler.transform(flat)  # type: ignore[assignment]
    X[:, :, :n_num] = np.asarray(flat_scaled, dtype=np.float32).reshape(n_windows, win_len, n_num)

    return X, y, scaler


class WindowDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.y[i]


def train_one_epoch(model, loader, optimizer, loss_fn, device) -> float:
    model.train()
    total = 0.0
    n = 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        pred = model(X)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()
        total += loss.item() * len(y)
        n += len(y)
    return total / max(n, 1)


@torch.no_grad()
def evaluate(model, loader, device) -> dict:
    model.eval()
    preds, ys = [], []
    for X, y in loader:
        preds.append(model(X.to(device)).cpu().numpy())
        ys.append(y.numpy())
    preds = np.concatenate(preds)
    ys = np.concatenate(ys)
    return {
        "mae": float(mean_absolute_error(ys, preds)),
        "rmse": float(np.sqrt(mean_squared_error(ys, preds))),
        "r2": float(r2_score(ys, preds)),
    }


def chronological_split(df: pd.DataFrame, val_fraction: float = 0.2):
    df = df.sort_values(["Year", "Round"]).reset_index(drop=True)
    cut = int((1 - val_fraction) * len(df))
    return df.iloc[:cut].copy(), df.iloc[cut:].copy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="laps.parquet")
    parser.add_argument("--model", choices=["lstm", "linear"], default="lstm")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    df = pd.read_parquet(args.data)
    df = encode(df)

    drivers, circuits = build_identity_vocab(df)
    print(f"identity vocab: {len(drivers)} drivers, {len(circuits)} circuits")
    df = add_identity_columns(df, drivers, circuits)

    train_pool = pd.DataFrame(df.loc[df["Year"] <= 2023])
    test_df = pd.DataFrame(df.loc[df["Year"] == 2024])
    if len(train_pool) == 0 or len(test_df) == 0:
        print("warning: train pool or 2024 test set is empty; check the data range")

    train_df, val_df = chronological_split(train_pool, val_fraction=0.2)

    X_tr, y_tr, scaler = make_windows(train_df, drivers, circuits, fit_scaler=True)
    X_val, y_val, _ = make_windows(val_df, drivers, circuits, scaler=scaler)
    X_te, y_te, _ = make_windows(test_df, drivers, circuits, scaler=scaler)

    if X_tr is None or y_tr is None:
        raise RuntimeError("no training windows produced; need stints with at least 5 valid laps")

    print(
        f"windows -- train: {len(X_tr)}, "
        f"val: {0 if X_val is None else len(X_val)}, "
        f"test: {0 if X_te is None else len(X_te)} "
        f"| feature_dim: {X_tr.shape[-1]}"
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    input_dim = X_tr.shape[-1]

    if args.model == "lstm":
        model = LSTMRegressor(input_dim).to(device)
    else:
        model = LinearBaseline(input_dim).to(device)

    train_loader = DataLoader(WindowDataset(X_tr, y_tr), batch_size=args.batch_size, shuffle=True)
    val_loader = None
    if X_val is not None and y_val is not None:
        val_loader = DataLoader(WindowDataset(X_val, y_val), batch_size=args.batch_size)
    test_loader = None
    if X_te is not None and y_te is not None:
        test_loader = DataLoader(WindowDataset(X_te, y_te), batch_size=args.batch_size)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = torch.nn.MSELoss()

    for epoch in range(args.epochs):
        tr_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        line = f"epoch {epoch + 1:02d} | train_loss {tr_loss:.4f}"
        if val_loader is not None:
            m = evaluate(model, val_loader, device)
            line += f" | val MAE {m['mae']:.3f} RMSE {m['rmse']:.3f} R2 {m['r2']:.3f}"
        print(line)

    if test_loader is not None:
        m = evaluate(model, test_loader, device)
        print(f"test  | MAE {m['mae']:.3f} RMSE {m['rmse']:.3f} R2 {m['r2']:.3f}")


if __name__ == "__main__":
    main()
