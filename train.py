"""Train the baseline LSTM, linear baseline, or XGBoost on the prepared lap dataset.

Usage:
    python train.py --data laps.parquet --model xgb
    python train.py --data laps.parquet --model lstm
    python train.py --data laps.parquet --model linear
    python train.py --data laps.parquet --model xgb --identity   # add driver/circuit one-hots
"""

import argparse
import copy
import math

import numpy as np
import pandas as pd
import torch
import xgboost as xgb
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

from model import LinearBaseline, LSTMRegressor  # type: ignore[import-not-found]

WINDOW = 8
COMPOUNDS = ["SOFT", "MEDIUM", "HARD"]
NUMERIC_FEATURES = ["TyreLife", "AirTemp", "TrackTemp", "Humidity", "Rainfall"]
DERIVED_FEATURES = [
    "TyreLifeSq",
    "SOFT_TyreLife",
    "MEDIUM_TyreLife",
    "HARD_TyreLife",
    "StintProgress",
    "RaceProgress",
]
STREET_CIRCUITS = {"Monaco", "Singapore", "Azerbaijan", "Saudi", "Miami", "Las Vegas"}
EARLY_STOP_PATIENCE = 7
GRAD_CLIP_NORM = 1.0


def is_street(circuit: str) -> bool:
    return any(kw in circuit for kw in STREET_CIRCUITS)


def encode(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in COMPOUNDS:
        df[f"is_{c}"] = (df["Compound"] == c).astype(float)
    for c in NUMERIC_FEATURES:
        coerced = pd.to_numeric(df[c], errors="coerce")
        df[c] = pd.Series(coerced, index=df.index).fillna(0.0).astype(float)
    return df


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Must be called after encode() so is_SOFT/is_MEDIUM/is_HARD already exist."""
    df = df.copy()
    df["TyreLifeSq"] = df["TyreLife"] ** 2
    df["SOFT_TyreLife"] = df["is_SOFT"] * df["TyreLife"]
    df["MEDIUM_TyreLife"] = df["is_MEDIUM"] * df["TyreLife"]
    df["HARD_TyreLife"] = df["is_HARD"] * df["TyreLife"]
    stint_max = (
        df.groupby(["Year", "Round", "Driver", "Stint"])["TyreLife"]
        .transform("max")
        .clip(lower=1)
    )
    df["StintProgress"] = df["TyreLife"] / stint_max
    race_total = (
        df.groupby(["Year", "Round"])["LapNumber"]
        .transform("max")
        .clip(lower=1)
    )
    df["RaceProgress"] = df["LapNumber"] / race_total
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


def feature_cols(drivers, circuits, use_identity: bool) -> list:
    cols = NUMERIC_FEATURES + DERIVED_FEATURES + [f"is_{c}" for c in COMPOUNDS]
    if use_identity:
        cols += [f"drv_{d}" for d in drivers] + [f"crc_{c}" for c in circuits]
    return cols


def compute_delta_target(df: pd.DataFrame) -> pd.DataFrame:
    """Replace DegSec with lap-to-lap ΔLapTime within each stint.

    Track evolution affects every lap equally, so taking the delta largely cancels
    it out — leaving a signal driven by tire wear rather than rubber build-up.
    """
    df = df.copy().sort_values(["Year", "Round", "Driver", "Stint", "LapNumber"])
    df["DeltaLapTime"] = df.groupby(["Year", "Round", "Driver", "Stint"])["LapTimeSec"].diff()
    df = df.dropna(subset=["DeltaLapTime"])
    before = len(df)
    df = df.loc[(df["DeltaLapTime"] >= -3.0) & (df["DeltaLapTime"] <= 5.0)].copy()
    print(f"delta target: kept {len(df):,} of {before:,} rows after outlier clip [-3, +5]s")
    df["DegSec"] = df["DeltaLapTime"]
    return df


def build_flat_features(df: pd.DataFrame, drivers, circuits, use_identity: bool):
    """Single-lap feature matrix for XGBoost (no windowing)."""
    cols = feature_cols(drivers, circuits, use_identity)
    X = df[cols].to_numpy(dtype=np.float32)
    y = df["DegSec"].to_numpy(dtype=np.float32)
    return X, y


def make_windows(
    df: pd.DataFrame,
    drivers,
    circuits,
    use_identity: bool = False,
    scaler: StandardScaler | None = None,
    fit_scaler: bool = False,
):
    n_scaled = len(NUMERIC_FEATURES) + len(DERIVED_FEATURES)
    cols = feature_cols(drivers, circuits, use_identity)

    X_list, y_list = [], []
    grouped = (
        df.sort_values(["Year", "Round", "Driver", "Stint", "LapNumber"])
        .groupby(["Year", "Round", "Driver", "Stint"], sort=False)
    )
    for _, g in grouped:
        if len(g) < WINDOW:
            continue
        feats = g[cols].to_numpy(dtype=np.float32)
        targets = g["DegSec"].to_numpy(dtype=np.float32)
        for i in range(WINDOW - 1, len(g)):
            X_list.append(feats[i - WINDOW + 1 : i + 1])
            y_list.append(targets[i])

    if not X_list:
        return None, None, scaler

    X = np.stack(X_list).astype(np.float32)
    y = np.array(y_list, dtype=np.float32)

    n_windows, win_len, _ = X.shape
    flat = X[:, :, :n_scaled].reshape(-1, n_scaled)
    if fit_scaler:
        scaler = StandardScaler().fit(flat)
    assert scaler is not None
    flat_scaled = scaler.transform(flat)
    X[:, :, :n_scaled] = np.asarray(flat_scaled, dtype=np.float32).reshape(n_windows, win_len, n_scaled)

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
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
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


def collect_window_meta(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    grouped = (
        df.sort_values(["Year", "Round", "Driver", "Stint", "LapNumber"])
        .groupby(["Year", "Round", "Driver", "Stint"], sort=False)
    )
    for _, g in grouped:
        if len(g) < WINDOW:
            continue
        for i in range(WINDOW - 1, len(g)):
            rows.append({
                "Compound":    g["Compound"].iloc[i],
                "CircuitType": "street" if is_street(g["Circuit"].iloc[i]) else "permanent",
            })
    return pd.DataFrame(rows)


def collect_flat_meta(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "Compound":    df["Compound"].values,
        "CircuitType": ["street" if is_street(c) else "permanent" for c in df["Circuit"]],
    })


def plot_predicted_vs_actual(
    preds: np.ndarray,
    y: np.ndarray,
    meta: pd.DataFrame,
    out_path: str = "predicted_vs_actual.png",
) -> None:
    import matplotlib.pyplot as plt
    import matplotlib.patheffects as pe
    from matplotlib.colors import LinearSegmentedColormap

    BG = "#0d0d0d"
    # clip axes to where 99% of data lives — removes empty upper-right void
    lo = min(float(np.percentile(y, 0.5)), float(np.percentile(preds, 0.5))) - 0.2
    hi = max(float(np.percentile(y, 99)), float(np.percentile(preds, 99))) + 0.3
    mask_clip = (y >= lo) & (y <= hi) & (preds >= lo) & (preds <= hi)
    y_c, p_c = y[mask_clip], preds[mask_clip]
    meta_c = meta.iloc[mask_clip].reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(7, 7))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    # ±1 s acceptable-error band
    band_x = np.linspace(lo, hi, 200)
    ax.fill_between(band_x, band_x - 1, band_x + 1,
                    color="#ffffff", alpha=0.06, zorder=1, label="±1 s band")

    # perfect-prediction diagonal
    ax.plot([lo, hi], [lo, hi], color="#666666", linewidth=1.2, linestyle="--", zorder=2)

    # hexbin density (shows where predictions cluster without overplotting)
    heat = LinearSegmentedColormap.from_list(
        "f1heat", ["#0d0d0d", "#1a1a2e", "#e94560", "#f5a623", "#ffffff"]
    )
    hb = ax.hexbin(y_c, p_c, gridsize=55, cmap=heat, mincnt=1, linewidths=0.1, zorder=3)
    cb = fig.colorbar(hb, ax=ax, pad=0.02)
    cb.set_label("lap count", color="#888888", fontsize=9)
    cb.ax.yaxis.set_tick_params(color="#888888")
    plt.setp(cb.ax.yaxis.get_ticklabels(), color="#888888", fontsize=8)
    cb.outline.set_edgecolor("#333333")

    # thin compound-coloured scatter on top for legend only
    COMPOUND_STYLE = {
        "SOFT":   ("#E8002D", "Soft"),
        "MEDIUM": ("#FFF200", "Medium"),
        "HARD":   ("#DDDDDD", "Hard"),
    }
    for compound, (color, label) in COMPOUND_STYLE.items():
        m = meta_c["Compound"].values == compound
        if m.sum() == 0:
            continue
        ax.scatter(y_c[m], p_c[m], c=color, s=5, alpha=0.2,
                   linewidths=0, zorder=4, label=label)

    mae  = mean_absolute_error(y, preds)
    rmse = np.sqrt(mean_squared_error(y, preds))
    r2   = r2_score(y, preds)
    ax.text(0.03, 0.97, f"MAE {mae:.3f} s     RMSE {rmse:.3f} s     R² {r2:.3f}",
            transform=ax.transAxes, va="top", ha="left", color="#cccccc", fontsize=10,
            path_effects=[pe.withStroke(linewidth=2, foreground=BG)])

    ax.set_xlabel("Actual DegSec (s)", color="#cccccc", fontsize=12)
    ax.set_ylabel("Predicted DegSec (s)", color="#cccccc", fontsize=12)
    ax.set_title("Predicted vs Actual: 2024 Test Set (XGBoost)",
                 color="#ffffff", fontsize=13, pad=12)
    ax.tick_params(colors="#888888")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333333")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")

    legend = ax.legend(framealpha=0.15, labelcolor="white", fontsize=9, loc="lower right")
    legend.get_frame().set_facecolor("#111111")

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"saved plot -> {out_path}")
    plt.close(fig)


def _breakdown_from_preds(preds: np.ndarray, y: np.ndarray, meta: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for compound in COMPOUNDS:
        mask = (meta["Compound"] == compound).values
        if mask.sum() < 10:
            continue
        rows.append({
            "Group": compound.capitalize(),
            "N":     int(mask.sum()),
            "MAE":   float(mean_absolute_error(y[mask], preds[mask])),
            "RMSE":  float(np.sqrt(mean_squared_error(y[mask], preds[mask]))),
            "R2":    float(r2_score(y[mask], preds[mask])),
        })
    for ct in ["street", "permanent"]:
        mask = (meta["CircuitType"] == ct).values
        if mask.sum() < 10:
            continue
        rows.append({
            "Group": ct.capitalize(),
            "N":     int(mask.sum()),
            "MAE":   float(mean_absolute_error(y[mask], preds[mask])),
            "RMSE":  float(np.sqrt(mean_squared_error(y[mask], preds[mask]))),
            "R2":    float(r2_score(y[mask], preds[mask])),
        })
    return pd.DataFrame(rows).set_index("Group")


@torch.no_grad()
def evaluate_breakdown(model, X: np.ndarray, y: np.ndarray, meta: pd.DataFrame, device) -> pd.DataFrame:
    model.eval()
    preds = model(torch.from_numpy(X).to(device)).cpu().numpy()
    return _breakdown_from_preds(preds, y, meta)


def chronological_split(df: pd.DataFrame, val_fraction: float = 0.2):
    df = df.sort_values(["Year", "Round"]).reset_index(drop=True)
    cut = int((1 - val_fraction) * len(df))
    return df.iloc[:cut].copy(), df.iloc[cut:].copy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="laps.parquet")
    parser.add_argument("--model", choices=["lstm", "linear", "xgb"], default="xgb")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--identity",
        action="store_true",
        help="include driver/circuit one-hot features",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="save predicted-vs-actual scatter plot after evaluation",
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    df = pd.read_parquet(args.data)
    df = encode(df)
    df = add_engineered_features(df)

    drivers, circuits = build_identity_vocab(df)
    print(f"identity vocab: {len(drivers)} drivers, {len(circuits)} circuits")
    if args.identity:
        df = add_identity_columns(df, drivers, circuits)

    train_df = pd.DataFrame(df.loc[df["Year"] <= 2022])
    val_df   = pd.DataFrame(df.loc[df["Year"] == 2023])
    test_df  = pd.DataFrame(df.loc[df["Year"] == 2024])
    if len(train_df) == 0 or len(val_df) == 0 or len(test_df) == 0:
        print("warning: one or more splits are empty; check the data range")
    print(f"split — train: {len(train_df):,} rows (≤2022)  val: {len(val_df):,} rows (2023)  test: {len(test_df):,} rows (2024)")

    # ── XGBoost path (no windowing) ───────────────────────────────────────────
    if args.model == "xgb":
        X_tr, y_tr = build_flat_features(train_df, drivers, circuits, args.identity)
        X_val, y_val = build_flat_features(val_df, drivers, circuits, args.identity)
        X_te, y_te = build_flat_features(test_df, drivers, circuits, args.identity)

        n_id = (len(drivers) + len(circuits)) if args.identity else 0
        print(
            f"samples — train: {len(X_tr):,}  val: {len(X_val):,}  test: {len(X_te):,}"
            f"  |  feature_dim: {X_tr.shape[1]}"
            f"  ({len(NUMERIC_FEATURES)} raw + {len(DERIVED_FEATURES)} engineered + 3 compound"
            + (f" + {n_id} identity" if args.identity else "") + ")"
        )

        booster = xgb.XGBRegressor(
            n_estimators=2000,
            learning_rate=0.02,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=20,
            gamma=0.1,
            reg_alpha=0.05,
            reg_lambda=1.0,
            early_stopping_rounds=50,
            eval_metric="rmse",
            random_state=args.seed,
            n_jobs=-1,
        )
        booster.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            verbose=100,
        )

        val_preds = booster.predict(X_val)
        val_m = {
            "mae":  float(mean_absolute_error(y_val, val_preds)),
            "rmse": float(np.sqrt(mean_squared_error(y_val, val_preds))),
            "r2":   float(r2_score(y_val, val_preds)),
        }
        print(f"\nval (2023)  — MAE {val_m['mae']:.3f}  RMSE {val_m['rmse']:.3f}  R2 {val_m['r2']:.3f}")

        te_preds = booster.predict(X_te)
        te_m = {
            "mae":  float(mean_absolute_error(y_te, te_preds)),
            "rmse": float(np.sqrt(mean_squared_error(y_te, te_preds))),
            "r2":   float(r2_score(y_te, te_preds)),
        }
        print(f"test (2024) — MAE {te_m['mae']:.3f}  RMSE {te_m['rmse']:.3f}  R2 {te_m['r2']:.3f}")

        te_meta = collect_flat_meta(test_df)
        print("\ntest (2024) — breakdown")
        bd = _breakdown_from_preds(te_preds, y_te, te_meta)
        print(bd.to_string(float_format=lambda x: f"{x:.3f}"))

        cols = feature_cols(drivers, circuits, args.identity)
        imp = pd.Series(booster.feature_importances_, index=cols).sort_values(ascending=False)
        print("\ntop-10 feature importances:")
        print(imp.head(10).to_string(float_format=lambda x: f"{x:.4f}"))

        if args.plot:
            plot_predicted_vs_actual(te_preds, y_te, te_meta)
        return

    # ── LSTM / Linear path ────────────────────────────────────────────────────
    X_tr, y_tr, scaler = make_windows(train_df, drivers, circuits, use_identity=args.identity, fit_scaler=True)
    X_val, y_val, _    = make_windows(val_df,   drivers, circuits, use_identity=args.identity, scaler=scaler)
    X_te,  y_te,  _    = make_windows(test_df,  drivers, circuits, use_identity=args.identity, scaler=scaler)
    te_meta = collect_window_meta(test_df)

    if X_tr is None or y_tr is None:
        raise RuntimeError("no training windows produced; need stints with at least 8 valid laps")

    n_id = (len(drivers) + len(circuits)) if args.identity else 0
    print(
        f"windows — train: {len(X_tr):,}  val: {0 if X_val is None else len(X_val):,}  test: {0 if X_te is None else len(X_te):,}"
        f"  |  feature_dim: {X_tr.shape[-1]}"
        f"  ({len(NUMERIC_FEATURES)} raw + {len(DERIVED_FEATURES)} engineered + 3 compound"
        + (f" + {n_id} identity" if args.identity else "") + ")"
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    input_dim = X_tr.shape[-1]

    if args.model == "lstm":
        model = LSTMRegressor(input_dim).to(device)
    else:
        model = LinearBaseline(input_dim).to(device)

    print(f"model: {model.__class__.__name__}  params: {sum(p.numel() for p in model.parameters()):,}")

    train_loader = DataLoader(WindowDataset(X_tr, y_tr), batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(WindowDataset(X_val, y_val), batch_size=args.batch_size) if X_val is not None and y_val is not None else None
    test_loader  = DataLoader(WindowDataset(X_te,  y_te),  batch_size=args.batch_size) if X_te  is not None and y_te  is not None else None

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn   = torch.nn.MSELoss()

    best_val_rmse  = math.inf
    best_state     = None
    patience_counter = 0

    for epoch in range(args.epochs):
        tr_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        line = f"epoch {epoch + 1:02d} | train_loss {tr_loss:.4f}"

        if val_loader is not None:
            m = evaluate(model, val_loader, device)
            line += f" | val MAE {m['mae']:.3f} RMSE {m['rmse']:.3f} R2 {m['r2']:.3f}"

            if m["rmse"] < best_val_rmse:
                best_val_rmse    = m["rmse"]
                best_state       = copy.deepcopy(model.state_dict())
                patience_counter = 0
                line            += "  *"
            else:
                patience_counter += 1
                if patience_counter >= EARLY_STOP_PATIENCE:
                    print(line)
                    print(f"\nearly stopping at epoch {epoch + 1} (no val RMSE improvement for {EARLY_STOP_PATIENCE} epochs)")
                    break

        print(line)

    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"\nrestored best checkpoint (val RMSE {best_val_rmse:.3f} s)")

    if test_loader is not None:
        m = evaluate(model, test_loader, device)
        print(f"\ntest (2024) — overall")
        print(f"  MAE {m['mae']:.3f}  RMSE {m['rmse']:.3f}  R2 {m['r2']:.3f}")
        if X_te is not None and y_te is not None:
            print("\ntest (2024) — breakdown")
            bd = evaluate_breakdown(model, X_te, y_te, te_meta, device)
            print(bd.to_string(float_format=lambda x: f"{x:.3f}"))


if __name__ == "__main__":
    main()
