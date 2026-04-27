"""Fetch F1 lap data via FastF1 and build a per-lap dataset with a tire-degradation target.

Usage:
    python data.py --years 2019 2020 2021 2022 2023 2024 --out laps.parquet
    python data.py --years 2023 --limit 2 --out smoke.parquet  # quick smoke test

The first run will download a lot of data into ./fastf1_cache. Subsequent runs are fast.

Note on TyreSurfaceTemp: the project proposal listed tire surface temperature as a
feature, but FastF1's public data (laps frame + standard telemetry) does not expose
per-lap tire temperatures. TrackTemp and AirTemp from the weather feed are the closest
thermal signals available and are included below.
"""

import argparse
import os

import numpy as np
import pandas as pd
import fastf1  # type: ignore[import-not-found]

CACHE_DIR = "fastf1_cache"
os.makedirs(CACHE_DIR, exist_ok=True)
fastf1.Cache.enable_cache(CACHE_DIR)

# Slick compounds only for the baseline; intermediate/wet add a lot of variance.
COMPOUNDS = ["SOFT", "MEDIUM", "HARD"]

# Rough fuel correction: F1 cars get faster as fuel burns off.
# Approx 0.03 s/lap of advantage per lap remaining (a standard ballpark figure).
FUEL_PER_LAP_S = 0.03

# Physical range for the degradation target. Below: track evolution / noise on the
# fresh-tire reference. Above: traffic, late SC restarts, and other anomalies that
# slipped past the IsAccurate filter. Anything outside this band is dropped during
# data prep so train.py does not need its own outlier clip.
DEG_MIN_S = -1.0
DEG_MAX_S = 10.0


def fetch_session(year: int, round_number: int):
    session = fastf1.get_session(year, round_number, "R")
    session.load(laps=True, telemetry=False, weather=True, messages=False)
    return session


def lap_records(session, year: int, round_number: int):
    laps = session.laps.copy()
    if laps.empty:
        return pd.DataFrame()

    laps = laps.loc[
        laps["IsAccurate"].fillna(False)
        & laps["LapTime"].notna()
        & laps["PitInTime"].isna()
        & laps["PitOutTime"].isna()
        & laps["Compound"].isin(COMPOUNDS)
    ].copy()
    if laps.empty:
        return laps

    laps["LapTimeSec"] = laps["LapTime"].dt.total_seconds()
    total_laps = laps["LapNumber"].max()
    laps["FuelCorrLap"] = laps["LapTimeSec"] - (total_laps - laps["LapNumber"]) * FUEL_PER_LAP_S

    weather = session.weather_data
    if weather is not None and not weather.empty:
        weather = weather[["Time", "AirTemp", "TrackTemp", "Humidity", "Rainfall"]].sort_values("Time")
        laps = laps.sort_values("LapStartTime")
        laps = pd.merge_asof(
            laps,
            weather,
            left_on="LapStartTime",
            right_on="Time",
            direction="nearest",
        )
    else:
        for col in ["AirTemp", "TrackTemp", "Humidity", "Rainfall"]:
            laps[col] = np.nan

    laps["Year"] = year
    laps["Round"] = round_number
    laps["Circuit"] = session.event["EventName"]

    cols = [
        "Year",
        "Round",
        "Circuit",
        "Driver",
        "Stint",
        "LapNumber",
        "Compound",
        "TyreLife",
        "LapTimeSec",
        "FuelCorrLap",
        "AirTemp",
        "TrackTemp",
        "Humidity",
        "Rainfall",
    ]
    return pd.DataFrame(laps[cols])


def compute_degradation_target(df: pd.DataFrame) -> pd.DataFrame:
    """Target = fuel-corrected lap time minus the driver's fastest lap on fresh tires
    (TyreLife <= 3) within the same stint. The fresh-tire reference is computed only
    over rows that survived the IsAccurate green-flag filter in lap_records()."""
    df = df.copy()
    stint_keys = ["Year", "Round", "Driver", "Stint"]
    fresh = (
        df.loc[df["TyreLife"] <= 3]
        .groupby(stint_keys, as_index=False)["FuelCorrLap"]
        .min()
        .rename(columns={"FuelCorrLap": "StintBase"})
    )
    df = df.merge(fresh, on=stint_keys, how="left")
    df = df.dropna(subset=["StintBase"])
    df["DegSec"] = df["FuelCorrLap"] - df["StintBase"]

    before = len(df)
    df = df.loc[(df["DegSec"] >= DEG_MIN_S) & (df["DegSec"] <= DEG_MAX_S)].copy()
    print(f"DegSec range filter [{DEG_MIN_S}, {DEG_MAX_S}]: kept {len(df)} of {before} rows")
    return df


def build_dataset(years, schedule_limit=None) -> pd.DataFrame:
    rows = []
    for year in years:
        try:
            schedule = fastf1.get_event_schedule(year, include_testing=False)
        except Exception as e:
            print(f"failed to load schedule for {year}: {e}")
            continue
        if schedule_limit:
            schedule = schedule.head(schedule_limit)
        for _, ev in schedule.iterrows():
            rnd = int(ev["RoundNumber"])
            try:
                session = fetch_session(year, rnd)
                rows.append(lap_records(session, year, rnd))
                print(f"loaded {year} R{rnd} {ev['EventName']}")
            except Exception as e:
                print(f"skip {year} R{rnd} {ev['EventName']}: {e}")
    if not rows:
        raise RuntimeError("no race data was loaded")
    df = pd.concat(rows, ignore_index=True)
    return compute_degradation_target(df)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--years", nargs="+", type=int, default=[2019, 2020, 2021, 2022, 2023, 2024])
    parser.add_argument("--limit", type=int, default=None,
                        help="optional cap on races per year (useful for quick smoke tests)")
    parser.add_argument("--out", default="laps.parquet")
    args = parser.parse_args()

    df = build_dataset(args.years, schedule_limit=args.limit)
    df.to_parquet(args.out)
    print(f"saved {len(df)} lap rows to {args.out}")


if __name__ == "__main__":
    main()
