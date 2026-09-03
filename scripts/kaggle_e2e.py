#!/usr/bin/env python3
"""Kaggle end-to-end: pull a real physiological time series through the
legacy Kaggle API and run the full TSO pipeline on it.

Default dataset: MIT-BIH cardiac arrhythmia (RR-interval / heart-rate
variability — a genuinely chaotic, clinically important signal).
"""

import argparse
import json
import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tso.demo import run_full_analysis

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
OUT = os.path.join(ROOT, "output", "kaggle_ecg")


def kaggle_download(dataset_ref, dest):
    """Legacy Kaggle API via the authenticated `kaggle` CLI."""
    os.makedirs(dest, exist_ok=True)
    print(f"  downloading {dataset_ref} (legacy API) ...")
    proc = subprocess.run(
        ["kaggle", "datasets", "download", "-d", dataset_ref,
         "-p", dest, "--unzip"],
        capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"kaggle download failed: {proc.stderr[-800:]}")
    print(f"  -> {dataset_ref} downloaded")


def load_hrv_signal(csv_path, max_beats=2500):
    """Extract rr_prev (time between consecutive heartbeats) for the record
    with the most beats — a continuous chaotic physiological series."""
    import pandas as pd
    df = pd.read_csv(csv_path)
    rec = df.groupby("record").size().idxmax()
    seg = df[df["record"] == rec]["rr_prev"].dropna().to_numpy()[:max_beats]
    return seg, rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="sumit042004/cardiac-arrhythmia-ecg-dataset-mit-bih")
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--no-neural", action="store_true", help="skip neural field training")
    args = ap.parse_args()

    print("=" * 70)
    print("TSO Kaggle end-to-end: real data -> phase space -> Koopman -> artwork")
    print("=" * 70)

    # 1. fetch via the legacy Kaggle API
    if not os.path.exists(os.path.join(DATA, "Cardiac_arrhythmia_dataset.csv")):
        kaggle_download(args.dataset, DATA)
    csv_path = os.path.join(DATA, "Cardiac_arrhythmia_dataset.csv")

    # 2. build the signal
    signal, record = load_hrv_signal(csv_path)
    print(f"  signal: RR-interval series of MIT-BIH record {record} "
          f"({len(signal)} heartbeats)")

    # 3. run the full TSO pipeline + artwork
    results = run_full_analysis(signal, f"mitbih-{record}-rr", args.out,
                                train_neural=not args.no_neural,
                                nf_iters=1600)

    print("\n" + "=" * 70)
    print("RESULTS — Kaggle ECG (heart-rate variability)")
    print("=" * 70)
    print(f"  Takens:   tau={results['tau']}, dim={results['dim']}")
    print(f"  eDMD-RFF: RMSE={results['edmd']['rmse']:.4f}  "
          f"skill over persistence={results['edmd']['skill_pct']:+.1f}%")
    print(f"  plain DMD: RMSE={results['dmd']['rmse']:.4f}  "
          f"skill={results['dmd']['skill_pct']:+.1f}%")
    nf = results.get("neural_field", {})
    if nf:
        print(f"  neural field: loss={nf['final_loss']:.5f}, "
              f"short-horizon corr={nf['short_horizon_corr']:.3f}")
    print(f"\n  artwork written to: {args.out}/")
    for f in results["files"]:
        print(f"    - {f}")


if __name__ == "__main__":
    main()
