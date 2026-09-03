#!/usr/bin/env python3
"""Fetch v11 outputs from the Modal volume into output/kaggle_kernel_v11/.

Usage: python scripts/fetch_v11.py [--seeds 0,1,2]
"""
import argparse
import os
import shutil

import modal

DEST = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "output", "kaggle_kernel_v11")


def fetch(seed: int) -> str:
    vol = modal.Volume.from_name("tso-v11-outputs")
    src = f"/out-seed{seed}"
    out = os.path.join(DEST, f"seed{seed}")
    os.makedirs(out, exist_ok=True)
    try:
        entries = vol.iterdir(src)
    except Exception:
        return f"seed{seed}: no outputs yet"
    n = 0
    for e in entries:
        if getattr(e, "size", None) is not None and e.size > 0:
            with open(os.path.join(out, os.path.basename(e.path)), "wb") as fh:
                vol.read_file_into_fileobj(e.path, fh)
            n += 1
    return f"seed{seed}: {n} files -> {out}"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0,1,2")
    args = ap.parse_args()
    for s in map(int, args.seeds.split(",")):
        print(fetch(s))