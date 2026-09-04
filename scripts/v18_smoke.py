#!/usr/bin/env python3
"""Local smoke test for the v18 matched-compute experiment (tiny corpus)."""
import json
import os
import pickle
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "kaggle_kernel_tso"))
import main  # noqa: E402

bundle = pickle.load(open(os.path.join(ROOT, "output", "modal_corpus",
                                       "corpus40.pkl"), "rb"))
samples, meta = bundle["samples"], bundle["meta"]
want = ["sunspots", "airline", "lorenz-x"]
sel = [(n, p) for n, p in samples if n in want]
selm = [m for m in meta if m["name"] in want]
battery = main.load_dynamics_battery(n_series=4, n=600, seed=11)
print("corpus:", len(sel), "battery:", len(battery), flush=True)

src = open(os.path.join(ROOT, "kaggle_kernel_tso", "driver_main.py")).read()
src = src.split('if __name__ == "__main__":')[0]
ns = vars(main)
exec(compile(src, "driver_main.py", "exec"), ns)

os.environ["V18_TSO_ITERS"] = "400"
os.environ["V18_MIN_MATCHED"] = "40"
os.environ["V18_TOTAL_ITERS"] = "60"
os.environ["V18_EVAL_SAMPLES"] = "8"
outdir = os.path.abspath(os.path.join(ROOT, "output", "v18_smoke"))
os.makedirs(outdir, exist_ok=True)
r = ns["run_matched_compute"](sel, selm, battery, outdir, "cpu")
print("SUMMARY:", json.dumps(r["summary"]), flush=True)
print("H2H:", json.dumps(r["h2h"]), flush=True)
print("SMOKE OK", flush=True)