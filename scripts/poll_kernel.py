#!/usr/bin/env python3
"""Resilient poller for a Kaggle kernel: retries through transient CLI
SSL/network errors, exits 0 when the kernel leaves RUNNING/QUEUED, exits 2
when the CLI hard-fails repeatedly.

Usage: python scripts/poll_kernel.py sehajrsingh/tso-v18-... [max_minutes]
"""
import subprocess
import sys
import time


def main():
    slug = sys.argv[1]
    max_min = float(sys.argv[2]) if len(sys.argv) > 2 else 120.0
    deadline = time.time() + max_min * 60
    consecutive_fail = 0
    while time.time() < deadline:
        try:
            r = subprocess.run(["kaggle", "kernels", "status", slug],
                               capture_output=True, text=True, timeout=60)
            out = (r.stdout + r.stderr).strip().splitlines()
            line = out[-1] if out else "(empty)"
            print(f"[{time.strftime('%H:%M', time.gmtime())}Z] {line}",
                  flush=True)
            consecutive_fail = 0
            if "RUNNING" in line or "QUEUED" in line:
                time.sleep(60)
                continue
            if "has status" in line:
                return 0  # done/error — caller decides
        except Exception as e:  # noqa: BLE001
            consecutive_fail += 1
            print(f"[{time.strftime('%H:%M', time.gmtime())}Z] cli blip "
                  f"({consecutive_fail}): {type(e).__name__}", flush=True)
            if consecutive_fail >= 5:
                return 2
            time.sleep(30)
    print("poll deadline reached, still running", flush=True)
    return 1


if __name__ == "__main__":
    sys.exit(main())