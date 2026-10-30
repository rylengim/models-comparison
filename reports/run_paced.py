"""Operational pacing for the recorded run; never retries a completion.

Run from the repository root: python reports/run_paced.py benchmark --resume ...
The package and request payloads are unchanged. Wall time includes pacing waits.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from freshqa.client import OpenRouterClient
from freshqa.__main__ import main

original = OpenRouterClient.complete
previous_start = 0.0


def paced(self, **kwargs):
    global previous_start
    time.sleep(max(0.0, 3.5 - (time.monotonic() - previous_start)))
    previous_start = time.monotonic()
    return original(self, **kwargs)


OpenRouterClient.complete = paced

if __name__ == "__main__":
    raise SystemExit(main())
