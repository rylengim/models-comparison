"""Shared operational pacing; never retries or clears a billing block.

Run from the repository root: python reports/run_paced.py benchmark --resume ...
The package and request payloads are unchanged. Wall time includes pacing waits.
"""
import fcntl
import json
import math
import os
import sys
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from freshqa.client import CostUnknownError, OpenRouterClient, OpenRouterError
from freshqa.__main__ import main as cli_main


class RequestGate:
    """Serialize start permissions across local processes sharing one file.

    A separate BLOCKED marker lets an in-flight request stop waiting workers
    immediately, without acquiring their flock. Already admitted requests may
    finish. State uses this host's monotonic clock; stale/invalid state fails
    closed and must be reconciled manually, just like the persistent marker.
    """

    def __init__(self, path, interval_s=2.5):
        if not str(path).strip():
            raise ValueError("FRESHQA_GATE_PATH must be a nonempty file path")
        try:
            self.path = Path(path).expanduser().resolve()
        except (ValueError, OSError):
            raise ValueError("FRESHQA_GATE_PATH must be a valid file path") from None
        if self.path.is_dir():
            raise ValueError("FRESHQA_GATE_PATH must name a file, not a directory")
        if isinstance(interval_s, bool) or not isinstance(interval_s, (int, float)) or not math.isfinite(interval_s) or interval_s <= 0:
            raise ValueError("FRESHQA_GATE_INTERVAL_S must be a positive finite number")
        self.interval_s = float(interval_s)
        self.blocked_path = self.path.with_name(self.path.name + ".BLOCKED")

    @classmethod
    def from_env(cls):
        default = Path(__file__).resolve().parents[1] / "tmp" / "openrouter-request-gate.timestamp"
        try:
            interval = float(os.getenv("FRESHQA_GATE_INTERVAL_S", "2.5"))
        except ValueError:
            raise ValueError("FRESHQA_GATE_INTERVAL_S must be a positive finite number") from None
        return cls(os.getenv("FRESHQA_GATE_PATH", str(default)), interval)

    def _check_blocked(self):
        if self.blocked_path.exists():
            raise OpenRouterError("Shared request gate BLOCKED after an unknown charge; no request sent. Reconcile billing before continuing.")

    def wait(self):
        self._check_blocked()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a+", encoding="ascii") as state:
            fcntl.flock(state, fcntl.LOCK_EX)
            self._check_blocked()
            state.seek(0)
            raw = state.read()
            now = time.monotonic()
            try:
                previous = float(raw) if raw.strip() else 0.0
                if not math.isfinite(previous) or previous < 0 or previous > now:
                    raise ValueError
            except ValueError:
                raise OpenRouterError("Invalid request gate state; no request sent. Reconcile the shared timestamp file.") from None
            time.sleep(max(0.0, self.interval_s - (now - previous)))
            self._check_blocked()
            state.seek(0)
            state.truncate()
            state.write(f"{time.monotonic():.9f}\n")
            state.flush()
            os.fsync(state.fileno())

    def call(self, completion, *args, **kwargs):
        self.wait()
        self._check_blocked()
        try:
            return completion(*args, **kwargs)
        except CostUnknownError:
            # Do not wait for flock: another worker may hold it while sleeping.
            with self.blocked_path.open("a", encoding="ascii") as marker:
                marker.flush()
                os.fsync(marker.fileno())
            raise


def _exception_chain(error):
    """Only exception classes and the last three module/function frame names."""
    chain, seen = [], set()
    while isinstance(error, BaseException) and id(error) not in seen and len(chain) < 8:
        seen.add(id(error))
        frames = []
        traceback = error.__traceback__
        while traceback is not None:
            frame = traceback.tb_frame
            frames.append({"module": frame.f_globals.get("__name__", "unknown"),
                           "function": frame.f_code.co_name})
            frames = frames[-3:]
            traceback = traceback.tb_next
        chain.append({"type": type(error).__name__, "frames": frames})
        error = error.__cause__ or error.__context__
    return chain


def main(argv=None):
    gate = RequestGate.from_env()
    original = OpenRouterClient.complete
    original_post = requests.post

    def paced(self, **kwargs):
        return gate.call(original, self, **kwargs)

    def diagnosed_post(*args, **kwargs):
        started = time.monotonic()
        try:
            return original_post(*args, **kwargs)
        except requests.RequestException as error:
            elapsed = time.monotonic() - started
            context = error.__cause__ or error.__context__
            reason = getattr(context, "reason", context)
            reason_type = type(reason).__name__ if isinstance(reason, BaseException) else "unknown"
            # Exception text, args and request details can contain credentials.
            try:
                print(f"POST transport failure: type={type(error).__name__} reason_type={reason_type} elapsed_s={elapsed:.6f}",
                      file=sys.stderr, flush=True)
                print("POST exception chain: " + json.dumps(_exception_chain(error)),
                      file=sys.stderr, flush=True)
            except OSError:
                pass  # Diagnostic output must not replace the transport error.
            raise

    OpenRouterClient.complete = paced
    requests.post = diagnosed_post
    try:
        return cli_main(argv)
    finally:
        OpenRouterClient.complete = original
        requests.post = original_post

if __name__ == "__main__":
    raise SystemExit(main())
