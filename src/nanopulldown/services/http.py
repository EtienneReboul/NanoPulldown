"""
nanopulldown.services.http
==========================
Polite access to the public web services (InterPro, ScanProsite): a shared,
ramped rate limiter per service, Retry-After-aware retries, and a
process-wide circuit breaker.
"""
from __future__ import annotations

import sys
import threading
import time

import requests

# ── polite access to the public web services ───────────────────────────────
# Neither InterPro nor ScanProsite publishes a numeric rate limit. InterPro's
# only guidance (NAR 2025, D444): tens of thousands of requests in a very short
# period can make the API unavailable for everyone; "start querying our API
# gradually, then ramp up". So every outgoing request goes through a shared,
# ramped rate limiter; 429/5xx honour Retry-After and slow the limiter down;
# repeated throttling trips a circuit breaker that aborts the run (the
# per-sequence cache keeps all progress, rerun later).

class RateLimitAbort(Exception):
    """Circuit breaker tripped: the service keeps throttling us."""


STOP = threading.Event()
_throttle_events = 0
_throttle_lock = threading.Lock()


class RateLimiter:
    """Thread-safe minimum spacing between requests, ramping from 10% to 100%
    of the target rate over `ramp_s` seconds after the first request."""

    def __init__(self, per_s: float, ramp_s: float = 0.0, max_throttle_events: int = 10):
        self.base = 1.0 / max(per_s, 1e-6)
        self.max_events = max_throttle_events
        self.ramp_s, self.slow, self.next, self.t0 = ramp_s, 1.0, 0.0, None
        self.lock = threading.Lock()

    def wait(self) -> None:
        if STOP.is_set():
            raise RateLimitAbort("aborted after repeated throttling")
        with self.lock:
            now = time.monotonic()
            self.t0 = self.t0 if self.t0 is not None else now
            ramp = min(1.0, (now - self.t0) / self.ramp_s) if self.ramp_s else 1.0
            interval = self.base * self.slow / (0.1 + 0.9 * ramp)
            at = max(now, self.next)
            self.next = at + interval
        time.sleep(max(0.0, at - now))

    def slow_down(self, factor: float = 2.0, cap: float = 8.0) -> None:
        with self.lock:
            self.slow = min(cap, self.slow * factor)



def http_get(limiter: RateLimiter, url: str, tries: int = 3, **kw):
    """requests.get through `limiter`. 429 / 502 / 503 / 504 => wait Retry-After
    (default 60 s x attempt), slow the limiter 2x, count a throttle event; too
    many events (limiter.max_events) trips STOP for the whole process."""
    global _throttle_events
    for attempt in range(1, tries + 1):
        limiter.wait()
        r = requests.get(url, **kw)
        if r.status_code not in (429, 502, 503, 504):
            return r
        with _throttle_lock:
            _throttle_events += 1
            if _throttle_events >= limiter.max_events:
                STOP.set()
        limiter.slow_down()
        if STOP.is_set():
            raise RateLimitAbort(f"HTTP {r.status_code} from {url.split('?')[0]}: too many throttle events")
        try:
            wait_s = float(r.headers.get("Retry-After", ""))
        except ValueError:
            wait_s = 60.0 * attempt
        print(f"  [throttle] HTTP {r.status_code} from {url.split('?')[0]} -- sleeping {wait_s:.0f}s, "
              f"rate now /{limiter.slow:g} (event {_throttle_events}/{limiter.max_events})",
              file=sys.stderr, flush=True)
        time.sleep(min(wait_s, 900))
    r.raise_for_status()
    return r
