"""Bounded observation of durable jobs; never cancels the jobs being observed."""
import threading
import time
import httpx


def observe(operation, *, complete_state, wait_seconds, poll_seconds=5, cancel=None):
    if (type(wait_seconds) is not int or not 1 <= wait_seconds <= 3600
            or type(poll_seconds) is not int or not 1 <= poll_seconds <= 60):
        raise ValueError("Invalid coordinator wait bounds")
    cancellation = cancel if cancel is not None else threading.Event()
    deadline = time.monotonic() + wait_seconds
    last, attempts, transport_errors = None, 0, 0
    while not cancellation.is_set():
        if time.monotonic() >= deadline:
            break
        attempts += 1
        try:
            last = operation()
        except httpx.TransportError:
            # A response may be lost after creation; replay the same immutable plan.
            # HTTP authentication/policy/status failures are deliberately not retried.
            transport_errors += 1
        else:
            if last["state"] == complete_state:
                return {**last, "attempts": attempts, "transport_errors": transport_errors}
        cancellation.wait(max(0, min(poll_seconds, deadline - time.monotonic())))
    return {"state": "stopped" if cancellation.is_set() else "timed_out", "last_observation": last,
            "attempts": attempts, "transport_errors": transport_errors,
            "jobs_cancelled": False}

