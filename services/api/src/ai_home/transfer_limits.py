"""In-process probe admission budgets, independent of credits and job leases."""
from collections import deque
import threading
import time


class TransferProbeLimiter:
    def __init__(self, clock=time.monotonic):
        self.clock = clock
        self.lock = threading.Lock()
        self.events = deque()

    def admit(self, node_id, reserved_bytes):
        """At most 10 requests/node, 1000 total and 64 MiB per rolling minute.

        Uploads reserve their full maximum before reading. Failed requests consume
        their reservation. Scope is one API process, not a distributed quota.
        """
        with self.lock:
            now = self.clock()
            while self.events and self.events[0][0] <= now - 60:
                self.events.popleft()
            if (sum(event[1] == node_id for event in self.events) >= 10
                    or len(self.events) >= 1000
                    or sum(event[2] for event in self.events) + reserved_bytes > 64 * 1048576):
                return False
            self.events.append((now, node_id, reserved_bytes))
            return True
