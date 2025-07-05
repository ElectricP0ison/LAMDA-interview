import time
from typing import List

class Timer:
    """Simple context manager for timing code blocks."""
    def __init__(self, storage: List[float]):
        self.storage = storage
        self._start = None

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._start is not None:
            self.storage.append(time.perf_counter() - self._start)
        self._start = None
