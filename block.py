import time
import torch
from functools import wraps


class Block(torch.nn.Module):
    """Base class for all pipeline blocks. Auto-times every forward() call."""

    def __init__(self):
        super().__init__()
        self._measure_runtime = True
        self._last_runtime    = None

    @staticmethod
    def measure_runtime_decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            if not getattr(self, "_measure_runtime", False):
                return func(self, *args, **kwargs)
            start = time.perf_counter()
            out   = func(self, *args, **kwargs)
            self._last_runtime = time.perf_counter() - start
            return out
        return wrapper

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if "forward" in cls.__dict__:
            cls.forward = Block.measure_runtime_decorator(cls.forward)

    def enable_runtime_measure(self, enable=True):
        self._measure_runtime = enable

    def runtime_measure(self):
        """Return wall-clock seconds of last forward() call."""
        return self._last_runtime
