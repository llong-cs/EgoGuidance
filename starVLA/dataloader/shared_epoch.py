"""Small shared-memory epoch primitive for persistent DataLoader workers."""

import multiprocessing as mp
from numbers import Integral


class SharedEpoch:
    """Publish an epoch integer from the trainer process to worker copies.

    PyTorch persistent workers keep the dataset object they received when the
    pool was created.  Assigning ``dataset.epoch`` in the trainer therefore does
    not reach those copies.  This synchronized scalar is intentionally the only
    shared state: samples still derive all randomness from ``(epoch, index,
    seed)`` and workers never advance it themselves.
    """

    def __init__(self, epoch: int = 0):
        self._value = mp.Value("q", self._validate(epoch))

    @staticmethod
    def _validate(epoch: int) -> int:
        if isinstance(epoch, bool) or not isinstance(epoch, Integral) or int(epoch) < 0:
            raise ValueError(f"epoch must be a non-negative integer, got {epoch!r}.")
        return int(epoch)

    def get(self) -> int:
        with self._value.get_lock():
            return int(self._value.value)

    def set(self, epoch: int) -> None:
        value = self._validate(epoch)
        with self._value.get_lock():
            self._value.value = value
