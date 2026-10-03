"""Small internal helpers: an optional tqdm progress wrapper."""
from __future__ import annotations

import logging

logger = logging.getLogger("jevu")


def progress(iterable, *, total=None, desc=None, enabled=True):
    """Wrap ``iterable`` in a tqdm progress bar if enabled and tqdm is installed.

    Falls back to the plain iterable when disabled or when tqdm is unavailable, so the
    library has no hard dependency on tqdm.
    """
    if not enabled:
        return iterable
    try:
        from tqdm.auto import tqdm
    except Exception:  # pragma: no cover - tqdm optional
        logger.debug("tqdm not installed; progress bars disabled")
        return iterable
    return tqdm(iterable, total=total, desc=desc)
