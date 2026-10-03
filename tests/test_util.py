import logging

from jevu._util import progress


def test_progress_disabled_passthrough():
    assert list(progress([1, 2, 3], enabled=False)) == [1, 2, 3]


def test_progress_enabled_iterates():
    # works whether or not tqdm is installed (graceful fallback)
    assert list(progress(iter([1, 2]), total=2, desc="x", enabled=True)) == [1, 2]


def test_jevu_logger_has_nullhandler():
    import jevu  # noqa: F401 - import side effect attaches the handler
    handlers = logging.getLogger("jevu").handlers
    assert any(isinstance(h, logging.NullHandler) for h in handlers)
