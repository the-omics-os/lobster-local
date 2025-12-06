"""Shared utilities for queue persistence with robust locking and atomic writes."""

from __future__ import annotations

import json
import os
import tempfile
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterable

__all__ = [
    "InterProcessFileLock",
    "queue_file_lock",
    "atomic_write_json",
    "atomic_write_jsonl",
    "backups_enabled",
]


class InterProcessFileLock:
    """File-based lock compatible with multiple processes.

    Uses ``fcntl.flock`` on POSIX platforms and ``msvcrt.locking`` on Windows.
    The lock file contains no data – it simply acts as a synchronization primitive.
    """

    def __init__(self, lock_path: Path):
        self.lock_path = Path(lock_path)
        self._handle = None

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()

    def acquire(self):
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = open(self.lock_path, "a+")

        if os.name == "nt":  # pragma: no cover - Windows specific
            import msvcrt

            msvcrt.locking(self._handle.fileno(), msvcrt.LK_LOCK, 1)
        else:  # POSIX
            import fcntl

            fcntl.flock(self._handle.fileno(), fcntl.LOCK_EX)

    def release(self):
        if not self._handle:
            return

        if os.name == "nt":  # pragma: no cover - Windows specific
            import msvcrt

            msvcrt.locking(self._handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(self._handle.fileno(), fcntl.LOCK_UN)

        self._handle.close()
        self._handle = None


@contextmanager
def queue_file_lock(thread_lock: threading.Lock, lock_path: Path):
    """Acquire both thread and process safe locks for queue operations."""

    with thread_lock:
        with InterProcessFileLock(lock_path):
            yield


def atomic_write_jsonl(
    target_path: Path,
    entries: Iterable[Any],
    serializer: Callable[[Any], dict],
) -> None:
    """Persist entries to ``target_path`` atomically as JSON Lines."""

    target_path = Path(target_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)

    temp_fd, temp_path = tempfile.mkstemp(
        prefix=f"{target_path.name}.", suffix=".tmp", dir=target_path.parent
    )
    temp_file = Path(temp_path)

    try:
        with os.fdopen(temp_fd, "w", encoding="utf-8") as handle:
            for entry in entries:
                json.dump(serializer(entry), handle, default=str)
                handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())

        os.replace(temp_file, target_path)

    except Exception:
        if temp_file.exists():
            temp_file.unlink()
        raise


def atomic_write_json(
    target_path: Path,
    data: dict,
    indent: int = 2,
) -> None:
    """Persist a dict to ``target_path`` atomically as JSON.

    Uses the same atomic write pattern as ``atomic_write_jsonl``:
    write to temp file, fsync, then atomic rename.
    """

    target_path = Path(target_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)

    temp_fd, temp_path = tempfile.mkstemp(
        prefix=f"{target_path.name}.", suffix=".tmp", dir=target_path.parent
    )
    temp_file = Path(temp_path)

    try:
        with os.fdopen(temp_fd, "w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=indent, default=str)
            handle.flush()
            os.fsync(handle.fileno())

        os.replace(temp_file, target_path)

    except Exception:
        if temp_file.exists():
            temp_file.unlink()
        raise


def backups_enabled() -> bool:
    """Return True only when explicitly enabled via env var."""

    value = os.getenv("LOBSTER_ENABLE_QUEUE_BACKUPS")
    if value is None:
        return False

    return value.lower() in {"1", "true", "yes"}
