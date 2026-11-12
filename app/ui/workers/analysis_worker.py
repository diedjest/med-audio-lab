"""Generic one-shot background task runner."""

from __future__ import annotations

import inspect
import threading
import traceback
from typing import Any, Callable

from PySide6.QtCore import QObject, QThread, Signal

#: Parameter names the worker fills in when the task function declares them.
CANCEL_PARAM = "cancel_event"
PROGRESS_PARAM = "progress_callback"


class TaskCancelled(Exception):
    """Raised by a task (or by the progress reporter) to abort a cancelled task."""


class AnalysisWorker(QThread):
    """Runs ``task_fn(*args, **kwargs)`` on a background thread, exactly once.

    Cooperative hooks are injected when the task function declares them as named
    parameters (``**kwargs`` alone does not count):

    * ``cancel_event: threading.Event`` - set by :meth:`cancel`; poll it, or
    * ``progress_callback(fraction, message)`` - emits ``progress`` and raises
      :class:`TaskCancelled` once cancellation has been requested, so a task that
      reports progress is cancellable with no extra code.  This is the name
      ``CwtEngine.compute`` uses, so it can be passed directly as ``task_fn``.

    Signals (exactly one of ``result_ready`` / ``error`` is emitted per run):

    * ``result_ready(task_name, result)``
    * ``progress(fraction, message)``
    * ``error(task_name, message)`` - any exception, including cancellation
      (message ``"Cancelled."``).  The traceback is kept in :attr:`last_traceback`.

    If :meth:`cancel` was called, the outcome is always the cancelled error, even
    when the task happened to finish.  The worker is single-use.
    """

    result_ready = Signal(str, object)
    progress = Signal(float, str)
    error = Signal(str, str)

    def __init__(
        self,
        task_fn: Callable[..., Any],
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
        parent: QObject | None = None,
        *,
        task_name: str | None = None,
    ) -> None:
        super().__init__(parent)
        self._task_fn = task_fn
        self._args = tuple(args)
        self._kwargs = dict(kwargs or {})
        self._task_name = task_name or getattr(task_fn, "__name__", type(task_fn).__name__)
        self._cancel_event = threading.Event()
        self._last_traceback = ""

    # ------------------------------------------------------------- state
    @property
    def task_name(self) -> str:
        """Name reported in ``result_ready`` / ``error``."""
        return self._task_name

    @property
    def cancel_event(self) -> threading.Event:
        """The event set by :meth:`cancel`."""
        return self._cancel_event

    @property
    def is_cancelled(self) -> bool:
        """True once :meth:`cancel` has been called."""
        return self._cancel_event.is_set()

    @property
    def last_traceback(self) -> str:
        """Formatted traceback of the exception that ended the run ("" if none)."""
        return self._last_traceback

    def cancel(self) -> None:
        """Ask the task to stop; it is aborted at its next cancellation check."""
        self._cancel_event.set()

    # --------------------------------------------------------------- run
    def _report(self, fraction: float, message: str = "") -> None:
        if self._cancel_event.is_set():
            raise TaskCancelled()
        self.progress.emit(float(min(1.0, max(0.0, fraction))), str(message))

    def _call_arguments(self) -> dict[str, Any]:
        kwargs = dict(self._kwargs)
        try:
            params = inspect.signature(self._task_fn).parameters
        except (TypeError, ValueError):  # some builtins have no signature
            return kwargs
        if CANCEL_PARAM in params:
            kwargs.setdefault(CANCEL_PARAM, self._cancel_event)
        if PROGRESS_PARAM in params:
            kwargs.setdefault(PROGRESS_PARAM, self._report)
        return kwargs

    def run(self) -> None:
        name = self._task_name
        try:
            result = self._task_fn(*self._args, **self._call_arguments())
            if self._cancel_event.is_set():
                raise TaskCancelled()
        except TaskCancelled:
            self.error.emit(name, "Отменено.")
        except BaseException as exc:  # nothing may escape the thread
            self._last_traceback = traceback.format_exc()
            self.error.emit(name, str(exc) or type(exc).__name__)
        else:
            self.result_ready.emit(name, result)


__all__: list[str] = ["AnalysisWorker", "TaskCancelled"]
