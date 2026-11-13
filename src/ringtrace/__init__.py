"""ringtrace: find fraud rings in a synthetic UPI ledger.

Importing the package pins every numeric library to an explicit thread budget before any of
them loads, because the compute box is shared.
"""

from ringtrace import threads as _threads

_threads.pin()

__all__ = ["CLASSES", "N_CLASSES"]

from ringtrace.classes import CLASSES, N_CLASSES  # noqa: E402
