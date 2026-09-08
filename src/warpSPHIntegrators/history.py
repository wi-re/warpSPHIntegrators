"""Step history: the shared reuse primitive behind ``priorStep`` and multistep.

``priorStep`` (a single ``StageResult``) is the one-entry degenerate case of this --
see NOTES.md S2. Both feed the same underlying need: "give me a derivative evaluated
at an earlier step, without recomputing it." A two-stage scheme only ever needs the
last one; an order-k Adams-Bashforth scheme needs the last k-1. ``StepHistory`` is the
one container both read from, so the two mechanisms do not drift apart.

Only the *projection* onto tagged derivative fields is stored per entry (the ``k``
object schemes already produce), not the whole state -- a real SPH system carries far
more per step than its integrated derivatives, and history entries otherwise become
the dominant memory cost for anything beyond a two-step method.
"""

from typing import Any, NamedTuple, Optional

from .specs import StageResult


class HistoryEntry(NamedTuple):
    """One remembered stage, with an optional prior-state snapshot for BDF methods."""

    t: float
    dt: float
    update: Any
    aux: Any = None
    state: Any = None

    def as_prior_step(self) -> StageResult:
        """The ``priorStep=`` shape schemes already accept."""
        return StageResult(aux=self.aux, update=self.update)


def _uid_fingerprint(uid) -> Optional[tuple]:
    """A cheap, comparable stand-in for tensor identity: pointer + shape.

    Not a hash of contents -- the point is to catch the *particle set itself* moving
    (a resort, a resize), not a change to individual uid values, which a real resort
    wouldn't produce anyway under this library's stable-indexing assumption
    (NOTES.md S3.0).
    """
    if uid is None:
        return None
    data_ptr = uid.data_ptr() if hasattr(uid, 'data_ptr') else id(uid)
    shape = tuple(uid.shape) if hasattr(uid, 'shape') else None
    return (data_ptr, shape)


class StepHistory:
    """An ordered, bounded run of :class:`HistoryEntry`, newest last.

    Two guards turn a silent wrong answer into an automatic restart (NOTES.md S2g):
    a ``dt`` change (fixed multistep coefficients are only correct under constant
    ``dt``) and a ``uid`` identity change (a resort or a resize invalidates every
    entry's meaning). Neither is a correctness *check* in the sense of raising --
    ``pushed`` just drops the stale entries and starts over, exactly as if history had
    been empty, because that is always a valid (if less accurate) thing to do.
    """

    __slots__ = ('_entries', 'maxlen', '_uid_fingerprint')

    def __init__(self, maxlen: int, entries=(), _uid_fingerprint=None):
        if maxlen < 1:
            raise ValueError(f'StepHistory needs maxlen >= 1, got {maxlen}')
        self.maxlen = maxlen
        self._entries = tuple(entries)[-maxlen:]
        self._uid_fingerprint = _uid_fingerprint

    def __len__(self):
        return len(self._entries)

    def __iter__(self):
        return iter(self._entries)

    def __getitem__(self, index):
        return self._entries[index]

    @property
    def entries(self):
        return self._entries

    @property
    def latest(self) -> Optional[HistoryEntry]:
        return self._entries[-1] if self._entries else None

    def as_prior_step(self) -> Optional[StageResult]:
        latest = self.latest
        return latest.as_prior_step() if latest is not None else None

    def pushed(self, entry: HistoryEntry, *, uid=None) -> 'StepHistory':
        """A new history with ``entry`` appended, restarting first if invalidated.

        Returns a new :class:`StepHistory` rather than mutating -- history is carried
        on ``IntegrationResult`` the same way state is, so it follows the same
        value-not-reference convention (NOTES.md 4.1) instead of becoming a second,
        aliasing exception to it.
        """
        entries = self._entries
        if entries:
            last = entries[-1]
            dt_changed = abs(last.dt - entry.dt) > 1e-12 * max(abs(last.dt), abs(entry.dt), 1.0)
            uid_changed = (
                uid is not None
                and self._uid_fingerprint is not None
                and self._uid_fingerprint != _uid_fingerprint(uid)
            )
            if dt_changed or uid_changed:
                entries = ()
        new_fingerprint = _uid_fingerprint(uid) if uid is not None else self._uid_fingerprint
        return StepHistory(self.maxlen, (*entries, entry), _uid_fingerprint=new_fingerprint)

    def __repr__(self):
        return f'StepHistory(maxlen={self.maxlen}, entries={len(self._entries)})'
