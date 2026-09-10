"""Structured right-hand-side interface (NOTES S3.12, Phase 14).

Collapses the right-hand-side input surface to exactly two shapes: **a plain
callable**, or a **typed :class:`RHS`** whose declared capabilities cover every
split any registered or planned scheme asks for. A scheme receives its dynamics as
``f`` and never needs to know which of the two it got: the combined right-hand
side is ``f(state, dt, ...)`` in both cases (a plain callable *is* its combined
form, and an :class:`RHS`'s ``__call__`` is its combined form). Schemes that need
a *split* of the dynamics — the additive ``f = f_exp + f_imp`` (IMEX Euler, ARK)
or the semilinear ``f = L·y + N`` (Rosenbrock-W, exponential integrators) — call
:func:`resolve` to turn their ``f`` into the concrete accessors they need, and get
a specific capability error *before the solve* if the split is not carried.

Why a concrete class and not a structural ``Protocol``: a plain function must
never ``isinstance``-match the structured form, so the capability test
``isinstance(f, RHS)`` is safe and the two input shapes stay disjoint.

The additive split (``IMEXRHS``) is one shape of :class:`RHS`; the semilinear
split (``SemilinearRHS``) is another. ``IMEXRHS`` is now a thin constructor that
returns an :class:`RHS` rather than its own NamedTuple — nothing downstream
constructs the old dataclass, so the reshape preserves no external contract.
"""

from dataclasses import dataclass, fields, is_dataclass
from typing import Any, Callable, FrozenSet, Optional

import torch

from .fields import flatten_integrated, unflatten_integrated
from .util import split_return

#: The capability names an :class:`RHS` may declare.
CAPABILITIES = frozenset({'explicit', 'implicit', 'linear', 'nonlinear'})


# --------------------------------------------------------------------------- #
# Update arithmetic                                                            #
# --------------------------------------------------------------------------- #

def _combine_updates(a, b, op):
    """Field-wise ``op(a_field, b_field)`` over two same-typed update objects.

    Tensor fields are combined with ``op``; any non-tensor field (rare) is copied
    from ``a``. The result is a new object of ``a``'s type.
    """
    if type(a) is not type(b):
        raise TypeError(
            f'cannot combine updates of different types: '
            f'{type(a).__name__} and {type(b).__name__}')
    kwargs = {}
    for fld in fields(a):
        va = getattr(a, fld.name)
        vb = getattr(b, fld.name)
        if isinstance(va, torch.Tensor) and isinstance(vb, torch.Tensor):
            kwargs[fld.name] = op(va, vb)
        else:
            kwargs[fld.name] = va
    return type(a)(**kwargs)


def add_updates(a, b):
    """``a + b`` over an update object's fields (falls back to ``__add__``)."""
    if is_dataclass(a) and not isinstance(a, type):
        return _combine_updates(a, b, lambda x, y: x + y)
    return a + b


def sub_updates(a, b):
    """``a - b`` over an update object's fields (falls back to ``__sub__``)."""
    if is_dataclass(a) and not isinstance(a, type):
        return _combine_updates(a, b, lambda x, y: x - y)
    return a - b


# --------------------------------------------------------------------------- #
# The structured RHS                                                           #
# --------------------------------------------------------------------------- #

def _additive_combined(explicit, implicit):
    """``f = f_exp + f_imp``; an absent half contributes nothing."""
    def combined(state, dt, *args, **kwargs):
        k_e, _ = split_return(explicit(state, dt, *args, **kwargs)) if explicit is not None else (None, None)
        k_i, _ = split_return(implicit(state, dt, *args, **kwargs)) if implicit is not None else (None, None)
        if k_e is None:
            return k_i
        if k_i is None:
            return k_e
        return add_updates(k_e, k_i)
    return combined


def _semilinear_combined(linear, nonlinear):
    """``f = L·y + N``; an absent part contributes nothing."""
    def combined(state, dt, *args, **kwargs):
        k_l, _ = split_return(linear(state, dt, *args, **kwargs)) if linear is not None else (None, None)
        k_n, _ = split_return(nonlinear(state, dt, *args, **kwargs)) if nonlinear is not None else (None, None)
        if k_l is None:
            return k_n
        if k_n is None:
            return k_l
        return add_updates(k_l, k_n)
    return combined


class RHS:
    """A structured right-hand side: the combined ``f`` plus declared split parts.

    ``__call__(state, dt, *args, **kwargs)`` is the combined right-hand side
    ``f = f_exp + f_imp = L·y + N`` — always defined, the ground truth a scheme
    uses when it treats the dynamics as a single fully-implicit (or just
    evaluated) call. ``provides`` is a :class:`frozenset` over
    :data:`CAPABILITIES` naming which optional accessors are declared.

    The optional accessors all share the ``(state, dt, *args, **kwargs) ->
    update[, aux]`` contract of the combined callable, so a driver routes each one
    through ``updateStep`` exactly as it routes the combined form:

    * ``explicit(state, ...)`` / ``implicit(state, ...)`` — the additive halves.
    * ``linear(state, ...)`` — the linear-operator action ``L·y``: it reads the
      integrated field(s) it acts on from ``state`` and returns ``L·(field)`` as
      an update. To apply ``L`` to an *arbitrary* vector (a Krylov iterate, a
      preconditioner probe), the caller hands it a state whose integrated field is
      that vector (built with :func:`unflatten_integrated`).
    * ``nonlinear(state, ...)`` — the semilinear remainder ``N = f − L·y``.

    The combined form is resolved from the parts in this order: an explicit
    ``combined`` callable, else the additive halves (``explicit + implicit``),
    else the semilinear parts (``linear + nonlinear``). When both an additive and
    a semilinear view are supplied (the overlap case — the same stiff operator
    behind both ``implicit`` and ``linear``), the additive halves define ``f`` and
    the semilinear parts are a view of it; :func:`check_contracts` verifies the
    two views agree.

    Use the constructors :func:`IMEXRHS`, :func:`SemilinearRHS`, or ``RHS``
    directly with any subset of parts.
    """

    provides: FrozenSet[str]

    def __init__(self, *, combined: Optional[Callable] = None,
                 explicit: Optional[Callable] = None, implicit: Optional[Callable] = None,
                 linear: Optional[Callable] = None, nonlinear: Optional[Callable] = None):
        if combined is not None:
            self._combined = combined
        elif explicit is not None or implicit is not None:
            self._combined = _additive_combined(explicit, implicit)
        elif linear is not None or nonlinear is not None:
            self._combined = _semilinear_combined(linear, nonlinear)
        else:
            raise ValueError(
                'RHS needs a combined callable or at least one split part '
                '(explicit/implicit or linear/nonlinear)')
        self._explicit = explicit
        self._implicit = implicit
        self._linear = linear
        self._nonlinear = nonlinear
        self.provides = frozenset(
            name for name, part in (('explicit', explicit), ('implicit', implicit),
                                    ('linear', linear), ('nonlinear', nonlinear))
            if part is not None)

    def __call__(self, state, dt, *args, **kwargs):
        return self._combined(state, dt, *args, **kwargs)

    def __repr__(self):
        return f'RHS(provides={sorted(self.provides)})'


def IMEXRHS(explicit: Callable, implicit: Callable) -> RHS:
    """An additive-split RHS: ``f = explicit + implicit``.

    Thin constructor returning an :class:`RHS` with ``provides =
    {"explicit", "implicit"}`` and a summing ``__call__``. Replaces the old
    ``IMEXRHS`` NamedTuple; the two halves are read by the same capability
    check (NOTES S3.12), and the combined form lets a scheme that only needs the
    full right-hand side treat the split as fully implicit.
    """
    return RHS(explicit=explicit, implicit=implicit)


def SemilinearRHS(linear: Callable, nonlinear: Optional[Callable] = None,
                  combined: Optional[Callable] = None) -> RHS:
    """A semilinear-split RHS: ``f = L·y + N``.

    Two forms:

    * ``SemilinearRHS(linear=L, nonlinear=N)`` — the split is given directly;
      ``__call__`` is ``L·y + N`` and ``provides = {"linear", "nonlinear"}``.
    * ``SemilinearRHS(linear=L, combined=f)`` — the combined ``f`` is given and
      the nonlinear remainder is synthesized as ``N = f − L·y`` on demand by
      :func:`resolve`; ``provides = {"linear"}``.
    """
    if combined is not None and nonlinear is not None:
        raise ValueError('SemilinearRHS takes (linear, nonlinear) or (linear, combined), not both')
    if combined is not None:
        return RHS(combined=combined, linear=linear)
    return RHS(linear=linear, nonlinear=nonlinear)


# --------------------------------------------------------------------------- #
# Driver resolution                                                            #
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class ResolvedRHS:
    """The concrete accessors a driver resolved from an ``f``.

    ``combined`` is always set. The rest are ``None`` when the split is absent —
    the driver then contributes nothing for that part, exactly as it does today
    for a plain callable in the IMEX/ARK drivers.
    """
    combined: Callable
    explicit: Optional[Callable]
    implicit: Optional[Callable]
    linear: Optional[Callable]
    nonlinear: Optional[Callable]


def _synthesize_nonlinear(combined: Callable, linear: Callable) -> Callable:
    """``N(state) = f(state) − L·y(state)``."""
    def nonlinear(state, dt, *args, **kwargs):
        k_c, _ = split_return(combined(state, dt, *args, **kwargs))
        k_l, _ = split_return(linear(state, dt, *args, **kwargs))
        return sub_updates(k_c, k_l)
    return nonlinear


def _describe(f) -> str:
    if isinstance(f, RHS):
        return f'an RHS providing {sorted(f.provides) or "only the combined f"}'
    return 'a plain callable (which carries only the combined f)'


def _capability_error(scheme_name: str, f, capability: str):
    return TypeError(
        f'{scheme_name} needs a "{capability}" accessor on its right-hand side, but was '
        f'given {_describe(f)}. Wrap the dynamics in a structured RHS that declares '
        f'"{capability}" (e.g. SemilinearRHS(linear=..., ...) for the semilinear split).')


def resolve(f, *, scheme_name: str, need_linear: bool = False) -> ResolvedRHS:
    """Resolve the split accessors a driver needs from ``f`` (a plain callable or an :class:`RHS`).

    Synthesizes what can be synthesized and requires what cannot:

    * ``implicit`` ← the declared ``implicit`` half, else the combined ``f``
      (a plain callable is fully implicit).
    * ``explicit`` ← the declared ``explicit`` half, else absent (``None``; the
      explicit part contributes nothing).
    * ``nonlinear`` ← the declared ``nonlinear``, else ``f − L·y`` when a
      ``linear`` part is available, else the combined ``f`` (with no linear part,
      the whole ``f`` is the nonlinear remainder).
    * ``linear`` ← the declared ``linear`` only — it cannot be synthesized. If
      ``need_linear`` and it is absent, raises a :class:`TypeError` naming the
      missing capability *before the solve*.

    A plain callable is never an :class:`RHS`, so it resolves to
    ``explicit=None, implicit=f, linear=None, nonlinear=f``.
    """
    if isinstance(f, RHS):
        combined = f
        explicit = f._explicit if 'explicit' in f.provides else None
        implicit = f._implicit if 'implicit' in f.provides else combined
        linear = f._linear if 'linear' in f.provides else None
        if 'nonlinear' in f.provides:
            nonlinear = f._nonlinear
        elif linear is not None:
            nonlinear = _synthesize_nonlinear(combined, linear)
        else:
            nonlinear = None
    else:
        combined = f
        explicit = None
        implicit = f
        linear = None
        nonlinear = f
    if need_linear and linear is None:
        raise _capability_error(scheme_name, f, 'linear')
    return ResolvedRHS(combined=combined, explicit=explicit, implicit=implicit,
                       linear=linear, nonlinear=nonlinear)


# --------------------------------------------------------------------------- #
# Debug-mode contract checks                                                   #
# --------------------------------------------------------------------------- #

def _zeroed_state(sample_state):
    """``sample_state`` with every integrated tensor field set to zero."""
    flat = flatten_integrated(sample_state)
    return unflatten_integrated(torch.zeros_like(flat), sample_state)


def _assert_updates_close(label: str, a, b, rtol: float, atol: float):
    for fld in fields(a):
        va = getattr(a, fld.name)
        vb = getattr(b, fld.name)
        if not (isinstance(va, torch.Tensor) and isinstance(vb, torch.Tensor)):
            continue
        if not torch.allclose(va, vb, rtol=rtol, atol=atol):
            diff = (va - vb).abs().max().item()
            raise AssertionError(f'RHS contract violated ({label}): field "{fld.name}" '
                                 f'mismatches by {diff:.3e} (rtol={rtol}, atol={atol})')


def _assert_update_zero(label: str, a, atol: float):
    for fld in fields(a):
        va = getattr(a, fld.name)
        if not isinstance(va, torch.Tensor):
            continue
        if not torch.allclose(va, torch.zeros_like(va), atol=atol):
            mag = va.abs().max().item()
            raise AssertionError(f'RHS contract violated ({label}): field "{fld.name}" '
                                 f'has magnitude {mag:.3e} (atol={atol})')


def check_contracts(rhs: RHS, sample_state, dt, *, rtol: float = 1e-8, atol: float = 1e-10,
                    seed: int = 0) -> None:
    """Verify the declared splits of ``rhs`` are mutually consistent at ``sample_state``.

    The three contracts (NOTES S3.12), each checked only for the parts ``rhs``
    declares:

    1. **Additivity is disjoint** — ``explicit + implicit == f``.
    2. **``linear`` is linear and homogeneous** — ``L·0 = 0`` and
       ``L(a+b) ≈ L(a) + L(b)`` on random vectors (the identity-preconditioner
       regression style).
    3. **Any two declared splits agree** — ``linear·y + nonlinear == f`` (and,
       with the additive halves declared, ``explicit + implicit == f`` from (1)).

    Intended as a debug-mode check (tests, or a harness); it is *not* run on the
    production step path. Raises :class:`AssertionError` on the first violation.
    """
    if not isinstance(rhs, RHS):
        raise TypeError('check_contracts expects an RHS')
    provides = rhs.provides

    f_update, _ = split_return(rhs(sample_state, dt))

    # (1) Additivity.
    if 'explicit' in provides and 'implicit' in provides:
        e, _ = split_return(rhs._explicit(sample_state, dt))
        i, _ = split_return(rhs._implicit(sample_state, dt))
        _assert_updates_close('explicit + implicit == f', add_updates(e, i), f_update, rtol, atol)

    # (2) Linearity of the linear part.
    if 'linear' in provides:
        lz, _ = split_return(rhs._linear(_zeroed_state(sample_state), dt))
        _assert_update_zero('L(0) == 0', lz, atol)
        flat = flatten_integrated(sample_state)
        gen = torch.Generator().manual_seed(seed)
        a = torch.randn(flat.shape, generator=gen, dtype=flat.dtype)
        b = torch.randn(flat.shape, generator=gen, dtype=flat.dtype)
        la, _ = split_return(rhs._linear(unflatten_integrated(a, sample_state), dt))
        lb, _ = split_return(rhs._linear(unflatten_integrated(b, sample_state), dt))
        lab, _ = split_return(rhs._linear(unflatten_integrated(a + b, sample_state), dt))
        _assert_updates_close('L(a+b) == L(a)+L(b)', lab, add_updates(la, lb), rtol, atol)

    # (3) The semilinear view agrees with the combined form.
    if 'linear' in provides and 'nonlinear' in provides:
        l, _ = split_return(rhs._linear(sample_state, dt))
        n, _ = split_return(rhs._nonlinear(sample_state, dt))
        _assert_updates_close('linear·y + nonlinear == f', add_updates(l, n), f_update, rtol, atol)


__all__ = [
    'CAPABILITIES', 'RHS', 'IMEXRHS', 'SemilinearRHS',
    'ResolvedRHS', 'resolve',
    'add_updates', 'sub_updates', 'check_contracts',
]
