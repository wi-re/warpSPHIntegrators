> Historical measurement record; original section identifiers are retained.
> [Record index](notes-index.md) · [Current execution plan](../../../REPOSITORY_PLAN.md)

### 3.7 Pain points and limitations

Ordered by what actually survives the §3.0 constraints. The three that used to head
this list — particle resorting, variable `dt`, and neighbour rebuild cost — are all
designed away by the surrounding simulation, and are kept here only as the conditions
under which the plan stops being valid.

**Multistep — what remains**

1. **Startup order caps the whole method.** [verified] Self-starting AB — ramping the
   order up as history accumulates — measures order **2.0 for AB3 and AB4 alike**,
   because the single AB1/Euler startup step contributes `O(dt²)` globally. With a
   Dormand–Prince starter for the first k−1 steps: AB3 → 2.97, AB4 → 3.99, ABM4 →
   4.03. So a high-order starter is not a refinement, it is the difference between
   AB4 and AB2. **Done 2026-08-24**: the starter (Dormand-Prince 5(4), unconditionally
   -- order 5, at or above every shipped order) is hard-coded inside
   `AdamsBashforth`/`AdamsBashforthMoulton`, deliberately *not* a `starter=` parameter
   a caller could override, which is what "must be registered scheme metadata, not
   caller policy" meant in practice: a caller-suppliable starter is exactly the
   footgun this pain point describes, one call away (pick a lower-order one, or one
   whose first stage isn't at `t^n`, and the whole run silently caps at the starter's
   own order instead of this scheme's).
2. **Backprop-through-time depth grows by `k`.** This is the one that bites the
   primary use case. History links step *n* to step *n−k* in the autograd graph, so a
   k-step method deepens BPTT by a factor of k on top of however many steps are already
   unrolled for training. Needs a documented detach policy — and note the inversion
   against implicit methods, which cost extra graph *width* per step but no extra depth
   (§3.4). If training-time memory is the binding constraint, that inversion may matter
   more than the RHS-evaluation count that motivated multistep in the first place.
3. **Memory.** k derivative-shaped buffers at 10⁶–10⁷ particles. Store the projection
   onto tagged derivative fields, not the user's whole update object (S2).
4. **`copied` / `ephemeral` semantics.** History entries were computed against a
   *previous* step's adjacency. Since that adjacency is carried and revalidated rather
   than rebuilt (§3.0), this is now a question about `lastStageSystem` and
   `copy_finalized_fields` bookkeeping rather than a correctness hazard — but read it
   before writing the code, not after.

**Multistep — dormant, guard rather than solve**

5. **Resorting would invalidate history silently.** Indices are stable today by
   deliberate design, so `k^{n-1}` stays meaningful indefinitely. If that ever changes,
   history becomes garbage with no error raised — a silent wrong answer, the same
   class of failure the value-dispatched cloning fix closed for state aliasing. The
   `uid` tensor makes the guard cheap (S2g-b): check its identity, restart on mismatch.
   Guard, don't build a permutation-tracking system for a case that does not exist.
6. **Variable `dt` would break the fixed coefficients.** Constant `dt` makes fixed
   β coefficients exactly correct. "For the most part" is doing real work in that
   sentence, so record `dt` per history entry and restart on change (S2g-a) — 0.5 d
   instead of the ~2 d of divided-difference coefficient regeneration a CFL-driven
   code would need.

**Implicit — what remains**

7. **The fixed-point default is not a stiff solver.** [verified] It stalls at
   `dt·ω = 1` and diverges past it (§3.2). Ship it as the default, because it covers
   the actual regime at 2 iterations per stage, but document the limit in the same
   breath and do not let the README imply otherwise.
8. **Boundary particles must be excluded from the solve** or the stage system is
   singular. §2.3's masking item becomes a prerequisite for Phase 3+, not a parallel
   track. (Phase 2's fixed-count Picard has no linear system and so no singularity —
   another reason to ship that first.)
9. **Convergence failure needs a step-rejection path**, the same machinery adaptive
   `dt` (§2.3) needs. Do not build two. Only applies once iteration counts stop being
   fixed, i.e. Phase 3+.
10. **Cost per step is honestly higher.** With adjacency carried through the state
    (§3.0) an implicit step is ~2–3× an explicit one at 2 Picard iterations, not the
    ~50× it would be if every iteration rebuilt neighbours. That is the real price of
    implicitness and it buys A-stability and symplecticity (§3.3).

**Implicit — resolved by §3.4, recorded so it is not re-litigated**

11. **Neither warp's lack of engine-level forward-mode AD, nor its bare existence in
    `warpSPHCore` for six operators, blocks or unblocks the solver on its own.** FD
    directional derivatives drive Newton to `dt·ω = 1000` with no AD at all
    [verified], and a fixed-count Picard needs no Jacobian action whatsoever — this
    still holds regardless of which operators `warpSPHCore` has since grown JVPs for.
    Exact JVPs (`torch.func.jvp` for torch states; `warpSPHCore`'s dual-tensor
    composition for warp states, scoped to six operators + Covariance — §3.4's
    2026-08-24 correction) are a fast path over FD, not a gate, and not a substitute
    for FD's genericity. The thing that *does* need care at scale is the FD matvec
    staying a directional derivative rather than a dense Jacobian — see §3.4's caution
    — and, for the JVP fast path specifically, failing loudly rather than silently
    when `f` includes an operator outside the wrapped set.
12. **Warp graph capture is fine with a fixed iteration count.** A data-dependent
    iteration count breaks capture (§2.1); the recommended default does not have one.
    Decide alongside §2.2, but the constraint points the same way as ML determinism
    does, which is a rare piece of luck.
