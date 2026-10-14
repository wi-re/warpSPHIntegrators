> Historical measurement record; original section identifiers are retained.
> [Record index](notes-index.md) · [Current execution plan](../../../REPOSITORY_PLAN.md)

### 3.0 Constraints from the surrounding simulation

These are properties of the warpSPH simulation this library is written for. None of
them is derivable from this repository, and every one of them removes work that a
general-purpose ODE library would have to do. They are recorded here because the plan
below is only correct under them — if any stops holding, re-read §3.7.

| Constraint | Why it is that way | What it buys |
|---|---|---|
| **Particles are never re-sorted.** No inlets or outlets either. | Deliberate performance trade to keep file I/O simple, and because ML bindings need to diff two states by elementwise comparison rather than by matching identities. | Particle index `i` means the same particle at every step. A step history is therefore **valid indefinitely** — this dissolves the largest objection to multistep. |
| A `uid` integer tensor is carried in the state. | Lets particles be restored to their origin ids if that ever changes. | The cheap guard that turns "indices moved" from a silent wrong answer into an automatic restart. |
| **`dt` is constant.** Where adaptivity exists it is applied *around* whole steps, not within them. | Networks would otherwise have to generalise across `dt`. | Fixed multistep coefficients are correct as written. No variable-step coefficient regeneration, no Nordsieck / fixed-leading-coefficient machinery. |
| **The neighbour list is carried through the state**, moved over in `initializeNewState`, with a cheap velocity-Verlet-style validity check and a rebuild only when it fails. | Already the right design. | The "one RHS evaluation = one neighbour rebuild" assumption is **false here**. An implicit iteration costs one rebuild plus N cheap checks, not N rebuilds. This was the single largest cost objection to implicit methods and it does not apply. |
| Gradients: torch has forward *and* reverse mode; warp currently has **reverse only**. | `wp.Tape` is a reverse-mode tape. | Decides the solver design — see §3.4. |
