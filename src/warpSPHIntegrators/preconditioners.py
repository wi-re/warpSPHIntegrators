"""Stage-aware local approximations for coupled RK Newton systems."""
import torch

from .fields import BlockState, flatten_integrated


def stage_block_preconditioner(jacobian_diagonal):
    """Approximate (I - dt*A*J)^-1 by an independent s-by-s solve per DOF.

    jacobian_diagonal is a length-N tensor, an s-by-N tensor, or a callable
    (BlockState, context) -> either shape. N is the flattened integrated size
    of one stage, in fields.flatten_integrated order. Stage-dependent columns
    use the corresponding stage Jacobian. Spatial/field off-diagonal coupling
    is omitted. The block driver supplies dt, tableau_a and stage_times.
    LU factors are cached only within one Newton iterate/context, with O(s²N)
    storage; this is a forward-solve preconditioner, not an adjoint operator.
    """
    cached = None

    def apply(vector, state, context):
        nonlocal cached
        if not isinstance(state, BlockState):
            raise ValueError('stage_block_preconditioner needs a coupled BlockState')
        s = len(state.states)
        n = flatten_integrated(state.states[0]).numel()
        if vector.ndim != 1 or vector.numel() != s*n:
            raise ValueError('Preconditioner vector must match the stage-major integrated layout')
        if cached is None or cached[0] is not state or cached[1] is not context:
            diagonal = jacobian_diagonal(state,context) if callable(jacobian_diagonal) else jacobian_diagonal
            diagonal = torch.as_tensor(diagonal,dtype=vector.dtype,device=vector.device)
            if diagonal.shape == (n,):
                diagonal = diagonal.expand(s,n)
            if diagonal.shape != (s,n) or not torch.isfinite(diagonal).all():
                raise ValueError('Jacobian diagonal must be finite and have shape (N,) or (s,N)')
            a = torch.as_tensor(context['tableau_a'],dtype=vector.dtype,device=vector.device)
            if a.shape != (s,s):
                raise ValueError('Preconditioner tableau must match the stage count')
            matrices = torch.eye(s,dtype=vector.dtype,device=vector.device).expand(n,s,s) - context['dt']*a.unsqueeze(0)*diagonal.T.unsqueeze(1)
            lu,pivots = torch.linalg.lu_factor(matrices)
            cached = (state,context,lu,pivots)
        corrected = torch.linalg.lu_solve(cached[2],cached[3],vector.reshape(s,n).T.unsqueeze(-1))
        return corrected.squeeze(-1).T.reshape(-1)
    return apply
