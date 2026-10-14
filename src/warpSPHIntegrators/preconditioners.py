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


def stage_shifted_preconditioner(build_shifted_solver, *, condition_limit=1e8):
    """Diagonalize the small RK stage matrix, then solve shifted spatial systems.

    build_shifted_solver(BlockState, context) returns solve(shift, vector),
    approximating (I-shift*J_frozen)^-1. Vectors/shifts may be complex. The
    application owns its sparse factors, multigrid hierarchy or matrix-free
    spatial solve, in flattened integrated field order. A common frozen Jacobian
    decouples stages exactly for linear systems and approximately otherwise.
    Stage-dependent Jacobians are not assumed equal by the actual Newton map.
    This helper adds no SciPy dependency. Factors are scoped to one iterate/context.
    It is a forward preconditioner; existing adjoint solves remain unchanged.
    """
    import math
    if not callable(build_shifted_solver) or not math.isfinite(condition_limit) or condition_limit<1:
        raise ValueError('Need a solver factory and finite condition_limit >=1')
    cached = None
    def apply(vector,state,context):
        nonlocal cached
        if not isinstance(state,BlockState):
            raise ValueError('stage_shifted_preconditioner requires BlockState')
        s=len(state.states);n=flatten_integrated(state.states[0]).numel()
        if vector.ndim!=1 or vector.numel()!=s*n or vector.dtype not in (torch.float32,torch.float64):
            raise ValueError('Expected a real floating stage-major vector')
        if cached is None or cached[0] is not state or cached[1] is not context:
            a=torch.as_tensor(context['tableau_a'],dtype=vector.dtype,device=vector.device)
            if a.shape!=(s,s) or not torch.isfinite(a).all():
                raise ValueError('Stage matrix must be finite and match the block')
            eigenvalues,basis=torch.linalg.eig(a)
            if float(torch.linalg.cond(basis))>condition_limit:
                raise ValueError('Stage eigenbasis is ill-conditioned; use a block/Schur preconditioner')
            spatial=build_shifted_solver(state,context)
            if not callable(spatial):raise TypeError('Factory must return a shifted solve callable')
            cached=(state,context,eigenvalues,basis,torch.linalg.inv(basis),spatial)
        _,_,values,basis,inverse,spatial=cached
        transformed=inverse@vector.reshape(s,n).to(basis.dtype)
        solved=[]
        for j in range(s):
            value=spatial(context['dt']*values[j],transformed[j])
            if not isinstance(value,torch.Tensor) or value.shape!=(n,) or value.device!=vector.device or value.dtype!=basis.dtype:
                raise ValueError('Shifted solve must preserve complex vector shape, dtype and device')
            solved.append(value)
        result=basis@torch.stack(solved)
        # A real operator and conjugate-pair solves must reconstruct a real vector.
        tolerance=100*torch.finfo(vector.dtype).eps*(1+result.real.abs().max())
        if not torch.isfinite(result).all() or result.imag.abs().max()>tolerance:
            raise ValueError('Shifted solves did not reconstruct a finite real correction')
        return result.real.reshape(-1).to(vector.dtype)
    return apply
