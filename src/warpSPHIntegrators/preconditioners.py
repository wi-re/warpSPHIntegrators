"""Stage-aware local approximations for coupled RK Newton systems."""
import torch

from .fields import BlockState, flatten_integrated


def _stage_matrix(context,vector):
    value=context['tableau_a']
    if isinstance(value,torch.Tensor):return value.to(dtype=vector.dtype,device=vector.device).clone()
    return torch.tensor(value,dtype=vector.dtype,device=vector.device)


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
            a = _stage_matrix(context,vector)
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
            a=_stage_matrix(context,vector)
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


def stage_schur_preconditioner(build_operator, decompose):
    """Solve a frozen coupled RK system in a real orthogonal Schur basis.

    decompose(tableau_a) returns real (T,Q) with A=Q*T*Q.T, orthogonal Q,
    and quasi-upper-triangular T. Supplying this small-matrix decomposition
    keeps the backend independent of SciPy or a particular sparse library.
    build_operator(BlockState,context) returns (solve_shift,apply_jacobian):
    approximate (I-shift*J)^-1 and the matching frozen J action on one stage's
    flattened integrated vector. Shifted solves accept complex tensors;
    Jacobian actions accept real tensors. One/two-stage Schur blocks and
    their upper-triangular coupling are solved in reverse order. Only each
    two-stage block is diagonalized, avoiding a global ill-conditioned basis.
    Factors and decomposition are scoped to one Newton iterate/context.
    This is a forward preconditioner; it does not replace an adjoint solve.
    """
    if not callable(build_operator) or not callable(decompose):
        raise TypeError('Operator and Schur decomposition factories must be callable')
    cached=None
    def apply(vector,state,context):
        nonlocal cached
        if not isinstance(state,BlockState):raise ValueError('Schur preconditioner requires BlockState')
        s=len(state.states);n=flatten_integrated(state.states[0]).numel()
        if vector.ndim!=1 or vector.numel()!=s*n or vector.dtype not in (torch.float32,torch.float64):
            raise ValueError('Expected a real floating stage-major vector')
        tolerance=100*s*torch.finfo(vector.dtype).eps
        if cached is None or cached[0] is not state or cached[1] is not context:
            t,q=decompose(context['tableau_a'])
            t=torch.as_tensor(t,device=vector.device);q=torch.as_tensor(q,device=vector.device)
            if t.is_complex() or q.is_complex():raise ValueError('Expected a real Schur decomposition')
            t=t.to(vector.dtype);q=q.to(vector.dtype)
            a=_stage_matrix(context,vector)
            identity=torch.eye(s,dtype=vector.dtype,device=vector.device)
            if (t.shape!=(s,s) or q.shape!=(s,s) or not torch.isfinite(t).all() or not torch.isfinite(q).all()
                    or not torch.allclose(q.T@q,identity,atol=tolerance,rtol=tolerance)
                    or not torch.allclose(q@t@q.T,a,atol=tolerance,rtol=tolerance)
                    or torch.tril(t,-2).abs().max()>tolerance):
                raise ValueError('Expected a finite real orthogonal Schur decomposition')
            subdiagonal=torch.diagonal(t,offset=-1).abs()>tolerance
            if s>2 and (subdiagonal[:-1]&subdiagonal[1:]).any():
                raise ValueError('Real Schur blocks must have size one or two')
            spatial,jacobian=build_operator(state,context)
            if not callable(spatial) or not callable(jacobian):
                raise TypeError('Operator factory must return shifted solve and Jacobian action callables')
            cached=(state,context,t,q,spatial,jacobian)
        _,_,t,q,spatial,jacobian=cached
        transformed=q.T@vector.reshape(s,n)
        solved=[None]*s;actions=[None]*s
        complex_dtype=torch.complex128 if vector.dtype==torch.float64 else torch.complex64
        def shifted(shift,value):
            output=spatial(shift,value)
            if (not isinstance(output,torch.Tensor) or output.shape!=(n,) or output.device!=vector.device
                    or output.dtype!=complex_dtype or not torch.isfinite(output).all()):
                raise ValueError('Shifted solve must preserve complex vector shape, dtype and device')
            return output
        def real(value):
            if value.imag.abs().max()>tolerance*(1+value.real.abs().max()):
                raise ValueError('Schur block did not reconstruct a real correction')
            return value.real.to(vector.dtype)
        i=s-1
        while i>=0:
            first=i-1 if i>0 and abs(t[i,i-1])>tolerance else i
            rhs=transformed[first:i+1].clone()
            for row in range(first,i+1):
                for j in range(i+1,s):rhs[row-first]+=context['dt']*t[row,j]*actions[j]
            if first==i:
                values=real(shifted((context['dt']*t[i,i]).to(complex_dtype),rhs[0].to(complex_dtype))).unsqueeze(0)
            else:
                eigenvalues,basis=torch.linalg.eig(t[first:i+1,first:i+1])
                if torch.linalg.cond(basis)>1e6:raise ValueError('Two-stage Schur block is ill-conditioned')
                local=torch.linalg.solve(basis,rhs.to(complex_dtype))
                values=real(basis@torch.stack([shifted(context['dt']*eigenvalues[j],local[j]) for j in range(2)]))
            for row in range(first,i+1):
                solved[row]=values[row-first]
                action=jacobian(solved[row])
                if (not isinstance(action,torch.Tensor) or action.shape!=(n,) or action.dtype!=vector.dtype
                        or action.device!=vector.device or not torch.isfinite(action).all()):
                    raise ValueError('Jacobian action must preserve the real vector layout')
                actions[row]=action
            i=first-1
        result=q@torch.stack(solved)
        if not torch.isfinite(result).all():raise ValueError('Nonfinite Schur correction')
        return result.reshape(-1)
    return apply
