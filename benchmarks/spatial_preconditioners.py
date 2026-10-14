"""CPU sparse reference/ILU and shifted-space adapters for the periodic scale case.

SciPy is a benchmark-only dependency. These are concrete 1D stencil experiments,
not generic SPH assemblies or a claim of scalable 3D direct factorization.
"""
import numpy as np
import scipy.sparse as sparse
from scipy.sparse.linalg import splu,spilu
import torch
from warpSPHIntegrators import stage_shifted_preconditioner


def spatial_preconditioner(diagonal,diffusion,mode,metrics=None):
    metrics={} if metrics is None else metrics
    metrics.update(factorizations=0,factor_seconds=0.,factor_nnz=0)
    import time
    def jacobian(d):
        n=len(d);rows=np.arange(n)
        return sparse.coo_matrix((np.concatenate([d,np.full(n,diffusion),np.full(n,diffusion)]),
            (np.tile(rows,3),np.concatenate([rows,(rows+1)%n,(rows-1)%n]))),shape=(n,n)).tocsc()
    def factor(matrix):
        start=time.perf_counter()
        lu=spilu(matrix,drop_tol=1e-4,fill_factor=10) if mode.endswith('ilu') else splu(matrix)
        metrics['factorizations']+=1;metrics['factor_seconds']+=time.perf_counter()-start
        metrics['factor_nnz']=max(metrics['factor_nnz'],lu.L.nnz+lu.U.nnz)
        return lu
    if mode.startswith('shifted'):
        def build(state,context):
            d=diagonal(state,context)[0].detach().cpu().numpy()
            j=jacobian(d).astype(np.complex128);identity=sparse.eye(len(d),format='csc',dtype=np.complex128)
            factors={}
            def solve(shift,vector):
                key=complex(shift.item())
                if key not in factors:factors[key]=factor(identity-key*j)
                out=factors[key].solve(vector.detach().cpu().numpy())
                return torch.as_tensor(out,dtype=vector.dtype,device=vector.device)
            return solve
        return stage_shifted_preconditioner(build)
    cached=None
    def apply(vector,state,context):
        nonlocal cached
        if cached is None or cached[0] is not state or cached[1] is not context:
            d=diagonal(state,context).detach().cpu().numpy();s,n=d.shape
            a=context['tableau_a'];js=[jacobian(row) for row in d]
            matrix=sparse.bmat([[((sparse.eye(n,format='csc') if i==j else sparse.csc_matrix((n,n)))
                    -context['dt']*a[i,j]*js[j]) for j in range(s)] for i in range(s)],format='csc')
            cached=(state,context,factor(matrix))
        out=cached[2].solve(vector.detach().cpu().numpy())
        return torch.as_tensor(out,dtype=vector.dtype,device=vector.device)
    return apply
