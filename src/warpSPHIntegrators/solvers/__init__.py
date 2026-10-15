"""Nonlinear/Krylov solvers, diagnostics and preconditioners."""
from importlib import import_module as _import_module

_EXPORTS = {'NonlinearSolver': ('warpSPHIntegrators.solvers.nonlinear', 'NonlinearSolver'),
 'FixedPointSolver': ('warpSPHIntegrators.solvers.nonlinear', 'FixedPointSolver'),
 'RelaxedFixedPointSolver': ('warpSPHIntegrators.solvers.nonlinear', 'RelaxedFixedPointSolver'),
 'SolveDiagnostics': ('warpSPHIntegrators.solvers.nonlinear', 'SolveDiagnostics'),
 'SolveResult': ('warpSPHIntegrators.solvers.nonlinear', 'SolveResult'),
 'SolverOptions': ('warpSPHIntegrators.solvers.nonlinear', 'SolverOptions'),
 'JFNKSolver': ('warpSPHIntegrators.solvers.krylov', 'JFNKSolver'),
 'fd_matvec': ('warpSPHIntegrators.solvers.krylov', 'fd_matvec'),
 'jvp_matvec': ('warpSPHIntegrators.solvers.krylov', 'jvp_matvec'),
 'gmres': ('warpSPHIntegrators.solvers.krylov', 'gmres'),
 'identity_preconditioner': ('warpSPHIntegrators.solvers.krylov', 'identity_preconditioner'),
 'diagonal_preconditioner': ('warpSPHIntegrators.solvers.krylov', 'diagonal_preconditioner'),
 'stage_block_preconditioner': ('warpSPHIntegrators.solvers.preconditioners',
                                'stage_block_preconditioner'),
 'stage_shifted_preconditioner': ('warpSPHIntegrators.solvers.preconditioners',
                                  'stage_shifted_preconditioner'),
 'stage_schur_preconditioner': ('warpSPHIntegrators.solvers.preconditioners',
                                'stage_schur_preconditioner')}
__all__ = list(_EXPORTS)

def __getattr__(name):
    if name not in _EXPORTS:raise AttributeError(name)
    module,attribute=_EXPORTS[name]
    value=getattr(_import_module(module),attribute)
    globals()[name]=value
    return value

def __dir__():return sorted(set(globals()) | set(__all__))
