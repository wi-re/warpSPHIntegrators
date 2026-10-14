"""Executed outside the checkout by validate_wheel; imports only the installed wheel."""

from pathlib import Path
import importlib
import importlib.metadata
import json
import math
import pkgutil
import sys

import torch
import warpSPHIntegrators as package
from warpSPHIntegrators import (
    TensorState, tensor_rhs, getIntegrator, getPreferredIntegrator, filter_schemes,
    StepHistory, SemilinearRHS, testing, get_reference_state,
)


prefix = Path(sys.prefix).resolve()
assert Path(package.__file__).resolve().is_relative_to(prefix), package.__file__
assert package.__version__ == importlib.metadata.version('warpSPHIntegrators') == sys.argv[1]
assert (Path(package.__file__).parent / 'py.typed').is_file()
for module in pkgutil.walk_packages(package.__path__, package.__name__ + '.'):
    imported = importlib.import_module(module.name)
    assert Path(imported.__file__).resolve().is_relative_to(prefix), imported.__file__
assert all(hasattr(package, name) for name in package.__all__)
expected = json.loads(Path(sys.argv[2]).read_text())
assert {s.identifier.name for s in filter_schemes()} == {s['identifier'] for s in expected['schemes']}
assert getPreferredIntegrator(4) is getIntegrator('RK4')
y = torch.tensor([1., 2.], dtype=torch.float64)
for scheme in filter_schemes(state_kind='tensor'):
    state = TensorState(y)
    f = tensor_rhs(SemilinearRHS(lambda t, y: -y))
    options = {'s':3} if scheme.metadata.required_options else {}
    history = StepHistory(maxlen=5)
    for _ in range(6):
        result = scheme(state, .001, f, history=history, **options)
        state, history = result.state, result.history or history
    assert torch.allclose(state.y, y * math.exp(-.006), atol=2e-5, rtol=2e-5), scheme.name
# Keep the advanced protocol consumer in the installed-package gate too.
problem = testing.oscillator_problem()
result = getIntegrator('RK4')(problem.initial(), .01, problem.rhs)
assert torch.isfinite(get_reference_state(result.state).x).all()
print(f'Installed wheel {package.__version__}: all modules/exports, py.typed, '
      f'{len(filter_schemes(state_kind="tensor"))} tensor drivers and the advanced state protocol pass')
