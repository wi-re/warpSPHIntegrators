"""Portable, tensor-only restart files with no arbitrary object deserialization."""
import torch

from warpSPHIntegrators.state.history import HistoryEntry, StepHistory
from warpSPHIntegrators.state.specs import StageResult
from warpSPHIntegrators.adapters.tensor import TensorState, _TensorUpdate
from warpSPHIntegrators.runtime.run import RunCheckpoint
from warpSPHIntegrators.state.ownership import snapshot_state

_FORMAT = 'warpSPHIntegrators.tensor-run.v1'


def save_checkpoint(checkpoint, path):
    """Save a TensorState checkpoint, detaching tensors from autograd graphs.

    General protocol systems use in-memory RunCheckpoint or application-owned
    serialization. RHS functions, solver options, events and transactions must be
    supplied again on restart. Files contain only tensors and primitive metadata.
    """
    def state(value):
        if value is None:
            return None
        if type(value) is not TensorState:
            raise TypeError('Portable checkpoints require plain TensorState; serialize custom systems in the application')
        return dict(y=value.y.detach().clone(), t=float(value.t))

    def empty_aux(aux):
        return aux is None or isinstance(aux,(tuple,list)) and all(empty_aux(v) for v in aux)

    def stage(update, aux):
        if update is None and aux is None:return None  # cache-only history entry
        if type(update) is not _TensorUpdate or not empty_aux(aux):
            raise TypeError('Portable checkpoints require tensor updates without auxiliary objects')
        return dict(dy=update.dy.detach().clone(),aux=aux)

    def context(value):
        if isinstance(value,torch.Tensor):return value.detach().clone()
        if value is None or type(value) in (str,int,float,bool):return value
        if isinstance(value,(tuple,list)):return type(value)(context(v) for v in value)
        if isinstance(value,dict) and all(type(k) is str for k in value):
            return {k:context(v) for k,v in value.items()}
        raise TypeError('Portable history context must contain only tensors and primitive metadata')

    payload = dict(format=_FORMAT, scheme=checkpoint.scheme, state=state(checkpoint.state),
        history=[dict(t=e.t, dt=e.dt, stage=stage(e.update,e.aux), state=state(e.state))
                 for e in checkpoint.history], maxlen=checkpoint.history.maxlen,
        history_context=context(checkpoint.history.context),
        prior=None if checkpoint.prior_step is None else stage(checkpoint.prior_step.update,checkpoint.prior_step.aux),
        last_dt=checkpoint.last_dt, next_dt=checkpoint.next_dt,
        configuration=snapshot_state(checkpoint.configuration,detach=True))
    torch.save(payload,path)


def load_checkpoint(path, *, map_location=None):
    """Load a portable tensor checkpoint using torch's weights-only loader."""
    payload = torch.load(path,map_location=map_location,weights_only=True)
    if not isinstance(payload,dict) or payload.get('format') != _FORMAT:
        raise ValueError('Unsupported checkpoint format')
    def state(value):
        return None if value is None else TensorState(value['y'],t=value['t'])
    entries = [HistoryEntry(e['t'],e['dt'],None if e['stage'] is None else _TensorUpdate(e['stage']['dy']),
                           None if e['stage'] is None else e['stage']['aux'],state(e['state']))
               for e in payload['history']]
    prior = None if payload['prior'] is None else StageResult(payload['prior']['aux'],_TensorUpdate(payload['prior']['dy']))
    return RunCheckpoint(payload['scheme'],state(payload['state']),
        StepHistory(payload['maxlen'],entries,context=payload.get('history_context')),prior,payload['last_dt'],payload['next_dt'],
        payload['configuration'])
