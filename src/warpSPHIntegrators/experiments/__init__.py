"""Bounded tensor experiments: numerical contracts for future backend projects.

These functions operate on tensors and ordinary f(t,y) callables. They do not
implement the full-state lifecycle, transactional history or solve protocol.
They deliberately remain outside scheme lookup and the ordinary API.
"""
from .tensor import taylor_step, operator_step, multirate_step, interaction_picture_step, implicit_extrapolation_step, nordsieck_predict, nordsieck_rescale, nordsieck_adams_step, amf_matrix, low_storage_ck54_step, irkc_step, exponential_adams2_step
__all__=['taylor_step','operator_step','multirate_step','interaction_picture_step',
    'implicit_extrapolation_step','nordsieck_predict','nordsieck_rescale','nordsieck_adams_step','amf_matrix','low_storage_ck54_step','irkc_step','exponential_adams2_step']
