"""S18 mechanical adapter: unchanged S09 read or original workspace control.

No trainer, calibration policy, input parser, or new learned parameters. Main
width remains1024; smaller dimensions are solely mechanical test fixtures.
"""
from .campaign_semantics_grounded_actor import CopyConditionedActor


class S18Actor(CopyConditionedActor):
    def __init__(self, *, arm, control_microsteps=None, **kwargs):
        if arm not in ('original', 'context', 'workspace_control'):
            raise ValueError('unknown S18 arm')
        if any(k in kwargs for k in ('microsteps', 'context_read', 'read_scale', 'no_input')):
            raise ValueError('S18 fixes read scale, base recurrence and full public input')
        if arm == 'workspace_control':
            if type(control_microsteps) is not int or control_microsteps < 2:
                raise ValueError('control needs a prospectively computed integer recurrence')
            steps = control_microsteps
        else:
            if control_microsteps is not None:
                raise ValueError('control recurrence only applies to workspace control')
            steps = 2
        super().__init__(context_read=arm == 'context', read_scale=.5,
                         microsteps=steps, no_input=False, **kwargs)
        self.arm = arm
