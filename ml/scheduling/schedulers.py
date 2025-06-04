from math import log

from typing import Union, List
from typing_extensions import Self

import torch

from .util import Schedule, ConstSched


class CatSched(Schedule):
    def __init__(
        self,
        sched_l: Union[Schedule, float],
        sched_r: Union[Schedule, float],
        where: Union[float, int],
    ):
        super().__init__()
        self.sched_l = sched_l
        self.sched_r = sched_r
        self.where = where

        # convert number to ConstSched
        if not isinstance(sched_l, Schedule):
            self.sched_l = ConstSched(sched_l)
        if not isinstance(sched_r, Schedule):
            self.sched_r = ConstSched(sched_r)

    def set_ys(self) -> Self:
        if isinstance(self.where, float):
            frac = self.where  # interprete as rounded fraction of epoch
            n_steps_l = round(frac * self.n_steps)
            n_steps_r = self.n_steps - n_steps_l
            n_epochs_l, n_epochs_r = -1, -1
        elif isinstance(self.where, int):
            n_epochs_l = self.where  # interprete as epoch
            n_epochs_r = self.n_epochs - n_epochs_l
            n_steps_l, n_steps_r = -1, -1
        else:
            raise ValueError("Unkown type for 'where'.")

        # prepare schedules of left and right
        ys_list = []
        if n_steps_l > 0 or n_epochs_l > 0:  # don't prepare if length is 0
            self.sched_l.prep(n_steps_l, n_epochs_l, self.steps_per_epoch)
            ys_list.append(self.sched_l.ys)
        if n_steps_r > 0 or n_epochs_r > 0:  # don't prepare if length is 0
            self.sched_r.prep(n_steps_r, n_epochs_r, self.steps_per_epoch)
            ys_list.append(self.sched_r.ys)
        # concatenate left and right schedules
        self.ys = torch.concat(ys_list)

        # de-materialize left and right schedules
        self.sched_r.unprep()
        self.sched_l.unprep()

    def __repr__(self, _=None) -> str:
        return super().__repr__([self.sched_l, self.sched_r, self.where])


class LinSched(Schedule):
    def __init__(self, y_start, y_end):
        super().__init__()
        self.y_start = float(y_start)
        self.y_end = float(y_end)

    def set_ys(self) -> Self:
        self.ys = self.y_start + (self.y_end - self.y_start) * self.xs()

    def __repr__(self, _=None) -> str:
        return super().__repr__([self.y_start, self.y_end])


class CosSched(Schedule):
    def __init__(self, y_start, y_end):
        super().__init__()
        self.y_start = float(y_start)
        self.y_end = float(y_end)

    def set_ys(self) -> Self:
        cos = 0.5 + torch.cos(self.xs(-torch.pi, 0)) / 2
        self.ys = self.y_start + (self.y_end - self.y_start) * cos

    def __repr__(self, _=None) -> str:
        return super().__repr__([self.y_start, self.y_end])


class ExpSched(Schedule):
    def __init__(self, y_start, y_end):
        super().__init__()
        self.y_start = float(y_start)
        self.y_end = float(y_end)

    def set_ys(self):
        self.ys = torch.exp(self.xs(log(self.y_start), log(self.y_end)))

    def __repr__(self, _=None) -> str:
        return super().__repr__([self.y_start, self.y_end])


class LinWarmup(CatSched, Schedule):
    def __init__(self, y_start: float, y_end: float, where: int):
        super().__init__(LinSched(y_start, y_end), y_end, where)
        self.y_start = float(y_start)
        self.y_end = float(y_end)
        self.where = where

    def __repr__(self, _=None) -> str:
        return Schedule.__repr__(self, [self.y_start, self.y_end, self.where])


class ExpWarmup(CatSched, Schedule):
    def __init__(self, y_start: float, y_end: float, where: int):
        super().__init__(ExpSched(y_start, y_end), y_end, where)
        self.y_start = float(y_start)
        self.y_end = float(y_end)
        self.where = where

    def __repr__(self, _=None) -> str:
        return Schedule.__repr__(self, [self.y_start, self.y_end, self.where])


class CosWarmup(CatSched, Schedule):
    def __init__(self, y_start: float, y_end: float, where: int):
        super().__init__(CosSched(y_start, y_end), y_end, where)
        self.y_start = float(y_start)
        self.y_end = float(y_end)
        self.where = where

    def __repr__(self, _=None) -> str:
        return Schedule.__repr__(self, [self.y_start, self.y_end, self.where])


class MultiStep(Schedule):
    def __init__(self, start: float, gamma: float, *steps: List[Union[int, float]]):
        super().__init__()
        self.start = float(start)
        self.gamma = float(gamma)
        self.steps = steps  # ratio of steps if float, epochs if int

    def set_ys(self) -> Self:
        last_step = 0
        value = self.start
        self.ys = torch.full((self.n_steps,), torch.nan)

        for step in self.steps:

            if isinstance(step, int):
                step = step * self.steps_per_epoch
            if isinstance(step, float):
                step = int(step * self.n_steps)

            self.ys[last_step:step] = value  # fill ys
            value *= self.gamma  # update value
            last_step = step  # update last step

        self.ys[last_step:] = value  # fill remaining

    def __repr__(self, _=None) -> str:
        return Schedule.__repr__(self, [self.start, self.gamma] + list(self.steps))


class StepSched(Schedule):
    def __init__(self, start: float, gamma: float, step_every: int, warmup_epochs: int):
        super().__init__()
        self.start = float(start)
        self.gamma = float(gamma)
        self.step_every = step_every
        self.warmup_epochs = warmup_epochs

    def set_ys(self) -> Self:
        milestones = [
            i - self.warmup_epochs for i in range(self.step_every, self.n_epochs, self.step_every)
        ]
        milestones = [i * self.steps_per_epoch for i in milestones]

        last_step = 0
        value = self.start
        self.ys = torch.full((self.n_steps,), torch.nan)
        for step in milestones:
            self.ys[last_step:step] = value
            value *= self.gamma
            last_step = step
        self.ys[last_step:] = value

    def __repr__(self, _=None) -> str:
        return Schedule.__repr__(
            self, [self.start, self.gamma, self.step_every, self.warmup_epochs]
        )


class StepCycleSched(Schedule):
    """
    Creates a schedule where we rise from min_v to max_v in cos schedule,
    then decay to min_v in a cos schedule. the rise and falls happen in length of cycle_length.

    Then we decay the max_v using the decay factor. and repeat until another cycle won't fit in the schedule.
    """

    def __init__(self, min_v: float, max_v: float, decay: float, cycle_length: int):
        super().__init__()
        self.min_v = float(min_v)
        self.max_v = float(max_v)
        self.decay = float(decay)
        self.cycle_length = cycle_length

    def set_ys(self) -> Self:
        # if 0 < cycle length < 1, get the number of steps else treat it as epochs and convert to steps
        if 0 < self.cycle_length < 1:
            cycle_length = int(self.cycle_length * self.n_steps)
        elif self.cycle_length > 1:
            cycle_length = int(self.cycle_length * self.steps_per_epoch)
        else:
            raise ValueError("cycle_length must be > 0")

        # calculate the number of cycles
        n_cycles = self.n_steps // cycle_length
        # calculate the remaining steps
        n_remaining = self.n_steps % cycle_length

        # create the schedule for the cycles
        xs = torch.linspace(-torch.pi, 0, cycle_length // 2)
        cycle = torch.cat(
            [
                self.min_v + (self.max_v - self.min_v) * (0.5 + torch.cos(xs) / 2),
                self.max_v + (self.min_v - self.max_v) * (0.5 + torch.cos(xs) / 2),
            ]
        )
        # repeat the cycle for n_cycles
        ys = cycle.repeat(n_cycles)

        # decay the max value
        for i in range(n_cycles):
            ys[i * cycle_length : (i + 1) * cycle_length] *= self.decay**i

        # add the remaining steps as 0
        if n_remaining > 0:
            ys = torch.cat([ys, torch.zeros(n_remaining)])

        self.ys = ys
