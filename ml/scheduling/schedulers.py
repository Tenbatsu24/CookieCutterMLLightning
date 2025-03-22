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
