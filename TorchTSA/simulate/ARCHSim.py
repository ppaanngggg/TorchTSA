import math
import random
import typing

import numpy as np


class ARCHSim:

    def __init__(
            self,
            _alpha_arr: typing.Sequence[float],
            _const: float,
            _mu: float = 0.0,
    ):
        # AR part
        self.alpha_arr = np.array(_alpha_arr)
        assert np.all(self.alpha_arr > 0)
        self.alpha_num = len(_alpha_arr)

        self.const = _const
        assert self.const > 0
        self.mu = _mu

        self.ret = []
        self.square = [0.0] * self.alpha_num

    def sample(self) -> float:
        new_value = self.mu

        tmp = self.square[-self.alpha_num:]
        tmp.reverse()
        tmp = np.array(tmp)
        sigma = self.const + (tmp * self.alpha_arr).sum()

        new_info = random.gauss(0, math.sqrt(sigma))
        new_value += new_info
        self.ret.append(new_value)
        self.square.append(new_info ** 2)

        return new_value

    def sample_n(self, _num: int) -> np.ndarray:
        for _ in range(_num):
            self.sample()
        return np.array(self.ret[-_num:])
