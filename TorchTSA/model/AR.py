import typing

import numpy as np
import torch
import torch.optim as optim
from torch.autograd import Variable


class ARModel:

    def __init__(
            self, _theta_num: int = 1,
            _use_const: bool = True
    ):
        self.theta_num = _theta_num
        self.use_const = _use_const

        self.params = []
        if self.use_const:
            self.use_var = Variable(
                torch.zeros(1), requires_grad=True
            )
            self.params.append(self.use_var)
        self.theta_var = Variable(
            torch.zeros(self.theta_num), requires_grad=True
        )
        self.params.append(self.theta_var)

        self.optimizer = optim.Adam(self.params)

    def fit(self, _arr: typing.Union[np.ndarray, typing.Sequence]):
        arr = np.array(_arr)
        input(arr)
