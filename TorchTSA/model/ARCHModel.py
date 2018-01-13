import logging
import typing

import numpy as np
import torch
import torch.optim as optim
from torch.autograd import Variable
from torch.distributions import Normal


class ARCHModel:

    def __init__(
            self,
            _alpha_num: int = 1,
            _use_mu: bool = True,
    ):
        self.alpha_num = _alpha_num
        self.use_mu = _use_mu

        self.log_alpha_arr: np.ndarray = None
        self.log_const_arr: np.ndarray = None
        self.mu_arr: np.ndarray = None

    @staticmethod
    def stack_delay_arr(
            _arr: typing.Sequence[float], _num: int
    ) -> np.ndarray:
        ret_list = []
        for i in range(_num):
            shift = i + 1
            ret_list.append(_arr[_num - shift: -shift])
        return np.stack(ret_list)

    def fit(
            self, _arr: typing.Sequence[float],
            _max_iter: int = 20,
    ):
        assert len(_arr) > self.alpha_num
        arr = np.array(_arr)

        # get mu and remove mu from original data
        if self.use_mu:
            self.mu_arr = np.mean(_arr, keepdims=True)
            arr = arr - self.mu_arr
        else:
            self.mu_arr = np.zeros(1)

        # y_var
        y_var = Variable(
            torch.from_numpy(arr[self.alpha_num:]).float()
        )
        # x_var
        square_arr = arr ** 2
        x_arr = self.stack_delay_arr(square_arr, self.alpha_num)
        x_var = Variable(torch.from_numpy(x_arr).float())

        # get vars and optimizer
        params = []
        # 1. const
        if self.log_const_arr is None:
            self.log_const_arr = np.log(
                np.std(arr, keepdims=True) ** 2
            )
        log_const_var = Variable(
            torch.from_numpy(self.log_const_arr).float(),
            requires_grad=True
        )
        params.append(log_const_var)
        # 2. alpha
        if self.log_alpha_arr is None:
            self.log_alpha_arr = np.empty(self.alpha_num)
            init_value = 1. / np.sqrt(len(arr))
            self.log_alpha_arr.fill(init_value)
        log_alpha_var = Variable(
            torch.from_numpy(self.log_alpha_arr).float().unsqueeze(0),
            requires_grad=True
        )
        params.append(log_alpha_var)

        optimizer = optim.LBFGS(params, max_iter=_max_iter)

        def closure():
            optimizer.zero_grad()
            out = torch.mm(
                torch.exp(log_alpha_var), x_var
            ) + torch.exp(log_const_var)
            loss = -Normal(0, torch.sqrt(out)).log_prob(y_var).mean()
            logging.info('loss: {}'.format(loss.data.numpy()[0]))
            loss.backward()
            return loss

        optimizer.step(closure)

        self.log_const_arr = log_const_var.data.numpy()
        self.log_alpha_arr = log_alpha_var.data.numpy()[0]

    def getAlpha(self) -> np.ndarray:
        return np.exp(self.log_alpha_arr)

    def getConst(self) -> np.ndarray:
        return np.exp(self.log_const_arr)

    def getMu(self) -> np.ndarray:
        return self.mu_arr
