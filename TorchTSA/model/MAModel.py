import logging
import typing

import numpy as np
import torch
import torch.optim as optim
from torch.autograd import Variable
from torch.distributions import Normal


class MAModel:

    def __init__(
            self, _theta_num: int = 1,
            _use_const: bool = True
    ):
        self.theta_num = _theta_num
        self.use_const = _use_const

        tmp_params = []
        if self.use_const:
            self.const_var = Variable(
                torch.zeros(1), requires_grad=True
            )
            tmp_params.append(self.const_var)
        self.theta_var = Variable(
            torch.rand(1, self.theta_num) - 0.5,
            requires_grad=True
        )
        tmp_params.append(self.theta_var)

        self.normal = Normal(
            Variable(torch.zeros(1)), 1.0
        )
        self.optimizer = optim.SGD(tmp_params, lr=0.1)

    def fit(
            self,
            _arr: typing.Sequence[float],
            _max_iter: int = 50,
            _tol: float = 1e-4,
    ):
        arr = np.array(_arr)
        assert len(arr) > self.theta_num

        x_list = []
        for i in range(self.theta_num):
            shift = i + 1
            x_list.append(arr[self.theta_num - shift: -shift])
        x_var = Variable(torch.from_numpy(np.stack(x_list))).float()
        y_var = Variable(torch.from_numpy(arr[self.theta_num:])).float()

        last_loss = None
        for iter in range(_max_iter):
            logging.info('ITER: {}/{}'.format(iter + 1, _max_iter))

            tmp = torch.mm(self.theta_var, x_var)
            if self.use_const:
                tmp += self.const_var
            residual = y_var - tmp
            log_likelihood = self.normal.log_prob(residual)
            loss = -log_likelihood.mean()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            loss_value = loss.data[0]
            if last_loss is not None:
                if abs(last_loss - loss_value) < _tol:
                    break
            last_loss = loss_value
        else:
            logging.warning('Maybe not converged')

    def thetas(self) -> np.ndarray:
        return self.theta_var.data.numpy()[0]

    def const(self) -> typing.Union[None, float]:
        if self.use_const:
            return self.const_var.data.numpy()[0]
        else:
            return None
