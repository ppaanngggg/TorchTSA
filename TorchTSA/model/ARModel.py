import logging
import typing

import numpy as np
import torch
import torch.optim as optim
from torch.autograd import Variable
from torch.distributions import Normal


class ARModel:

    def __init__(
            self, _theta_num: int = 1,
            _use_const: bool = True
    ):
        self.theta_num = _theta_num
        self.use_const = _use_const

        if self.use_const:  # optional const
            self.const_arr = np.array([0.0])
        self.theta_arr = np.random.rand(self.theta_num) - 0.5

        self.normal = Normal(  # to get log_prob
            Variable(torch.zeros(1)), 1.0
        )

    def fit(
            self,
            _arr: typing.Sequence[float],
            _max_iter: int = 50,
            _tol: float = 1e-4,
    ):
        arr = np.array(_arr)
        assert len(arr) > self.theta_num

        # get dataset
        x_list = []
        for i in range(self.theta_num):
            shift = i + 1
            x_list.append(arr[self.theta_num - shift: -shift])
        x_var = Variable(torch.from_numpy(np.stack(x_list))).float()
        y_var = Variable(torch.from_numpy(arr[self.theta_num:])).float()

        # get vars
        tmp_params = []
        if self.use_const:
            const_tensor = torch.from_numpy(self.const_arr).float()
            const_var = Variable(
                const_tensor, requires_grad=True
            )
            tmp_params.append(const_var)

        theta_tensor = torch.from_numpy(self.theta_arr).float().unsqueeze(0)
        theta_var = Variable(
            theta_tensor, requires_grad=True
        )
        tmp_params.append(theta_var)
        optimizer = optim.SGD(tmp_params, lr=0.5)

        last_loss = None
        for i in range(_max_iter):
            logging.info('ITER: {}/{}'.format(i + 1, _max_iter))

            tmp = torch.mm(theta_var, x_var)
            if self.use_const:
                tmp += const_var
            residual = y_var - tmp
            log_likelihood = self.normal.log_prob(residual)
            loss = -log_likelihood.mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_value = loss.data[0]
            if last_loss is not None:
                if abs(last_loss - loss_value) < _tol:
                    break
            last_loss = loss_value
        else:
            logging.warning('Maybe not converged')

        self.theta_arr = theta_var.data.numpy()[0]
        if self.use_const:
            self.const_arr = const_var.data.numpy()

    def getThetas(self) -> np.ndarray:
        return self.theta_arr

    def getConst(self) -> typing.Union[None, np.ndarray]:
        if self.use_const:
            return self.const_arr
        else:
            return None

    def predict(
            self, _arr: typing.Sequence[float]
    ) -> float:
        assert len(_arr) >= self.theta_num

        tmp = np.array(_arr[-self.theta_num:])
        tmp = tmp[::-1]

        value = (tmp * self.getThetas()).sum()
        if self.use_const:
            value += self.getConst()[0]
        return value
