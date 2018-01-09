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

        if self.use_const:  # optional const
            self.const_arr = np.zeros(1)
        self.theta_arr = np.zeros(self.theta_num)
        self.sigma_arr = np.ones(1)

    def fit(
            self,
            _arr: typing.Sequence[float],
            _max_iter: int = 20000,
            _tol: float = 1e-8,
    ):
        assert len(_arr) > self.theta_num

        # get y_var
        arr = np.array(_arr)
        y_var = Variable(torch.from_numpy(arr).float())
        std_value = np.std(arr)  # init sigma
        self.sigma_arr[:] = std_value
        if self.use_const:
            mean_value = np.mean(arr)
            self.const_arr[:] = mean_value

        tmp_params = []
        # get latent var
        latent_arr = np.zeros(self.theta_num + len(arr))
        if self.use_const:
            latent_arr[:self.theta_num] = self.const_arr
            latent_arr[self.theta_num:] = arr - self.const_arr
        else:
            latent_arr[self.theta_num:] = arr
        latent_var = Variable(
            torch.from_numpy(latent_arr).float(),
            requires_grad=True
        )
        tmp_params.append(latent_var)  # latent vars as params

        # get vars and optimizer
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

        sigma_tensor = torch.from_numpy(self.sigma_arr).float()
        sigma_var = Variable(
            sigma_tensor, requires_grad=True
        )
        tmp_params.append(sigma_var)

        optimizer = optim.SGD(tmp_params, lr=0.5)

        last_loss = None
        for i in range(_max_iter):
            logging.info('ITER: {}/{}'.format(i + 1, _max_iter))

            x_list = []
            for j in range(self.theta_num):
                shift = j + 1
                x_list.append(latent_var[self.theta_num - shift: -shift])
            x_var = torch.stack(x_list)

            # get loss
            tmp = torch.mm(theta_var, x_var)
            if self.use_const:
                tmp += const_var
            loss = -Normal(0, sigma_var).log_prob(y_var - tmp).mean()

            # update
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # check loss change
            loss_value = loss.data[0]
            if last_loss is not None:
                if abs(last_loss - loss_value) < _tol:
                    break
            last_loss = loss_value
        else:
            logging.warning('Maybe not converged')

        # update array
        self.theta_arr = theta_var.data.numpy()[0]
        if self.use_const:
            self.const_arr = const_var.data.numpy()
        self.sigma_arr = sigma_var.data.numpy()[0]

    def getThetas(self) -> np.ndarray:
        return self.theta_arr

    def getConst(self) -> typing.Union[None, np.ndarray]:
        if self.use_const:
            return self.const_arr
        else:
            return None
