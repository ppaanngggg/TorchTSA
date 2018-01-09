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

        self.const_arr: np.ndarray = None
        self.theta_arr: np.ndarray = np.zeros(self.theta_num)
        self.sigma_arr: np.ndarray = None  # MLE

        self.latent_arr: np.ndarray = None

    def fit(
            self, _arr: typing.Sequence[float],
            _max_iter: int = 20,
    ):
        assert len(_arr) > self.theta_num

        # get y_var
        arr = np.array(_arr)
        y_var = Variable(torch.from_numpy(arr).float())

        # get vars and optimizer
        tmp_params = []
        # 1. const
        if self.const_arr is None:  # init const if not
            if self.use_const:
                mean_value = np.mean(arr)
                self.const_arr = np.array([mean_value])
            else:
                self.const_arr = np.zeros(1)

        const_tensor = torch.from_numpy(self.const_arr).float()
        if self.use_const:  # if estimate const
            const_var = Variable(
                const_tensor, requires_grad=True
            )
            tmp_params.append(const_var)
        else:
            const_var = Variable(const_tensor)
        # 2. theta
        theta_tensor = torch.from_numpy(
            self.theta_arr
        ).float().unsqueeze(0)  # expand one dim
        theta_var = Variable(
            theta_tensor, requires_grad=True
        )
        tmp_params.append(theta_var)
        # 3. sigma
        if self.sigma_arr is None:  # init sigma if not
            std_value = np.std(arr)
            self.sigma_arr = np.array([std_value])
        sigma_tensor = torch.from_numpy(self.sigma_arr).float()
        sigma_var = Variable(
            sigma_tensor, requires_grad=True
        )
        tmp_params.append(sigma_var)

        normal = Normal(0, sigma_var)
        optimizer = optim.LBFGS(tmp_params, max_iter=_max_iter)

        # init latent arr
        self.latent_arr = np.zeros(self.theta_num + len(arr))

        def closure():
            # update latent arr
            new_latent_arr = np.zeros(self.theta_num + len(arr))
            tmp_theta = theta_var.data.numpy()[0]
            tmp_theta = tmp_theta[::-1]
            tmp_const = const_var.data.numpy()[0]
            tmp_arr = arr - tmp_const
            for i, v in enumerate(tmp_arr):
                new_latent_arr[self.theta_num + i] = v - np.sum(
                    tmp_theta * self.latent_arr[i:self.theta_num + i]
                )
            self.latent_arr = new_latent_arr
            # x var
            x_list = []
            for i in range(self.theta_num):
                shift = i + 1
                x_list.append(
                    self.latent_arr[self.theta_num - shift: -shift]
                )
            x_var = Variable(
                torch.from_numpy(np.stack(x_list)).float()
            )
            # loss
            optimizer.zero_grad()
            out = torch.mm(theta_var, x_var) + const_var - y_var
            loss = -normal.log_prob(out).mean()
            logging.info('loss: {}'.format(loss.data.numpy()[0]))
            loss.backward()
            return loss

        optimizer.step(closure)

        # update array
        self.theta_arr = theta_var.data.numpy()[0]
        if self.use_const:
            self.const_arr = const_var.data.numpy()
        self.sigma_arr = sigma_var.data.numpy()

    def getThetas(self) -> np.ndarray:
        return self.theta_arr

    def getConst(self) -> np.ndarray:
        return self.const_arr

    def getSigma(self) -> np.ndarray:
        return self.sigma_arr
