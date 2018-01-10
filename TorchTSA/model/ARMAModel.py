import logging
import typing

import numpy as np
import torch
import torch.optim as optim
from torch.autograd import Variable
from torch.distributions import Normal


class ARMAModel:

    def __init__(
            self,
            _phi_num: int = 1,
            _theta_num: int = 1,
            _use_const: bool = True
    ):
        self.phi_num = _phi_num  # len of phi_arr
        self.theta_num = _theta_num  # len of theta_arr
        self.use_const = _use_const

        self.phi_arr: np.ndarray = np.zeros(self.phi_num)
        self.theta_arr: np.ndarray = np.zeros(self.theta_num)
        self.const_arr: np.ndarray = None
        self.sigma_arr: np.ndarray = None

        self.latent_arr: np.ndarray = None

    def fit(
            self, _arr: typing.Sequence[float],
            _max_iter: int = 20,
    ):
        assert len(_arr) > self.phi_num
        assert len(_arr) > self.theta_num

        # y_var
        arr = np.array(_arr)
        y_var = Variable(
            torch.from_numpy(arr[self.phi_num:]).float()
        )

        # ar_x_var
        ar_x_list = []
        for i in range(self.phi_num):
            shift = i + 1
            ar_x_list.append(arr[self.phi_num - shift: -shift])
        ar_x_arr = np.stack(ar_x_list)
        ar_x_var = Variable(torch.from_numpy(ar_x_arr).float())

        # get vars and optimizer
        params = []
        # 1. const
        if self.const_arr is None:
            if self.use_const:
                mean_value = np.mean(arr)
                self.const_arr = np.array([mean_value])
            else:
                self.const_arr = np.zeros(1)
        if self.use_const:
            const_var = Variable(
                torch.from_numpy(self.const_arr).float(),
                requires_grad=True
            )
            params.append(const_var)
        else:
            const_var = Variable(
                torch.from_numpy(self.const_arr).float()
            )
        # 2. sigma
        if self.sigma_arr is None:
            std_value = np.std(arr)
            self.sigma_arr = np.array([std_value])
        sigma_var = Variable(
            torch.from_numpy(self.sigma_arr).float(),
            requires_grad=True
        )
        params.append(sigma_var)
        # 3. phi
        if self.phi_num > 0:
            phi_var = Variable(
                torch.from_numpy(self.phi_arr).float().unsqueeze(0),
                requires_grad=True
            )
            params.append(phi_var)
        # 4. theta
        if self.theta_num > 0:
            theta_var = Variable(
                torch.from_numpy(self.theta_arr).float().unsqueeze(0),
                requires_grad=True
            )
            params.append(theta_var)

        normal = Normal(0, sigma_var)
        optimizer = optim.LBFGS(params, max_iter=_max_iter)

        # init latent arr
        self.latent_arr = np.zeros(
            len(arr) + self.theta_num - self.phi_num
        )

        def closure():
            # update latent var
            tmp_phi = phi_var.data.numpy()
            tmp_theta = theta_var.data.numpy()
            tmp_const = const_var.data.numpy()
            tmp_y = arr[self.phi_num:]
            ma_x_list = []
            for i in range(self.theta_num):
                shift = i + 1
                ma_x_list.append(
                    self.latent_arr[self.theta_num - shift: -shift]
                )
            ma_x_arr = np.stack(ma_x_list)

            self.latent_arr[self.theta_num:] = \
                tmp_y - tmp_phi.dot(ar_x_arr) - \
                tmp_theta.dot(ma_x_arr) - tmp_const

            ma_x_list = []
            for i in range(self.theta_num):
                shift = i + 1
                ma_x_list.append(
                    self.latent_arr[self.theta_num - shift: -shift]
                )
            ma_x_var = Variable(
                torch.from_numpy(np.stack(ma_x_list)).float()
            )
            # loss
            optimizer.zero_grad()
            out = torch.mm(phi_var, ar_x_var) + torch.mm(theta_var, ma_x_var)
            loss = -normal.log_prob(out + const_var - y_var).mean()
            logging.info('loss: {}'.format(loss.data.numpy()[0]))
            loss.backward()
            return loss

        optimizer.step(closure)

        # update array
        if self.use_const:
            self.const_arr = const_var.data.numpy()
        self.sigma_arr = sigma_var.data.numpy()
        if self.phi_num > 0:
            self.phi_arr = phi_var.data.numpy()[0]
        if self.theta_num > 0:
            self.theta_arr = theta_var.data.numpy()[0]

    def getPhis(self) -> np.ndarray:
        return self.phi_arr

    def getThetas(self) -> np.ndarray:
        return self.theta_arr

    def getConst(self) -> np.ndarray:
        return self.const_arr

    def getSigma(self) -> np.ndarray:
        return self.sigma_arr
