import logging
import typing

import numpy as np
import torch
import torch.optim as optim
from torch.autograd import Variable
from torch.distributions import Normal


def logit(x):
    return np.log(x) - np.log(1 - x)


def ilogit(x):
    return 1. / (1. + np.exp(-x))


class GARCHModel:

    def __init__(
            self,
            _alpha_num: int = 1,
            _beta_num: int = 1,
            _use_mu: bool = True,
    ):
        assert _alpha_num > 0
        assert _beta_num >= 0

        # fitter params
        self.alpha_num = _alpha_num
        self.beta_num = _beta_num
        self.use_mu = _use_mu

        # model params
        self.mu_arr: np.ndarray = None
        self.logit_alpha_arr: np.ndarray = None
        self.logit_beta_arr: np.ndarray = None
        self.log_const_arr: np.ndarray = None

        # latent for MA part
        self.latent_arr: np.ndarray = None

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

        # init params and latent
        arr = np.array(_arr)

        # get vars and optimizer
        # 0. y_var
        y_var = Variable(
            torch.from_numpy(arr[self.alpha_num:]).float()
        )
        params = []
        # 1. mu_var
        if self.use_mu:
            self.mu_arr = np.mean(_arr, keepdims=True)
        else:
            self.mu_arr = np.zeros(1)
        if self.use_mu:
            mu_var = Variable(
                torch.from_numpy(self.mu_arr).float(),
                requires_grad=True
            )
            params.append(mu_var)
        else:
            mu_var = Variable(
                torch.from_numpy(self.mu_arr).float()
            )
        init_value = 1. / np.sqrt(len(arr))
        # 2. logit_alpha_var
        if self.logit_alpha_arr is None:
            self.logit_alpha_arr = np.empty(self.alpha_num)
            self.logit_alpha_arr.fill(logit(init_value))
        logit_alpha_var = Variable(
            torch.from_numpy(self.logit_alpha_arr).float().unsqueeze(0),
            requires_grad=True
        )
        params.append(logit_alpha_var)
        # 3. logit_beta_arr
        if self.beta_num > 0:
            if self.logit_beta_arr is None:
                self.logit_beta_arr = np.empty(self.beta_num)
                self.logit_beta_arr.fill(logit(init_value))
            logit_beta_var = Variable(
                torch.from_numpy(self.logit_beta_arr).float().unsqueeze(0),
                requires_grad=True
            )
            params.append(logit_beta_var)
        # 4. log_const_arr
        if self.log_const_arr is None:
            self.log_const_arr = np.empty(1)
            self.log_const_arr.fill(np.log(init_value))
        log_const_var = Variable(
            torch.from_numpy(self.log_const_arr).float(),
            requires_grad=True
        )
        params.append(log_const_var)

        optimizer = optim.LBFGS(params, max_iter=_max_iter)

        if self.beta_num > 0:  # alloc latent_arr
            self.latent_arr = np.empty(
                len(_arr) + self.beta_num - self.alpha_num
            )

        def closure():
            # ar_x_var
            tmp_mu = mu_var.data.numpy()
            square_arr = (arr - tmp_mu) ** 2
            ar_x_arr = self.stack_delay_arr(square_arr, self.alpha_num)
            ar_x_var = Variable(torch.from_numpy(ar_x_arr).float())
            if self.beta_num > 0:
                # estimate latent_arr
                tmp_arr = ilogit(
                    logit_alpha_var.data.numpy()
                ).dot(ar_x_arr) + np.exp(log_const_var.data.numpy())  # ar and const
                self.latent_arr[self.beta_num:] = tmp_arr
                inv_beta_arr = ilogit(logit_beta_var.data.numpy())[::-1]
                padding_value = max(0, tmp_mu / (1 - inv_beta_arr.sum()))
                self.latent_arr[:self.beta_num] = padding_value
                for i in range(self.beta_num, len(self.latent_arr)):
                    begin_i = i - self.beta_num
                    self.latent_arr[i] = self.latent_arr[i] + (
                            self.latent_arr[begin_i:i] * inv_beta_arr
                    ).sum()
                # ma_x_var
                ma_x_arr = self.stack_delay_arr(self.latent_arr, self.beta_num)
                ma_x_var = Variable(torch.from_numpy(ma_x_arr).float())

            # print(
            #     'PARAMS:',
            #     ilogit(logit_alpha_var.data.numpy()),
            #     ilogit(logit_beta_var.data.numpy()),
            #     np.exp(log_const_var.data.numpy()),
            #     mu_var.data.numpy()
            # )
            # get the ML loss
            optimizer.zero_grad()
            out = torch.mm(
                torch.sigmoid(logit_alpha_var), ar_x_var
            )
            if self.beta_num > 0:
                out = out + torch.mm(
                    torch.sigmoid(logit_beta_var), ma_x_var
                )
            out = out + torch.exp(log_const_var)
            loss = -Normal(
                mu_var, torch.sqrt(out)
            ).log_prob(y_var).mean()
            loss.backward()
            # print(
            #     'GRAD:',
            #     logit_alpha_var.grad.data.numpy(),
            #     logit_beta_var.grad.data.numpy(),
            #     log_const_var.grad.data.numpy(),
            #     mu_var.grad.data.numpy()
            # )
            logging.info('loss: {}'.format(loss.data.numpy()[0]))

            return loss

        optimizer.step(closure)

        self.mu_arr = mu_var.data.numpy()
        self.log_const_arr = log_const_var.data.numpy()
        self.logit_alpha_arr = logit_alpha_var.data.numpy()[0]
        if self.beta_num > 0:
            self.logit_beta_arr = logit_beta_var.data.numpy()[0]

    def getAlphas(self) -> np.ndarray:
        return ilogit(self.logit_alpha_arr)

    def getBetas(self) -> np.ndarray:
        return ilogit(self.logit_beta_arr)

    def getConst(self) -> np.ndarray:
        return np.exp(self.log_const_arr)

    def getMu(self) -> np.ndarray:
        return self.mu_arr
