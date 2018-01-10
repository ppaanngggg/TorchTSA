import typing

from TorchTSA.simulate.ARMASim import ARMASim


class MASim(ARMASim):

    def __init__(
            self,
            _theta_arr: typing.Union[float, typing.Sequence[float]],
            _const: float = 0.0, _sigma: float = 1.0,
    ):
        if isinstance(_theta_arr, float) or isinstance(_theta_arr, int):
            _theta_arr = (_theta_arr,)

        super().__init__(
            _theta_arr=_theta_arr, _const=_const, _sigma=_sigma
        )
