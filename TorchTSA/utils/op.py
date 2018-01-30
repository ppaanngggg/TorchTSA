import typing

import numpy as np


def stack_delay_arr(
        _arr: typing.Sequence[float], _num: int
) -> np.ndarray:
    ret_list = []
    for i in range(_num):
        shift = i + 1
        ret_list.append(_arr[_num - shift: -shift])
    return np.stack(ret_list)
