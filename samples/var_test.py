import time

import numpy as np

from TorchTSA.model import VARModel
from TorchTSA.simulate import VARSim

B = np.array([
    [0.5, -0.12],
    [0.35, 0.1]
])

sim = VARSim(
    _phi_arr=np.array([
        [[0.5, 0.3],
         [-0.3, 0.4]],
        [[0.3, -0.2],
         [0.2, 0.1]]
    ]),
    _mu_arr=np.array([0.1, 0.1]),
    _cov_arr=B.dot(B.T)
)
sim_data = sim.sample_n(2000)

var_model = VARModel(2, 2)
start_time = time.time()
var_model.fit(sim_data)
print(time.time() - start_time)

print(B.dot(B.T))
print(var_model.getPhis())
print(var_model.getMu())
print(var_model.getCov())
