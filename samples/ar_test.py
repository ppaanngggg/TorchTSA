import logging

from TorchTSA.model import ARModel
from TorchTSA.simulate import ARSim

logging.basicConfig(level=logging.INFO)

ar_sim = ARSim(_theta_arr=[0.3, 0.2], _const=0.0)
sim_arr = ar_sim.sample_n(1000)

ar_model = ARModel(_theta_num=2, _use_const=True)

ar_model.fit(sim_arr)
print(ar_model.thetas(), ar_model.const())
