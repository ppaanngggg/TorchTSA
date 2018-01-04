import logging

from statsmodels.tsa.ar_model import AR

from TorchTSA.model import ARModel
from TorchTSA.simulate import ARSim

logging.basicConfig(level=logging.INFO)

# simulate data
ar_sim = ARSim(_theta_arr=[0.3, -0.2], _const=0.0)
sim_data = ar_sim.sample_n(1000)

ar_model = ARModel(_theta_num=2, _use_const=True)
ar_model.fit(sim_data)
print(ar_model.getThetas(), ar_model.getConst())
print(ar_model.predict(sim_data))

sm_model = AR(sim_data)
sm_ret = sm_model.fit(2)
print(sm_ret.params)
print(sm_model.predict(
    sm_ret.params, len(sim_data), len(sim_data)
))
