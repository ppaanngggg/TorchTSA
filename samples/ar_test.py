import logging

import pyflux as pf

from TorchTSA.model import ARModel
from TorchTSA.simulate import ARSim

logging.basicConfig(level=logging.INFO)

# simulate data
ar_sim = ARSim(_phi_arr=[0.8, -0.2], _const=0.0)
sim_data = ar_sim.sample_n(1000)

ar_model = ARModel(_phi_num=2, _use_const=True)
ar_model.fit(sim_data)
print(ar_model.getPhis(), ar_model.getConst(), ar_model.getSigma())
print(ar_model.predict(sim_data))

pf_model = pf.ARIMA(data=sim_data, ar=2, ma=0, integ=0)
pf_ret = pf_model.fit("MLE")
pf_ret.summary()
print(pf_model.predict())