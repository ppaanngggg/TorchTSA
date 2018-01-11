import logging

import pyflux as pf

from TorchTSA.model import ARMAModel
from TorchTSA.simulate import ARMASim

logging.basicConfig(level=logging.INFO)

arma_sim = ARMASim(
    _phi_arr=(0.8,),
    _theta_arr=(-0.5, 0.3),
)
sim_data = arma_sim.sample_n(1000)

arma_model = ARMAModel(1, 2)
arma_model.fit(sim_data)
print(
    arma_model.getPhis(), arma_model.getThetas(),
    arma_model.getConst(), arma_model.getSigma(),
)
print(arma_model.predict(sim_data))

pf_model = pf.ARIMA(data=sim_data, ar=1, ma=2, integ=0)
pf_ret = pf_model.fit("MLE")
pf_ret.summary()
print(pf_model.predict())