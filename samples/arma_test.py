import logging
import time

import matplotlib.pyplot as plt
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
start_time = time.time()
arma_model.fit(sim_data)
print('fitting time:', time.time() - start_time)
print(
    arma_model.getPhis(), arma_model.getThetas(),
    arma_model.getConst(), arma_model.getSigma(),
)
print('predict value:', arma_model.predict(sim_data))

plt.subplot('211')
plt.plot(sim_data)
plt.subplot('212')
sim_line, = plt.plot(arma_sim.latent[3:], label='simulation')
arma_line, = plt.plot(arma_model.getLatent(), label='arma')
plt.legend(handles=[sim_line, arma_line])
plt.show()

pf_model = pf.ARIMA(data=sim_data, ar=1, ma=2, integ=0)
start_time = time.time()
pf_ret = pf_model.fit("MLE")
print('fitting time:', time.time() - start_time)
pf_ret.summary()
print(pf_model.predict())
