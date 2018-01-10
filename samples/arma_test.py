import matplotlib.pyplot as plt

from TorchTSA.simulate import ARMASim

arma_sim = ARMASim(
    _phi_arr=(0.8,),
    _theta_arr=(-0.5, 0.3),
)
sim_data = arma_sim.sample_n(1000)

plt.plot(sim_data)
plt.show()
