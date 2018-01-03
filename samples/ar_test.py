import matplotlib.pyplot as plt

from TorchTSA.simulate import ARSim

ar_sim = ARSim(0.9)
sim_arr = ar_sim.sample_n(1000)

plt.plot(sim_arr)
plt.show()
