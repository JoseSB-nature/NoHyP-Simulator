import matplotlib.pyplot as plt
import os
import numpy as np
import scienceplots

# plt.style.use("science")


files = [os.path.join('output', f) for f in os.listdir("output") if 'probe' in f]


fig, ax = plt.subplots(3,2,sharex=True)

for ind,f in enumerate(files[:6]):
    j = ind%2
    i = int(ind/2)
    x,t,h = np.loadtxt(f,skiprows=1,delimiter=';',usecols=(0,1,2),unpack=True)
    X = x[0]
    ax[i,j].plot(t[100:],h[100:])
    ax[i,j].grid()
    ax[i,j].set_title (f'x={X:.2f}m')

plt.show()
fig.savefig('img/probes.png',dpi=300)
