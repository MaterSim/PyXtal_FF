from monty.serialization import loadfn
import numpy as np
import matplotlib.pyplot as plt

data = loadfn("training.json")

energies, volumes = [], []
for d in data:
    energy = d['outputs']['energy']/d['num_atoms']
    structure = d['structure']
    volume = structure.volume/d['num_atoms']

    energies.append(energy)
    volumes.append(volume)

plt.scatter(volumes, energies)
plt.xlim(15, 30)
plt.show()

