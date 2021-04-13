from ase.io import read
import numpy as np
import matplotlib.pyplot as plt
from monty.serialization import loadfn
from pyxtal_ff import PyXtal_FF
from pyxtal_ff.calculator import PyXtalFFCalculator
factor = 10.
uncertainty_type = 'epistemic'
uncertainty_type = 'aleatoric'

f = "Si.cif"
mliap = "Si-SNAP/20-20-checkpoint.pth"

ff = PyXtal_FF(model={'system': ["Si"]}, logo=False)
ff.run(mode='predict', mliap=mliap)
calc = PyXtalFFCalculator(ff=ff)

Si = read(f)

Pred_eng, Aleatoric, Epistemic = [], [], []
for fac in [0.90, 0.92, 0.94, 0.96, 0.98, 1.00, 1.02, 1.04, 1.06, 1.08, 1.10]:
    struc = Si.copy()
    pos = struc.get_scaled_positions().copy()
    struc.set_cell(fac*struc.cell)
    struc.set_scaled_positions(pos)
    struc.set_calculator(calc)
    
    Pred_eng.append(struc.get_potential_energy()/len(struc))
    Aleatoric.append(struc.calc.results['aleatoric']/len(struc))
    Epistemic.append(struc.calc.results['epistemic']/len(struc))

data = loadfn("data.json")
energy = np.array(data['energy']) / 8
volume = np.array(data['volume']) / 8

print(list(np.abs(Pred_eng - energy)))
print(Aleatoric)
print(Epistemic)

uncertain = Aleatoric if uncertainty_type == 'aleatoric' else Epistemic
plt.plot(volume, energy, color='r', label='True')
plt.errorbar(volume, Pred_eng, yerr=np.array(uncertain)*factor, color='g', label='Prediction', alpha=0.5)
plt.legend()
plt.savefig("plot.png", dpi=300)
