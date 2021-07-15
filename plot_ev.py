from ase.io import read
import numpy as np
from ase.db import connect
import matplotlib.pyplot as plt
from monty.serialization import loadfn
from pyxtal_ff import PyXtal_FF
from pyxtal_ff.calculator import PyXtalFFCalculator

factor = 1.
uncertainty_type = 'epistemic'
uncertainty_type = 'aleatoric'

# Initialize PyXtal_FF calculator
mliap = "potentials/20-20-checkpoint.pth"
ff = PyXtal_FF(model={'system': ["Si"]}, logo=False)
ff.run(mode='predict', mliap=mliap)
calc = PyXtalFFCalculator(ff=ff)

# Plot ground state Si
dft_si_gs, volumes = [], []
gs_pred_eng, gs_aleatoric, gs_epistemic = [], [], []
with connect('ase_database/si_ground_state.db') as db:
    for i, row in enumerate(db.select()):
        struc = db.get_atoms(row.id)

        # pred
        struc.set_calculator(calc)
        gs_pred_eng.append(struc.get_potential_energy()/len(struc))
        gs_aleatoric.append(struc.calc.results['aleatoric']/len(struc))
        gs_epistemic.append(struc.calc.results['epistemic']/len(struc))
        
        # dft
        dft_si_gs.append(row.data['energy']/len(struc))
        volumes.append(struc.get_volume()/len(struc))
        
        # there is a duplicate structure in si_ground_state.db
        if i == 10: break

uncertain = gs_aleatoric if uncertainty_type == 'aleatoric' else gs_epistemic
plt.plot(volumes, dft_si_gs, color='r', label='Ground State True')
plt.errorbar(volumes, gs_pred_eng, yerr=np.array(uncertain)*factor, color='g', label='Ground state Prediction', alpha=0.5)

# Plot Orthorhombic
dft_si_ortho, volumes = [], []
ortho_pred_eng, ortho_aleatoric, ortho_epistemic = [], [], []
with connect('ase_database/si_orthrombic.db') as db:
    for i, row in enumerate(db.select()):
        struc = db.get_atoms(row.id)
        struc.set_calculator(calc)
        volumes.append(struc.get_volume()/len(struc))

        dft_si_ortho.append(row.data['energy']/len(struc))
        ortho_pred_eng.append(struc.get_potential_energy()/len(struc))
        ortho_aleatoric.append(struc.calc.results['aleatoric']/len(struc))
        ortho_epistemic.append(struc.calc.results['epistemic']/len(struc))

uncertain = ortho_aleatoric if uncertainty_type == 'aleatoric' else ortho_epistemic
plt.plot(volumes, dft_si_ortho, color='b', label='Orthorhombic True')
plt.errorbar(volumes, ortho_pred_eng, yerr=np.array(uncertain)*factor, color='k', label='Orthorhombic Prediction', alpha=0.5)
plt.xlabel('Volume (A^3/atom)')
plt.ylabel('Energy (eV/atom)')
plt.legend()
plt.savefig("plot.png", dpi=300)
