import numpy as np
from ase.io import read
from ase import units
from lammps import lammps

from pyxtal_ff import PyXtal_FF
from pyxtal_ff.calculator import PyXtalFFCalculator
from pyxtal_ff.calculator.lammpslib import LAMMPSlib
import warnings; warnings.simplefilter("ignore")

#np.random.seed(0)
dicts = {"sna": "SiO2-snap",
         "so3": "SiO2-so3"}

for des in dicts.keys():
    print("\n", des)
    folder = dicts[des]
    mliap  = folder + "/30-30-checkpoint.pth"
    lmpiap = folder + "/NN_weights.txt"
    lmpdes = folder + "/DescriptorParam.txt"
    
    # ase pyxtal_ff calculator
    ff = PyXtal_FF(model={'system': ["Si", "O"]}, logo=False)
    ff.run(mode='predict', mliap=mliap)
    calc_pff = PyXtalFFCalculator(ff=ff)
    
    # ase lammps calculatoor
    lammps_name=''
    comm=None
    log_file='lammps.log'
    cmd_args = ['-echo', 'log', '-log', log_file,
                '-screen', 'none', '-nocite']
    lmp = lammps(lammps_name, cmd_args, comm)
    
    parameters = ["mass 1 28.0855",
                  "mass 2 15.9999",
                  "pair_style hybrid/overlay &",
                  "mliap model nn " + lmpiap + " descriptor " + des + " " + lmpdes + " &",
                  "zbl 1.0 2.0",
                  "pair_coeff 1 1 zbl 8.0 8.0",
                  "pair_coeff 2 2 zbl 14.0 14.0",
                  "pair_coeff 1 2 zbl 8.0 14.0",
                  "pair_coeff * * mliap O Si",
                  ]
    
    calc_lmp = LAMMPSlib(lmp=lmp, lmpcmds=parameters)
    
    # check for single configuration
    for i in range(20):
        # initial silicon crystal
        si = read('lt_quartz.cif')
        # set the ordering
        si.set_tags([2]*3+[1]*6) 
        si.positions += 0.25*(np.random.random_sample([len(si),3])-0.5)
        eng = []
        force = []
        stress = []
        for calc in [calc_lmp, calc_pff]:
            si.set_calculator(calc)
            eng.append(si.get_potential_energy())
            force.append(si.get_forces())
            stress.append(si.get_stress())
        #calc.print_energy()
        e_diff = eng[0]-eng[1]
        f_diff = np.linalg.norm((force[0] - force[1]).flatten())
        s_diff = np.linalg.norm((stress[0] - stress[1]).flatten())/units.GPa
    
        print("{:3d} {:8.3f} eV {:8.3f} GPa {:8.3f} {:8.3f} {:8.3f}".format(i, eng[0], -stress[0][0]/units.GPa, e_diff, f_diff, s_diff))
        if abs(e_diff) > 1e-2 or f_diff > 1e-2 or s_diff > 1e-2:
            print("eng: ", eng[0], eng[1])
            print("Forces from LAMMPS and PyXtal_FF")
            for f1, f2 in zip(force[0], force[1]):
                print("{:8.4f} {:8.4f} {:8.4f} -> {:8.4f} {:8.4f} {:8.4f} -> {:8.4f} {:8.4f} {:8.4f}".format(*f1, *f2, *(f1-f2)))
            print("\n Breakdown of Pyxtal_FF")
            calc.print_all()
            break
