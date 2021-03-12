from pyxtal_ff import PyXtal_FF
from pyxtal_ff.calculator import PyXtalFFCalculator, optimize, elastic_properties
from pyxtal_ff.calculator.elasticity import fit_elastic_constants
from pyxtal_ff.calculator.phonon import Phonon
from ase.optimize import BFGS
from ase import units
import numpy as np

if  __name__ == "__main__":

    from optparse import OptionParser
    from ase.build import bulk

    parser = OptionParser()
    parser.add_option("-f", "--file", dest="file",
                      help="pretrained file from pyxtal_ff, REQUIRED",
                      metavar="file")

    (options, args) = parser.parse_args()

    # load calculator
    #calc = PyXtalFFCalculator(mliap=options.file, logo=False)

    ff = PyXtal_FF(model={'system': ["Si"]}, logo=False)
    ff.run(mode='predict', mliap=options.file)
    calc = PyXtalFFCalculator(ff=ff)


    # initial structure and calculator
    si = bulk('Si', 'diamond', a=5.0, cubic=True)
    si.set_calculator(calc)

    # geometry optimization
    si = optimize(si, box=True)
    print('equlirum cell para: ', si.get_cell()[0][0])
    print('equlirum energy: ', si.get_potential_energy())
    print('equlirum stress', -si.get_stress())

    #Elastic Properties
    C, C_err = fit_elastic_constants(si, symmetry='cubic', optimizer=BFGS)
    C /= units.GPa

    print("Bulk modulus, Shear modulus, Young's modulus, Poisson's ratio")
    k1, g1, e1, v1, k2, g2, e2, v2, k3, g3, e3, v3 = elastic_properties(C)
    print(k1, g1, e1, v1)
    print(k2, g2, e2, v2)
    print(k3, g3, e3, v3)

    # Phonon properties
    supercell_matrix=np.diag([2,2,2])
    ph = Phonon(si, calc, supercell_matrix=supercell_matrix) 
    force_constants = ph.get_force_constants()
    ph.phonon.set_force_constants(force_constants)

    ph.phonon.auto_band_structure()
    #path = [[[0, 0, 0], [0.5, 0, 0.5], [0.375, 0.375, 0.75], [0, 0, 0], [0.5, 0.5, 0.5]]]
    #labels = ["$\\Gamma$", "X", "K", "$\\Gamma$", "L"]
    #qpoints, connections = get_band_qpoints_and_path_connections(path, npoints=51)
    #ph.phonon.run_band_structure(qpoints, path_connections=connections, labels=labels)

    ph.phonon.set_mesh([10,10,10])
    ph.phonon.set_total_DOS()
    ph.phonon.plot_band_structure_and_dos().savefig('Si_phonon.png', dpi=300)
    print('The detected space group is {:s} with a tolerance of {:.2e}'.format(
          ph.phonon._symmetry.dataset['international'], ph.phonon._symprec))
