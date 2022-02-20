import numpy as np
from ase import units
from ase.optimize import LBFGS
from ase.optimize.fire import FIRE
from ase.constraints import ExpCellFilter
from ase.calculators.calculator import Calculator, all_changes
from ase.spacegroup.symmetrize import FixSymmetry, check_symmetry
from pyxtal_ff import PyXtal_FF
from pyxtal_ff.utilities import compute_descriptor
from pyxtal_ff.utilities.base_potential import ZBL
np.set_printoptions(formatter={'float': '{: 8.4f}'.format})


class PyXtalFFCalculator(Calculator):
    implemented_properties = ['energy', 'forces', 'stress']
    nolabel = True

    def __init__(self, style='ase', **kwargs):
        self.style = style
        Calculator.__init__(self, **kwargs)

    def calculate(self, atoms=None,
                  properties=['energy'],
                  system_changes=all_changes):

        Calculator.calculate(self, atoms, properties, system_changes)

        #chem_symbols = list(set(atoms.get_chemical_symbols()))
        #self.ff = PyXtal_FF(model={'system': chem_symbols}, logo=self.parameters.logo)
        #self.ff.run(mode='predict', mliap=self.parameters.mliap)

        # base potential
        if self.parameters.ff._descriptors['base_potential']:
            self.base_potential = ZBL(self.parameters.ff._descriptors['base_potential']['inner'],
                                      self.parameters.ff._descriptors['base_potential']['outer'],
                                      atomic_energy=True)
            base_results = self.base_potential.calculate(atoms)
            base_energy = base_results['energy']
            base_forces = base_results['force']
            base_stress = base_results['stress'] # eV/A^3
            base_energies = base_results['energies']
        else:
            base_energy = 0
            base_forces = np.zeros([len(atoms), 3])
            base_stress = np.zeros([6])
            base_energies = 0.

        desp = compute_descriptor(self.parameters.ff._descriptors, atoms)
        energies, forces, stress = self.parameters.ff.model.calculate_properties(desp, bforce=True, bstress=True)

        self.desp = desp
        self.results['energies'] = energies + base_energies
        self.results['energy'] = energies.sum() + base_energy
        self.results['free_energy'] = energies.sum() + base_energy
        self.results['forces'] = forces + base_forces

        # pyxtal_ff and lammps uses: xx, yy, zz, xy, xz, yz
        # ase uses: xx, yy, zz, yz, xz, xy
        # from eV/A^3 to GPa 
        self.results['stress_zbl'] = base_stress/units.GPa
        self.results['energy_zbl'] = base_energy
        self.results['forces_zbl'] = base_forces
        self.results['stress_ml'] = stress 
        self.results['energy_ml'] = energies.sum()
        self.results['forces_ml'] = forces


        # ase counts the stress differently
        if self.style == 'ase':
            self.results['stress'] = -(stress * units.GPa + base_stress)[[0, 1, 2, 5, 4, 3]]
        else:
            self.results['stress'] = self.results['stress_zbl'] + self.results['stress_ml']

    def __str__(self):
        s = "\nASE calculator with pyxtal_ff force field\n"
        return s

    def __repr__(self):
        return str(self)

    def print_stresses(self):
        print("stress_ml (GPa, xx, yy, zz, xy, xz, yz):", self.results["stress_ml"])
        print("stress_zbl(GPa, xx, yy, zz, xy, xz, yz):", self.results['stress_zbl'])

    def print_energy(self):
        print("energy_ml (eV):", self.results["energy_ml"])
        print("energy_zbl(eV):", self.results['energy_zbl'])

    def print_forces(self):
        print("forces (eV/A)")
        for f1, f2 in zip(self.results["forces_ml"], self.results['forces_zbl']):
            print("{:8.3f} {:8.3f} {:8.3f} -> {:8.3f} {:8.3f} {:8.3f}".format(*f1, *f2))

    def print_all(self):
        self.print_energy()
        self.print_forces()
        self.print_stresses()


def elastic_properties(C):
    Kv = C[:3,:3].mean()
    Gv = (C[0,0]+C[1,1]+C[2,2] - (C[0,1]+C[1,2]+C[2,0]) + 3*(C[3,3]+C[4,4]+C[5,5]))/15
    Ev = 1/((1/(3*Gv))+(1/(9*Kv)))
    vv  = 0.5*(1-((3*Gv)/(3*Kv+Gv))); 

    S = np.linalg.inv(C)
    Kr = 1/((S[0,0]+S[1,1]+S[2,2])+2*(S[0,1]+S[1,2]+S[2,0])) 
    Gr = 15/(4*(S[0,0]+S[1,1]+S[2,2])-4*(S[0,1]+S[1,2]+S[2,0])+3*(S[3,3]+S[4,4]+S[5,5])) 
    Er = 1/((1/(3*Gr))+(1/(9*Kr))) 
    vr = 0.5*(1-((3*Gr)/(3*Kr+Gr))) 

    Kh = (Kv+Kr)/2    
    Gh = (Gv+Gr)/2    
    Eh = (Ev+Er)/2    
    vh = (vv+vr)/2   
    return Kv, Gv, Ev, vv, Kr, Gr, Er, vr, Kh, Gh, Eh, vh

def optimize(atoms, sym=True, box=False, P=0.0, method='FIRE', fmax=0.01, steps=1000, logfile='ase.log'):
    """
    Geometry relaxation

    Args:
        Atoms: ase atoms
        sym: whether or not fix symmetry
        box: whether or not relax box
        P: external pressure in GPa
        method: optimization method
        fmax: toleration force
        steps: maximum number of steps
        logfile: output of the log file
    """
    if sym:
        atoms.set_constraint(FixSymmetry(atoms))
    if box:
        ecf = ExpCellFilter(atoms, scalar_pressure=P*units.GPa)
        if method == 'FIRE':
            dyn = FIRE(ecf, logfile=logfile)
        else:
            dyn = LBFGS(ecf, logfile=logfile)
    else:
        if method == 'FIRE':
            dyn = FIRE(atoms, logfile=logfile)
        else:
            dyn = FIRE(atoms, logfile=logfile)

    dyn.run(fmax=fmax, steps=steps)
    atoms.set_constraint()
    return atoms

if  __name__ == "__main__":

    from optparse import OptionParser
    from ase.build import bulk

    parser = OptionParser()
    parser.add_option("-f", "--file", dest="file",
                      help="pretrained file from pyxtal_ff, REQUIRED",
                      metavar="file")

    (options, args) = parser.parse_args()
    print(options.file)
    calc = PyXtalFFCalculator(mliap=options.file, logo=False)
    si = bulk('Si', 'diamond', a=5.459, cubic=True)
    si.set_calculator(calc)
    print(si.get_potential_energy())
    print(si.get_forces())
    print(si.get_stress())

    box = mushybox(si)
    dyn = BFGS(box)
    dyn.run(fmax=0.01)
    print('equlirum cell para: ', si.get_cell()[0][0])
    print('equlirum energy: ', si.get_potential_energy())

    Cijs, names, C = elastic_tensor(si, calc)
    for name, Cij in zip(names, Cijs):
        print("{:s}: {:8.2f}(GPa)".format(name, Cij))

    print(elastic_properties(C))
