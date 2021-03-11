from ase import units
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.nvtberendsen import NVTBerendsen
from ase.io.trajectory import Trajectory
from ase.md import MDLogger
from ase.build import bulk
from pyxtal_ff import PyXtal_FF
from pyxtal_ff.calculator import PyXtalFFCalculator, optimize
from optparse import OptionParser

class Params():
    def __init__(self,):
        self.time_step = 1 # fs
        self.run_step = 50000 # 50ps
        self.temperature = 300 # K
        self.taut = 0.5*1000
        self.save_interval = 10

if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option("-f", "--file", dest="file", 
            help="pretrained file from pyxtal_ff, REQUIRED",
            metavar="file")

    (options, args) = parser.parse_args()

    p = Params()
    # initial structure and calculator
    si = bulk('Si', 'diamond', a=5.0, cubic=True)

    ff = PyXtal_FF(model={'system': ["Si"]}, logo=False)
    ff.run(mode='predict', mliap=options.file)
    calc = PyXtalFFCalculator(ff=ff)
    si.set_calculator(calc)

    # geometry optimization
    si = optimize(si, box=True)
    print('equlirum cell para: ', si.get_cell()[0][0])
    print('equlirum energy: ', si.get_potential_energy())
    print('equlirum stress', -si.get_stress()/units.GPa)

    # Build the 2*2*2 supercell
    atoms = si*2
    atoms.set_calculator(calc)
    print(atoms)

    # NVT MD simulation
    dyn = NVTBerendsen(atoms, p.time_step * units.fs, p.temperature, taut=p.taut*units.fs)
    MaxwellBoltzmannDistribution(atoms, p.temperature * units.kB)
    dyn.attach(MDLogger(dyn, atoms, 'md.log', header=True, stress=False, mode="w"), 
               interval=p.save_interval)
    traj = Trajectory('md.traj', 'w', atoms)
    dyn.attach(traj.write, interval=p.save_interval)

    dyn.run(p.run_step)

