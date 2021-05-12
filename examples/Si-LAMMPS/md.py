""" 
Example of MD simulation
"""

from time import time
from optparse import OptionParser
from ase.build import bulk
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.verlet import VelocityVerlet
from ase.md.langevin import Langevin
from ase import units
from pyxtal_ff import PyXtal_FF
from pyxtal_ff.calculator import PyXtalFFCalculator

import warnings
warnings.simplefilter("ignore")

parser = OptionParser()
parser.add_option("-f", "--file", dest="file",
                  help="pretrained file from pyxtal_ff, REQUIRED",
                  metavar="file")

(options, args) = parser.parse_args()


def printenergy(a, it, t0):
    """Function to print the potential, kinetic and total energy"""
    epot = a.get_potential_energy() / len(a)
    ekin = a.get_kinetic_energy() / len(a)
    t_now = time()
    print('Step: %4d [%6.2f]: Epot = %.3feV  Ekin = %.3feV (T=%3.0fK)  '
          'Etot = %.3feV ' % (\
          it, t_now-t0, epot, ekin, ekin / (1.5 * units.kB), epot + ekin))
    return t_now

ff = PyXtal_FF(model={'system': ["Si"]}, logo=False)
ff.run(mode='predict', mliap=options.file)
calc = PyXtalFFCalculator(ff=ff)

si = bulk('Si', 'diamond', a=5.659, cubic=True)
si = si*5
print("MD simulation for ", len(si), " atoms")
si.set_calculator(calc)

MaxwellBoltzmannDistribution(si, 1000*units.kB)
dyn = Langevin(si, timestep=5 * units.fs, temperature_K=1000, friction=0.02)
#dyn = VelocityVerlet(si, 5*units.fs)  # 2 fs time step.
t0 = time()
for i in range(10):
    dyn.run(steps=1)
    t_now = printenergy(si, i, t0)
    t0 = t_now

