from ase.io import read
from pyxtal_ff.descriptors.SNAP import SO4_Bispectrum as snap

# initial silicon crystal
si = read('../lt_quartz.cif')

f = snap({'Si':14, 'O':8}, lmax=3, rcut=4.9)
x = f.calculate(si)
print(x['x'])


