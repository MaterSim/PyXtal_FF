from ase.io import read, write

struc = read("Si.cif", format='cif')
write("Si.data", struc, format='lammps-data')
