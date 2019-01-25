# LAMMPS testing for calculating bispectrum components

# Initialize

clear
units metal
dimension 3
boundary p p p
atom_style atomic
atom_modify map array

# Generating box and atomic postions

read_data data.lammps

# Do I need potential?

# Calculate the bispectrum

compute b all sna/atom 1.4 0.99363 6 2.0 2.4 0.75 1.0 diagonal 3 rmin0 0.0

