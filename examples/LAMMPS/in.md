# Run NPT MD simulation for Si at 500 K.
variable nsteps index 100

# NOTE: These are not intended to represent real materials

units           metal

atom_style      atomic
atom_modify     map array
boundary        p p p
atom_modify     sort 0 0.0
read_data Si.data

# temperature

variable t equal 500.

mass            *       28.06

# Potential
pair_style mliap model nn Si-snap/NN_weights.txt descriptor sna Si-snap/DescriptorParams.txt
pair_coeff * * Si Si

# Set-up output

compute  eatom all pe/atom
compute  energy all reduce sum c_eatom

compute  satom all stress/atom NULL
compute  str all reduce sum c_satom[1] c_satom[2] c_satom[3]
variable press equal (c_str[1]+c_str[2]+c_str[3])/(3*vol)

thermo_style    custom step temp epair c_energy etotal press v_press
thermo          10
thermo_modify norm yes

# Set up NVE run

timestep 0.5e-3
neighbor 1.0 bin
neigh_modify once no every 1 delay 0 check yes

# Run MD

velocity all create $t 5287287 loop geom
fix 1 all npt temp ${t} ${t} 0.2 iso 0.0 0.0 2
run ${nsteps}
