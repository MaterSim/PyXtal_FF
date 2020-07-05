#python -m unittest pyxtal_ff/test_all.py
import unittest
import shutil
from ase.cluster.cubic import FaceCenteredCubic
from ase import Atoms
import numpy as np
from pkg_resources import resource_filename
from pyxtal_ff import PyXtal_FF
from pyxtal_ff.calculator import PyXtalFFCalculator
from ase.build import bulk

np.set_printoptions(formatter={'float': '{: 12.4f}'.format})

def get_rotated_struc(struc, angle=0, axis='x'):
    s_new = struc.copy()
    s_new.rotate(angle, axis)
    pbc = Atoms(s_new.symbols.numbers, positions=s_new.positions, cell=s_new.cell)
    return pbc #p_struc

def get_perturbed_struc(struc, eps):
    s_new = struc.copy()
    pos = s_new.positions
    pos[0,0] += eps
    pbc = Atoms(s_new.symbols.numbers, positions=pos, cell=s_new.cell)
    return pbc #p_struc

surfaces = [(1, 0, 0), (1, 1, 0), (1, 1, 1)]
layers = [2, 2, 2]
lc = 3.61000
cu = FaceCenteredCubic('Cu', surfaces, layers, latticeconstant=lc, vacuum=10)
rcut = 3.0
eps = 1e-8

#TrainData = resource_filename("pyxtal_ff", "datasets/Si/UCSD/test.json")
TrainData = resource_filename("pyxtal_ff", "datasets/Si/PyXtal/Si4.json")
parameters = {'lmax': 2}
system = ['Si']
descriptor = {'type': 'Bispectrum',
              'parameters': parameters,
              'Rc': 3.0,
              'N_train': 10,
              'stress': True,
              }

descriptor_comp = {'type': 'Bispectrum',
              'parameters': parameters,
              'Rc': 3.0,
              'N_train': 10,
              }

class TestEAMD(unittest.TestCase):
    from pyxtal_ff.descriptors.EAMD import EAMD
    symmetry = {'L': 2, 'eta': [0.36], 'Rs': [1.]}
    struc = get_rotated_struc(cu)
    rho0 = EAMD(symmetry, rcut, derivative=True).calculate(struc)
    struc = get_rotated_struc(cu, 10, 'x')
    rho1 = EAMD(symmetry, rcut, derivative=True).calculate(struc)
    struc = get_perturbed_struc(cu, eps)
    rho2 = EAMD(symmetry, rcut, derivative=False).calculate(struc)

    def test_rho_value(self):
        self.assertAlmostEqual(self.rho0['x'][0,0], 21.07766448405431)

    def test_rho_rotation_variance(self):
        array1 = self.rho0['x'].flatten()
        array2 = self.rho1['x'].flatten()
        self.assertTrue(np.allclose(array1, array2))

    def test_drhodR_rotation_variance(self):
        array1 = np.linalg.norm(self.rho0['dxdr'][0,:,:], axis=1)
        array2 = np.linalg.norm(self.rho1['dxdr'][0,:,:], axis=1)
        self.assertTrue(np.allclose(array1, array2))

    def test_drhodR_vs_numerical(self):
        array1 = (self.rho2['x'][0] - self.rho0['x'][0]).flatten()/eps
        array2 = self.rho0['dxdr'][0, :, 0].flatten()
        if not np.allclose(array1, array2):
            print('\n Numerical dGdR')
            print((self.rho2['x'][0] - self.rho0['x'][0])/eps)
            print('\n precompute')
            print(array2)
        self.assertTrue(np.allclose(array1, array2))

class TestACSF(unittest.TestCase):
    from pyxtal_ff.descriptors.ACSF import ACSF
    symmetry = {'G2': {'eta': [0.003214], 'Rs': [0]},
                'G4': {'lambda': [1], 'zeta':[1], 'eta': [0.000357]},
                'G5': {'lambda': [-1], 'zeta':[1], 'eta': [0.004]},
                }
    struc = get_rotated_struc(cu)
    g0 = ACSF(symmetry, rcut, derivative=True).calculate(struc)
    struc = get_rotated_struc(cu, 10, 'x')
    g1 = ACSF(symmetry, rcut, derivative=True).calculate(struc)
    struc = get_perturbed_struc(cu, eps)
    g2 = ACSF(symmetry, rcut, derivative=False).calculate(struc)

    def test_G2_value(self):
        self.assertAlmostEqual(self.g0['x'][0,0], 0.36925589)

    def test_G4_value(self):
        self.assertAlmostEqual(self.g0['x'][0,1], 0.00232827)

    def test_G_rotation_variance(self):
        array1 = self.g0['x'].flatten()
        array2 = self.g1['x'].flatten()
        self.assertTrue(np.allclose(array1, array2))

    def test_dGdR_rotation_variance(self):
        array1 = np.linalg.norm(self.g0['dxdr'][0,:,:], axis=1)
        array2 = np.linalg.norm(self.g1['dxdr'][0,:,:], axis=1)
        self.assertTrue(np.allclose(array1, array2))

    def test_dGdR_vs_numerical(self):
        array1 = (self.g2['x'][0] - self.g0['x'][0]).flatten()/eps
        array2 = self.g0['dxdr'][0, :, 0].flatten()
        if not np.allclose(array1, array2):
            print('\n Numerical dGdR')
            print((self.g2['x'][0] - self.g0['x'][0])/eps)
            print('\n precompute')
            print(array2)
        self.assertTrue(np.allclose(array1, array2))

class TestSO4(unittest.TestCase):
    from pyxtal_ff.descriptors.SO4 import SO4_Bispectrum
    struc = get_rotated_struc(cu)
    b0_poly = SO4_Bispectrum(lmax=1, rcut=rcut, stress=True, derivative=True).calculate(struc)#, backend='pymatgen')
    struc = get_rotated_struc(cu, 20, 'x')
    b1_poly = SO4_Bispectrum(lmax=1, rcut=rcut, stress=True, derivative=True).calculate(struc)#, backend='pymatgen')
    struc = get_perturbed_struc(cu, eps)
    b2_poly = SO4_Bispectrum(lmax=1, rcut=rcut, derivative=False).calculate(struc)#, backend='pymatgen')

    def test_B_poly_rotation_variance(self):
        array1 = self.b0_poly['x'].flatten()
        array2 = self.b1_poly['x'].flatten()
        self.assertTrue(np.allclose(array1, array2))

    def test_dBdr_poly_rotation_variance(self):
        array1 = np.linalg.norm(self.b0_poly['dxdr'][0,:,:], axis=1)
        array2 = np.linalg.norm(self.b1_poly['dxdr'][0,:,:], axis=1)
        self.assertTrue(np.allclose(array1, array2))

    def test_dBdr_poly_vs_numerical(self):
        array1 = (self.b2_poly['x'][0] - self.b0_poly['x'][0]).flatten()/eps
        array2 = self.b0_poly['dxdr'][0, :, 0].flatten()
        self.assertTrue(np.allclose(array1, array2, rtol=1e-2, atol=1e-2))

class TestSO3(unittest.TestCase):
    from pyxtal_ff.descriptors.SO3 import SO3
    struc = get_rotated_struc(cu)
    p0 = SO3(nmax=1, lmax=1, rcut=rcut, derivative=True, stress=True).calculate(struc) #, backend='ase')
    struc = get_rotated_struc(cu, 20, 'x')
    p1 = SO3(nmax=1, lmax=1, rcut=rcut, derivative=True, stress=True).calculate(struc) #, backend='ase')
    struc = get_perturbed_struc(cu, eps)
    p2 = SO3(nmax=1, lmax=1, rcut=rcut, derivative=True, stress=True).calculate(struc) #, backend='ase')

    def test_SO3_rotation_variance(self):
        array1 = self.p0['x'].flatten()
        array2 = self.p1['x'].flatten()
        self.assertTrue(np.allclose(array1, array2))

    def test_dpdr_rotation_variance(self):
        array1 = np.linalg.norm(self.p0['dxdr'][0,:,:], axis=1)
        array2 = np.linalg.norm(self.p1['dxdr'][0,:,:], axis=1)
        self.assertTrue(np.allclose(array1, array2))

    def test_dpdr_vs_numerical(self):
        array1 = (self.p2['x'][0] - self.p0['x'][0])/eps
        array2 = self.p0['dxdr'][0, :, 0]
        self.assertTrue(np.allclose(array1, array2, rtol=1e-2, atol=1e-2))

class TestRegression(unittest.TestCase):

    model = {'system' : system,
             'hiddenlayers': [12, 12],
             'epoch': 10,
             'stress_coefficient': None,
             'force_coefficient': 0.03,
             'path': 'unittest/'
            }
    ff = PyXtal_FF(descriptors=descriptor, model=model)
    struc = bulk('Si', 'diamond', a=5.0, cubic=True)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree('unittest')
        
    def test_1_NN_LBFGS(self):
        self.ff.algorithm = 'NN'
        self.ff._model['optimizer'] = {'method': 'lbfgs'}
        (train_stat, _) = self.ff.run(mode='train', TrainData=TrainData)

    def test_2_NN_ADAM(self):
        self.ff.algorithm = 'NN'
        self.ff._model['optimizer']= {'method': 'ADAM'}
        self.ff._MODEL(self.ff._model)
        train_stat = self.ff.model.train('Train_db', self.ff.optimizer)
        self.ff.model.save_checkpoint(des_info=self.ff._descriptors)


    def test_3_lr(self):
        self.ff.algorithm = 'PR'
        self.ff._model['order'] = 1
        self.ff._MODEL(self.ff._model)
        train_stat = self.ff.model.train('Train_db', None)
        self.ff.model.save_checkpoint(des_info=self.ff._descriptors)

    def test_4_qr(self):
        self.ff.algorithm = 'PR'
        self.ff._model['order'] = 2
        self.ff._MODEL(self.ff._model)
        train_stat = self.ff.model.train('Train_db', None)

    def test_5_NN_calculator(self):
        #calc = PyXtalFFCalculator(mliap='unittest/12-12-checkpoint.pth', logo=False)
        ff = PyXtal_FF(model={'system': ["Si"]}, logo=False)
        ff.run(mode='predict', mliap='unittest/12-12-checkpoint.pth')
        calc = PyXtalFFCalculator(ff=ff)

        self.struc.set_calculator(calc)
        self.struc.get_potential_energy()
        self.struc.get_stress()

    def test_6_LR_calculator(self):
        #calc = PyXtalFFCalculator(mliap='unittest/PolyReg-checkpoint.pth', logo=False)
        ff = PyXtal_FF(model={'system': ["Si"]}, logo=False)
        ff.run(mode='predict', mliap='unittest/PolyReg-checkpoint.pth')
        calc = PyXtalFFCalculator(ff=ff)


        self.struc.set_calculator(calc)
        self.struc.get_potential_energy()
        self.struc.get_stress()


class TestRegressionComp(unittest.TestCase):

    model = {'system' : system,
             'stress_coefficient': None,
             'force_coefficient': 0.03,
             'path': 'unittest_comp/',
             'algorithm': 'PR',
            }
    ff = PyXtal_FF(descriptors=descriptor_comp, model=model)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree('unittest_comp')
        
    def test_lr_comp(self):
        self.ff._model['order'] = 1
        (train_stat, _) = self.ff.run(mode='train', TrainData=TrainData)

if __name__ == '__main__':

    unittest.main()
