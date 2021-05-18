#python -m unittest pyxtal_ff/test_all.py
import shutil
import unittest
import numpy as np
from ase import Atoms
from ase.build import bulk, sort
from ase.cluster.cubic import FaceCenteredCubic
from ase import units
from ase.optimize import BFGS

from pkg_resources import resource_filename
from pyxtal_ff import PyXtal_FF
from pyxtal_ff.calculator import PyXtalFFCalculator, optimize
from pyxtal_ff.calculator.elasticity import fit_elastic_constants
from pyxtal_ff.descriptors.SO3 import SO3
from pyxtal_ff.descriptors.EAD import EAD
from pyxtal_ff.descriptors.ACSF import ACSF
from pyxtal_ff.descriptors.SO4 import SO4_Bispectrum as SO42
from pyxtal_ff.descriptors.SNAP import SO4_Bispectrum as SO41
from pyxtal_ff.utilities.base_potential import ZBL
np.set_printoptions(formatter={'float': '{: 12.4f}'.format})

def get_rotated_struc(struc, angle=0, axis='x'):
    s_new = struc.copy()
    s_new.rotate(angle, axis)
    cell = 17.22*np.eye(3)
    p_struc = Atoms(s_new.symbols.numbers, positions=s_new.positions, cell=cell, pbc=True)
    return p_struc

def get_perturbed_struc(struc, p0, p1, eps):
    s_new = struc.copy()
    pos = s_new.positions
    pos[p0, p1] += eps
    cell = 17.22*np.eye(3)
    p_struc = Atoms(s_new.symbols.numbers, positions=pos, cell=cell, pbc=True)
    return p_struc

# NaCl Cluster
nacl = bulk('NaCl', crystalstructure='rocksalt', a=5.691694, cubic=True)
nacl = sort(nacl, tags=[0,4,1,5,2,6,3,7])
nacl.set_pbc((0,0,0))
nacl = get_rotated_struc(nacl, angle=1)

# Descriptors Parameters
eps = 1e-8
rc = 6.00
nmax, lmax = 2, 2
ead_params = {'L': lmax,
              'eta': [0.36, 0.036],
              'Rs': [1.2, 2.1]}
acsf_params = {'G2': {'eta': [0.36, 0.036],
                      'Rs': [1.2, 2.1]},
               'G4': {'eta': [0.36, 0.036],
                      'Rs': [1.2, 2.1],
                      'lambda': [-1, 1],
                      'zeta': [1.0, 1.5]},
               'G5': {'eta': [0.18, 0.018],
                      'Rs': [1.1, 2.3],
                      'lambda': [-1, 1],
                      'zeta': [1.0, 1.7]}}

# Neural Network Parameters
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

# model file
bp_model = resource_filename("pyxtal_ff", "datasets/Si/PyXtal/bp-16-16-checkpoint.pth")

class TestEAD(unittest.TestCase):
    struc = get_rotated_struc(nacl)
    rho0 = EAD(parameters=ead_params, Rc=rc, derivative=True).calculate(struc)
    struc = get_rotated_struc(nacl, 10, 'x')
    rho1 = EAD(parameters=ead_params, Rc=rc, derivative=True).calculate(struc)

    def test_rho_rotation_variance(self):
        array1 = self.rho0['x']
        array2 = self.rho1['x']
        self.assertTrue(np.allclose(array1, array2))

    def test_drhodR_rotation_variance(self):
        for i in range(len(self.rho0['x'])):
            array1 = np.linalg.norm(self.rho0['dxdr'][i,:,:], axis=1)
            array2 = np.linalg.norm(self.rho1['dxdr'][i,:,:], axis=1)
            self.assertTrue(np.allclose(array1, array2))

    def test_drhodR_vs_numerical(self):
        shp = self.rho0['x'].shape
        array1 = np.zeros([shp[0], shp[0], shp[1], 3])
        for _m in range(shp[0]):
            ids = np.where(self.rho0['seq'][:,1]==_m)[0]
            array1[self.rho0['seq'][ids, 0], _m, :, :] += self.rho0['dxdr'][ids,:,:]
        for j in range(shp[0]):
            for k in range(3):
                struc = get_perturbed_struc(nacl, j, k, eps)
                rho2 = EAD(parameters=ead_params, Rc=rc, derivative=False).calculate(struc)
                array2 = (rho2['x'] - self.rho0['x'])/eps
                self.assertTrue(np.allclose(array1[:,j,:,k], array2, atol=1e-4))

class TestACSF(unittest.TestCase):
    struc = get_rotated_struc(nacl)
    ACSF0 = ACSF(symmetry_parameters=acsf_params, Rc=rc, derivative=True).calculate(struc)
    struc = get_rotated_struc(nacl, 10, 'x')
    ACSF1 = ACSF(symmetry_parameters=acsf_params, Rc=rc, derivative=True).calculate(struc)

    def test_ACSF_rotation_variance(self):
        array1 = self.ACSF0['x']
        array2 = self.ACSF1['x']
        self.assertTrue(np.allclose(array1, array2))

    def test_dACSFdR_rotation_variance(self):
        for i in range(len(self.ACSF0['x'])):
            array1 = np.linalg.norm(self.ACSF0['dxdr'][i,:,:], axis=1)
            array2 = np.linalg.norm(self.ACSF1['dxdr'][i,:,:], axis=1)
            self.assertTrue(np.allclose(array1, array2))

    def test_dACSFdR_vs_numerical(self):
        shp = self.ACSF0['x'].shape
        array1 = np.zeros([shp[0], shp[0], shp[1], 3])
        for _m in range(shp[0]):
            ids = np.where(self.ACSF0['seq'][:,1]==_m)[0]
            array1[self.ACSF0['seq'][ids, 0], _m, :, :] += self.ACSF0['dxdr'][ids,:,:]
        
        for j in range(shp[0]):
            for k in range(3):
                struc = get_perturbed_struc(nacl, j, k, eps)
                ACSF2 = ACSF(symmetry_parameters=acsf_params, Rc=rc, derivative=False).calculate(struc)
                array2 = (ACSF2['x'] - self.ACSF0['x'])/eps
                self.assertTrue(np.allclose(array1[:,j,:,k], array2, atol=1e-6))

class TestwACSF(unittest.TestCase):
    struc = get_rotated_struc(nacl)
    wACSF0 = ACSF(symmetry_parameters=acsf_params, Rc=rc, derivative=True, atom_weighted=True).calculate(struc)
    struc = get_rotated_struc(nacl, 10, 'x')
    wACSF1 = ACSF(symmetry_parameters=acsf_params, Rc=rc, derivative=True, atom_weighted=True).calculate(struc)

    def test_wACSF_rotation_variance(self):
        array1 = self.wACSF0['x']
        array2 = self.wACSF1['x']
        self.assertTrue(np.allclose(array1, array2))

    def test_dwACSFdR_rotation_variance(self):
        for i in range(len(self.wACSF0['x'])):
            array1 = np.linalg.norm(self.wACSF0['dxdr'][i,:,:], axis=1)
            array2 = np.linalg.norm(self.wACSF1['dxdr'][i,:,:], axis=1)
            self.assertTrue(np.allclose(array1, array2))

    def test_dwACSFdR_vs_numerical(self):
        shp = self.wACSF0['x'].shape
        array1 = np.zeros([shp[0], shp[0], shp[1], 3])
        for _m in range(shp[0]):
            ids = np.where(self.wACSF0['seq'][:,1]==_m)[0]
            array1[self.wACSF0['seq'][ids, 0], _m, :, :] += self.wACSF0['dxdr'][ids,:,:]
        
        for j in range(shp[0]):
            for k in range(3):
                struc = get_perturbed_struc(nacl, j, k, eps)
                wACSF2 = ACSF(symmetry_parameters=acsf_params, Rc=rc, derivative=False, atom_weighted=True).calculate(struc)
                array2 = (wACSF2['x'] - self.wACSF0['x'])/eps
                self.assertTrue(np.allclose(array1[:,j,:,k], array2, atol=1e-6))

class TestSO3(unittest.TestCase):
    struc = get_rotated_struc(nacl)
    p0 = SO3(nmax=nmax, lmax=lmax, rcut=rc, derivative=True).calculate(struc)
    struc = get_rotated_struc(nacl, 10, 'x')
    p1 = SO3(nmax=nmax, lmax=lmax, rcut=rc, derivative=True).calculate(struc)

    def test_SO3_rotation_variance(self):
        array1 = self.p0['x']
        array2 = self.p1['x']
        self.assertTrue(np.allclose(array1, array2))

    def test_dPdR_rotation_variance(self):
        for i in range(len(self.p0['x'])):
            array1 = np.linalg.norm(self.p0['dxdr'][i,:,:], axis=1)
            array2 = np.linalg.norm(self.p1['dxdr'][i,:,:], axis=1)
            self.assertTrue(np.allclose(array1, array2))

    def test_dPdR_vs_numerical(self):
        shp = self.p0['x'].shape
        array1 = np.zeros([shp[0], shp[0], shp[1], 3])
        for _m in range(shp[0]):
            ids = np.where(self.p0['seq'][:,1]==_m)[0]
            array1[self.p0['seq'][ids, 0], _m, :, :] += self.p0['dxdr'][ids,:,:]
        
        for j in range(shp[0]):
            for k in range(3):
                struc = get_perturbed_struc(nacl, j, k, eps)
                p2 = SO3(nmax=nmax, lmax=lmax, rcut=rc, derivative=False).calculate(struc)
                array2 = (p2['x'] - self.p0['x'])/eps
                self.assertTrue(np.allclose(array1[:,j,:,k], array2, atol=1e-4))

class TestSNAP(unittest.TestCase):
    struc = get_rotated_struc(nacl)
    SNAP0 = SO41(weights={'Na':0.3,'Cl':0.7}, lmax=lmax, rcut=rc, rfac0=0.99363).calculate(struc)
    struc = get_rotated_struc(nacl, 10, 'x')
    SNAP1 = SO41(weights={'Na':0.3,'Cl':0.7}, lmax=lmax, rcut=rc, rfac0=0.99363).calculate(struc)

    def test_SNAP_rotation_variance(self):
        array1 = self.SNAP0['x']
        array2 = self.SNAP1['x']
        self.assertTrue(np.allclose(array1, array2))

    def test_dSNAPdR_rotation_variance(self):
        for i in range(len(self.SNAP0['x'])):
            array1 = np.linalg.norm(self.SNAP0['dxdr'][i,:,:], axis=1)
            array2 = np.linalg.norm(self.SNAP1['dxdr'][i,:,:], axis=1)
            self.assertTrue(np.allclose(array1, array2))

    def test_dSNAPdR_vs_numerical(self):
        shp = self.SNAP0['x'].shape
        array1 = np.zeros([shp[0], shp[0], shp[1], 3])
        for _m in range(shp[0]):
            ids = np.where(self.SNAP0['seq'][:,1]==_m)[0]
            array1[self.SNAP0['seq'][ids, 0], _m, :, :] += self.SNAP0['dxdr'][ids,:,:]
        
        for j in range(shp[0]):
            for k in range(3):
                struc = get_perturbed_struc(nacl, j, k, eps)
                SNAP2 = SO41(weights={'Na':0.3,'Cl':0.7}, lmax=lmax, rcut=rc, derivative=True, rfac0=0.99363).calculate(struc)
                array2 = (SNAP2['x'] - self.SNAP0['x'])/eps
                self.assertTrue(np.allclose(array1[:,j,:,k], array2, atol=1e-6))

class TestSO4(unittest.TestCase):
    struc = get_rotated_struc(nacl)
    so40 = SO42(lmax=lmax, rcut=rc, derivative=True).calculate(struc)
    struc = get_rotated_struc(nacl, 10, 'x')
    so41 = SO42(lmax=lmax, rcut=rc, derivative=True).calculate(struc)

    def test_so4_rotation_variance(self):
        array1 = self.so40['x']
        array2 = self.so41['x']
        self.assertTrue(np.allclose(array1, array2))

    def test_dso4dR_rotation_variance(self):
        for i in range(len(self.so40['x'])):
            array1 = np.linalg.norm(self.so40['dxdr'][i,:,:], axis=1)
            array2 = np.linalg.norm(self.so41['dxdr'][i,:,:], axis=1)
            self.assertTrue(np.allclose(array1, array2))

    def test_dso4dR_vs_numerical(self):
        shp = self.so40['x'].shape
        array1 = np.zeros([shp[0], shp[0], shp[1], 3])
        for _m in range(shp[0]):
            ids = np.where(self.so40['seq'][:,1]==_m)[0]
            array1[self.so40['seq'][ids, 0], _m, :, :] += self.so40['dxdr'][ids,:,:]
        
        for j in range(shp[0]):
            for k in range(3):
                struc = get_perturbed_struc(nacl, j, k, eps)
                so42 = SO42(lmax=lmax, rcut=rc, derivative=False).calculate(struc)
                array2 = (so42['x'] - self.so40['x'])/eps
                self.assertTrue(np.allclose(array1[:,j,:,k], array2, atol=1e-2))

class TestCalculator(unittest.TestCase):

    def testOptim(self):
        ff = PyXtal_FF(model={'system': ["Si"]}, logo=False)
        ff.run(mode='predict', mliap=bp_model)
        calc = PyXtalFFCalculator(ff=ff)
        si = bulk('Si', 'diamond', a=5.0, cubic=True)
        si.set_calculator(calc)
        si = optimize(si, box=True)
        self.assertTrue(abs(si.get_cell()[0][0]-5.469) <1e-2)

    def test_elastic(self):
        ff = PyXtal_FF(model={'system': ["Si"]}, logo=False)
        ff.run(mode='predict', mliap=bp_model)
        calc = PyXtalFFCalculator(ff=ff)
        si = bulk('Si', 'diamond', a=5.469, cubic=True)
        si.set_calculator(calc)
        C, C_err = fit_elastic_constants(si, symmetry='cubic', optimizer=BFGS)
        C /= units.GPa
        self.assertTrue(abs(C[0,0]-124.5)<1.0)

class TestZBL(unittest.TestCase):
    struc = nacl.copy()
    d1 = ZBL(2.0, 7.0).calculate(struc)
    d2 = ZBL(4.0, 4.5).calculate(struc)

    def test_ZBL(self):
        tenergy1 = 4.4444502 / len(self.struc)
        tforces1 = np.array(
                    [[-0.787367, -0.773506, -0.800989],
                     [-0.787367,  0.773506,  0.800989],
                     [ 0.787367, -0.800989,  0.773506],
                     [ 0.787367,  0.800989, -0.773506],
                     [ 0.814652, -0.80031,  -0.828745],
                     [ 0.814652,  0.80031,   0.828745],
                     [-0.814652, -0.828745,  0.80031 ],
                     [-0.814652,  0.828745, -0.80031]])
        tstress1 = np.array([2861.0152, 2861.0152, 2861.0152, 0, 0, 0])
        
        tenergy2 = 3.9436037 / len(self.struc)
        tforces2 = np.array(
                    [[-0.781195, -0.767443, -0.794710],
                     [-0.781195,  0.767443,  0.794710],
                     [ 0.781195, -0.794710,  0.767443],
                     [ 0.781195,  0.794710, -0.767443],
                     [ 0.808353, -0.794122, -0.822338],
                     [ 0.808353,  0.794122,  0.822338],
                     [-0.808353, -0.822338,  0.794122],
                     [-0.808353,  0.822338, -0.794122]])
        tstress2 = np.array([2838.7443, 2838.7443, 2838.7443, 0, 0, 0])

        energy1 = self.d1['energy'] / len(self.struc)
        forces1 = self.d1['force']
        stress1 = self.d1['stress'] * 1602176.6208
        energy2 = self.d2['energy'] / len(self.struc)
        forces2 = self.d2['force']
        stress2 = self.d2['stress'] * 1602176.6208

        self.assertTrue(abs(tenergy1-energy1) < 1e-7)
        self.assertTrue(np.allclose(tforces1, forces1))
        self.assertTrue(np.allclose(tstress1, stress1))

        self.assertTrue(abs(tenergy2-energy2) < 1e-7)
        self.assertTrue(np.allclose(tforces2, forces2))
        self.assertTrue(np.allclose(tstress2, stress2))

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
