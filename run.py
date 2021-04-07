# a slight modification based on table S5 from https://arxiv.org/pdf/1906.08888.pdf
from pyxtal_ff import PyXtal_FF
import os

TrainData = "training.json"
TestData  = "test.json"

url = 'https://raw.githubusercontent.com/materialsvirtuallab/mlearn/master/data/Si/'
if not os.path.exists(TrainData):
    print('Downloading the training and test data')
    os.system('wget ' + url + TrainData)
    os.system('wget ' + url + TestData)

descriptor = {'type': 'SNAP',
              'Rc': 5.0,
              'weights': {'Si': 1.0},
              'parameters': {'lmax': 3},
              'ncpu': 1,
             }

model = {'system' : ['Si'],
         'hiddenlayers': [20, 20],
         'activation': ['Tanh', 'Tanh', 'Linear'],
         #'path': 'Si-so4/',
         'optimizer': {'method': 'lbfgs'},
         'random_seed': 15,
         'force_coefficient': None,
         #'stress_coefficient': 2e-3,
         'alpha': 1e-6,
         'epoch': 1000,
         }

ff = PyXtal_FF(descriptors=descriptor, model=model)
ff.run(mode='train', TrainData=TrainData, TestData=TestData)
