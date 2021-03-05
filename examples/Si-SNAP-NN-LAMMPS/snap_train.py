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
              'weights': {'Si': 1.0},
              'Rc': 5.0,
              'parameters': {'lmax': 3},
              'ncpu': 1,
             }

model = {'system' : ['Si'],
         'hiddenlayers': [16, 16],
         'path': 'Si-snap/',
         #'restart': 'Si-snap/16-16-checkpoint.pth',
         'optimizer': {'method': 'lbfgs'},
         'force_coefficient': 2e-2,
         'stress_coefficient': 2e-3,
         'alpha': 1e-6,
         'epoch': 1000,
         }

ff = PyXtal_FF(descriptors=descriptor, model=model)
ff.run(mode='train', TrainData=TrainData, TestData=TestData)
