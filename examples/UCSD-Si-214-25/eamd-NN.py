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

parameters = {'L': 2, 'eta': [0.36],
              'Rs': [0.  , 0.75, 1.5 , 2.25, 3.  , 3.75, 4.5]}

descriptor = {'type': 'EAMD',
              'parameters': parameters,
              'Rc': 5.0,
              'ncpu': 4,
              }

model = {'system': ['Si'],
         'hiddenlayers': [16, 16],
         'path': 'Si-eamd/',
         'restart': 'Si-eamd/16-16-checkpoint.pth',
         'optimizer': {'method': 'lbfgs'},
         'force_coefficient': 2e-2,
         'stress_coefficient': 2e-3,
         'alpha': 1e-6,
         'epoch': 1000,
        }

ff = PyXtal_FF(descriptors=descriptor, model=model)
ff.run(mode='train', TrainData=TrainData, TestData=TestData)
