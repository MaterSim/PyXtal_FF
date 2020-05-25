# a slight modification based on table S5 from https://arxiv.org/pdf/1906.08888.pdf
from pyxtal_ff import PyXtal_FF
import os

TrainData = "training.json"
TestData  = "test.json"

url = 'https://raw.githubusercontent.com/materialsvirtuallab/mlearn/master/data/Si/'
if not os.path.exists(TrainData):
    print('Downloading the training and test data')
    os.system('wget ' + url + ' ' + TrainData)
    os.system('wget ' + url + ' ' + TestData)

descriptor = {'type': 'SOAP',
              'Rc': 4.9,
              'parameters': {'lmax': 4, 'nmax': 3},
              'ncpu': 4,
             }

model = {'system' : ['Si'],
         'hiddenlayers': [16, 16],
         'path': 'Si-so3/',
         #'restart': 'Si-so3/16-16-checkpoint.pth',
         'optimizer': {'method': 'lbfgs'},
         'force_coefficient': 3e-2,
         'stress_coefficient': 1e-5,
         'alpha': 1e-6,
         'epoch': 1000,
         }

ff = PyXtal_FF(descriptors=descriptor, model=model)
ff.run(mode='train', TrainData=TrainData, TestData=TestData)
