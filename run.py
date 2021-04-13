# a slight modification based on table S5 from https://arxiv.org/pdf/1906.08888.pdf
from pyxtal_ff import PyXtal_FF
import os
import shutil

TrainData = "training.json"
TestData  = "test.json"

url = 'https://raw.githubusercontent.com/materialsvirtuallab/mlearn/master/data/Si/'
if not os.path.exists(TrainData):
    print('Downloading the training and test data')
    os.system('wget ' + url + TrainData)
    os.system('wget ' + url + TestData)

descriptor = {'type': 'SNAP',
              'Rc': 4.9,
              'weights': {'Si': 1.0},
              'parameters': {'lmax': 3},
              'ncpu': 1,
             }

model = {'system' : ['Si'],
         'hiddenlayers': [20, 20],
         'random_seed': 12345,
         'activation': ['Tanh', 'Tanh', 'Linear'],
         'optimizer': {'method': 'lbfgs'},
         'force_coefficient': 1.,
         'stress_coefficient': 1.,
         'alpha': 1e-6,
         'epoch': 1200,
         }

for i in range(1):
    ff = PyXtal_FF(descriptors=descriptor, model=model)
    ff.run(mode='train', TrainData=TrainData, TestData=TestData)
    if i > 0:
        model['restart'] = 'Si-SNAP/20-20-checkpoint.pth'
    shutil.copyfile('Si-SNAP/20-20-checkpoint.pth', f"{i*500}.pth")
