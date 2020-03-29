# a slight modification based on table S5 from https://arxiv.org/pdf/1906.08888.pdf
# $ wget https://raw.githubusercontent.com/materialsvirtuallab/mlearn/master/data/Si/test.json
# $ wget https://raw.githubusercontent.com/materialsvirtuallab/mlearn/master/data/Si/training.json

from pyxtal_ff import PyXtal_FF

TrainData = "training.json"
TestData  = "test.json"

descriptor = {'Rc': 4.9,
              'parameters': {'lmax': 3},
              'force': True,
              'stress': False,
             }

model = {'system' : ['Si'],
         'hiddenlayers': [16, 16],
         'activation': ['Tanh', 'Tanh', 'Linear'], 
         'force_coefficient': 0.1, 
         'epoch': 1000,
         #'restart': 'Si-Bispectrum/16-16-checkpoint.pth',
         'optimizer': {'method': 'lbfgs'},
         }

ff = PyXtal_FF(descriptors=descriptor, model=model)
ff.run(mode='train', TrainData=TrainData, TestData=TestData)
