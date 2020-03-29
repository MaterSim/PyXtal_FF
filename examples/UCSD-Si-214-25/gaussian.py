# a slight modification based on table S5 from https://arxiv.org/pdf/1906.08888.pdf
# $ wget https://raw.githubusercontent.com/materialsvirtuallab/mlearn/master/data/Si/test.json
# $ wget https://raw.githubusercontent.com/materialsvirtuallab/mlearn/master/data/Si/training.json

from pyxtal_ff import PyXtal_FF

TrainData = "training.json"
TestData  = "test.json"

symmetry = {'G2': {'eta': [0.035709, 0.071418, 0.178545,
                           0.35709, 0.71418, 1.78545],
                   'Rs': [0]},
            'G4': {'lambda': [-1, 1],
                   'zeta': [1],
                   'eta': [0.035709, 0.071418, 0.178545, 0.35709]}
           }

descriptor = {'type': 'BehlerParrinello',
            'parameters': symmetry,
            'Rc': 5.2,
            'force': True,
            'stress': True,
           }

model = {'system': ['Si'],
         'hiddenlayers': [16, 16],
         'restart': 'Si-BehlerParrinello/16-16-checkpoint.pth',
         'optimizer': {'method': 'lbfgs'},
         'force_coefficient': 3e-2,
         'stress_coefficient': 1e-5,
         'alpha': 0,
         'epoch': 500,
        }

ff = PyXtal_FF(descriptors=descriptor, model=model)
ff.run(mode='train', TrainData=TrainData, TestData=TestData)
