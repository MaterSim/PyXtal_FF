#This is an example of using the mini-batch LBFGS method
#The original dataset has 10000 samples
#Here we use 1000 samples for each batch

from pyxtal_ff import PyXtal_FF
import os

train_data = 'OUTCAR_comp'
if not os.path.exists(train_data):
    print('downloading the training data')
    os.system('wget https://raw.githubusercontent.com/MDIL-SNU/SIMPLE-NN/master/examples/SiO2/ab_initio_output/OUTCAR_comp')

descriptor = {'Rc': 4.9, 
              'type': 'SO3',
              'parameters': {'nmax': 4, 'lmax': 3},
              'ncpu': 16,
             }
 
model = {'system' : ['Si', 'O'],
         'hiddenlayers': [30, 30],
         'force_coefficient': 0.1,
         'epoch': 100,
         'batch_size': 1000,
         'path': 'SO3-mini-batch/',
         'restart': 'SO3-mini-batch/30-30-checkpoint.pth',
         'memory': 'out',
         'optimizer': {'method': 'LBFGS'}
         }

#------------------------- Run NN calculation ------------------------------
ff = PyXtal_FF(descriptors=descriptor, model=model)
ff.run(mode='train', TrainData=train_data)
ff.run(mode='train', TrainData=train_data)
ff.run(mode='train', TrainData=train_data)
ff.run(mode='train', TrainData=train_data)
ff.run(mode='train', TrainData=train_data)
