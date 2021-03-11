#This is an example of using the full-batch LBFGS method
#The original dataset has 10000 samples
#Here we use only 250 samples for a quick test

from pyxtal_ff import PyXtal_FF
import os

train_data = 'OUTCAR_comp'
if not os.path.exists(train_data):
    print('downloading the training data')
    os.system('wget https://raw.githubusercontent.com/MDIL-SNU/SIMPLE-NN/master/examples/SiO2/ab_initio_output/OUTCAR_comp')

descriptor = {'Rc': 4.9, 
              'parameters': {'lmax': 3},
              'N_train': 250,
              'ncpu': 4,
             }
 
model = {'system' : ['Si', 'O'],
         'hiddenlayers': [30, 30],
         'activation': ['Tanh', 'Tanh', 'Linear'],
         'force_coefficient': 0.1,
         'epoch': 500,
         'path': 'SiO2-quick/',
         #'restart': 'SiO2-quick/30-30-checkpoint.pth',
         'optimizer': {'method': 'lbfgs'},
         }

#------------------------- Run NN calculation ------------------------------
ff = PyXtal_FF(descriptors=descriptor, model=model)
ff.run(mode='train', TrainData=train_data)
