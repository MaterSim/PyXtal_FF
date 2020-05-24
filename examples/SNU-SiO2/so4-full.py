#This is an example of using the full-batch LBFGS method
#The original dataset has 10000 samples
#Here we use only 1000 samples

from pyxtal_ff import PyXtal_FF

train_data = 'OUTCAR_comp'

descriptor = {'Rc': 4.9, 
              'parameters': {'lmax': 3},
              'N_train': 1000,
              'ncpu': 8,
             }
 
model = {'system' : ['Si', 'O'],
         'hiddenlayers': [30, 30],
         'activation': ['Tanh', 'Tanh', 'Linear'],
         'force_coefficient': 0.1,
         'epoch': 500,
         'path': 'SiO2-full-batch/',
         #'restart': 'SiO2-full-batch/30-30-checkpoint.pth',
         'optimizer': {'method': 'lbfgs'},
         }

#-------------------------------- Run NN calculation ------------------------------
ff = PyXtal_FF(descriptors=descriptor, model=model)
ff.run(mode='train', TrainData=train_data)
