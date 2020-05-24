#This is an example of using the mini-batch Adam method
#The original dataset has 10000 samples
#Here we use 200-500 samples for each batch

from pyxtal_ff import PyXtal_FF

train_data = 'OUTCAR_comp'

descriptor = {'Rc': 4.9, 
              'parameters': {'lmax': 3},
              'N_train': 100,
              'ncpu': 8,
             }
 
model = {'system' : ['Si', 'O'],
         'hiddenlayers': [30, 30],
         'activation': ['Tanh', 'Tanh', 'Linear'],
         'force_coefficient': 0.1,
         'epoch': 500,
         'batch_size': 200,
         'path': 'SiO2-mini-batch/'
         #'restart': 'SiO2-mini-batch/30-30-checkpoint.pth',
         'memory': 'out',
         'optimizer': {'method': 'Adam'}
         }

#-------------------------------- Run NN calculation ------------------------------
ff = PyXtal_FF(descriptors=descriptor, model=model)
ff.run(mode='train', TrainData=train_data)
