from pyxtal_ff import PyXtal_FF
train_data = 'OUTCAR_comp'

descriptor = {
              'Rc': 4.9, 
              'parameters': {'lmax': 3},
              'force': True,
              'stress': False,
              'N_train': 250,
             }
 
model = {'system' : ['Si', 'O'],
         'hiddenlayers': [30, 30],
         'activation': ['Tanh', 'Tanh', 'Linear'],
         'force_coefficient': 0.1,
         'epoch': 500,
         #'restart': 'Si-O-Bispectrum/30-30-checkpoint.pth',
         'optimizer': {'method': 'lbfgs'}}

#-------------------------------- Run NN calculation ------------------------------
ff = PyXtal_FF(descriptors=descriptor, model=model)
ff.run(mode='train', TrainData=train_data)
