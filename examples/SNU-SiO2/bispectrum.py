from pyxtal_ff import PyXtal_FF
train_data = 'OUTCAR_comp'

descriptor = {
              'Rc': 4.9, 
              'parameters': {'lmax': 3},
              'derivative': True,
              'N_train': 250,
             }
 
model = {'system' : ['Si', 'O'],
         'hiddenlayers': [30, 30],
         'activation': ['Tanh', 'Tanh', 'Linear'],
         'force_coefficient': 0.1,
         'epoch': 1000,
         'device': 'cpu',
         'restart': 'Si-O-Bispectrum/30-30-checkpoint.pth',
         'optimizer': {'method': 'lbfgs'}}

#-------------------------------- Run NN calculation ------------------------------
trainer = PyXtal_FF(TrainData=train_data, 
                    descriptors=descriptor, model=model)
trainer.run()
