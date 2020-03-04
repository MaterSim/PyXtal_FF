from pyxtal_ff import PyXtal_FF
train_data = 'OUTCAR_comp'

descriptor = {
              'Rc': 4.9, 
              'parameters': {'lmax': 3},
              'derivative': True,
              'N_train': 250,
             }
 
model = {'system' : ['Si', 'O'],
         'algorithm': 'PR', 
         'force_coefficient': 0.1,
         'order': 1,
         'alpha': 0,
        }

#-------------------------------- Run NN calculation ------------------------------
trainer = PyXtal_FF(TrainData=train_data, 
                    descriptors=descriptor, model=model)
trainer.run()
