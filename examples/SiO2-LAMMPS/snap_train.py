from pyxtal_ff import PyXtal_FF

trainData = "data/SiO2.db"

descriptor = {'Rc': 4.9, 
              'type': 'SNAP',
              'weights': {'Si': 14.0, 'O': 8.0},
              'parameters': {'lmax': 3},
              'ncpu': 16,
              'N_train': 250,
             }
 
model = {'system' : ['Si', 'O'],
         'hiddenlayers': [30, 30],
         'force_coefficient': 0.1,
         'epoch': 100,
         #'batch_size': 1000,
         'path': 'SiO2-snap/',
         'restart': 'SiO2-snap/30-30-checkpoint.pth',
         'alpha': 1e-6,
         }

#------------------------- Run NN calculation ------------------------------
ff = PyXtal_FF(descriptors=descriptor, model=model)
ff.run(mode='train', TrainData=trainData)
