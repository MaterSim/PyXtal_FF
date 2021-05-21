from pyxtal_ff import PyXtal_FF

trainData = "data/SiO2.db"

if False: #True:
    folder = 'SiO2-snap/'
    descriptor = {'Rc': 4.9, 
                  'type': 'SNAP',
                  'weights': {'Si': 14.0, 'O': 8.0},
                  'parameters': {'lmax': 3},
                  'base_potential': {'inner': 1.0, 'outer': 2.0}, 
                  'ncpu': 16,
                  'N_train': 250,
                 }
else:
    folder = 'SiO2-so3/'
    descriptor = {'Rc': 4.9, 
                  'type': 'SO3',
                  'weights': {'Si': 0.5, 'O': 0.5},
                  'parameters': {'lmax': 4, 'nmax': 3},
                  'base_potential': {'inner': 1.0, 'outer': 2.0}, 
                  'ncpu': 16,
                  'N_train': 250,
                 }

 
model = {'system' : ['Si', 'O'],
         'hiddenlayers': [30, 30],
         'force_coefficient': 0.1,
         'epoch': 100,
         'path': folder,
         #'restart': folder + '/30-30-checkpoint.pth',
         'alpha': 1e-6,
         }

#------------------------- Run NN calculation ------------------------------
ff = PyXtal_FF(descriptors=descriptor, model=model)
ff.run(mode='train', TrainData=trainData)
