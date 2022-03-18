# a demo script to explain the multi-batch idea when the database is large

from pyxtal_ff import PyXtal_FF
import os

# split your train data into several subfiles
# Run the following unix commands
# wget https://raw.githubusercontent.com/materialsvirtuallab/mlearn/master/data/Si/training.json
# mkdir data
# cp training.json data/training0.json
# cp training.json data/training1.json
# cp training.json data/training2.json

TrainDatas = [
              "data/training0.json", 
              "data/training1.json", 
              "data/training2.json",
             ] # can also be ase.db files

# define the descriptor
descriptor = {'type': 'SNAP',
              'weights': {'Si': 1.0},
              'Rc': 5.0,
              'parameters': {'lmax': 3},
              'base_potential': {'inner': 1.5, 'outer': 2.0}, #zbl potential
              'ncpu': 1,
             }

# define the model
model = {'system' : ['Si'],
         'hiddenlayers': [12, 12],
         'optimizer': {'method': 'lbfgs'},
         'force_coefficient': 2e-2,
         'stress_coefficient': 2e-3,
         "stress_group": ["Elastic"],
         'alpha': 1e-6,
         'epoch': 100, 
         }

# restart file for successive training
restart = '12-12-checkpoint.pth'

# train the model over all subfiles and repeat mutiple times
for it in range(10):
    for i, TrainData in enumerate(TrainDatas):

        folder = 'train_data_' + str(i) + '/'
        model['path'] = folder

        if os.path.exists(restart):
            model['restart'] = restart
        else:
            model['restart'] = None
        
        ff = PyXtal_FF(descriptors=descriptor, model=model)
        ff.run(mode='train', TrainData=TrainData)

        # copy the trained model to main directory
        restart1 = folder + '/' + restart
        os.system('cp ' + restart1 + ' ' + restart)


