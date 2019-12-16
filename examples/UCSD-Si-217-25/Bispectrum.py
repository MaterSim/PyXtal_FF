# a slight modification based on table S5 from https://arxiv.org/pdf/1906.08888.pdf
from pyxtal_ff import PyXtal_FF

train_data = "training.json"
test_data = "test.json" 

NN_model = {'system': ['Si'],
            'hiddenlayers': [3,3],
            'restart': 'Si-Bispectrum/3-3-parameters.json',
            'optimizer': {'method': 'L-BFGS-B',
                          'parameters': {'options': {'maxiter': 100}},
                         },
            'runner': 'numpy', #'pytorch',
            'force_coefficient': 0.03,
            'alpha': 0,
           }
descriptors = {'derivative': True,
               'Rc': 4.9,
               'parameters': {'lmax': 4}}

#-------------------------------- Run NN calculation ------------------------------
trainer = PyXtal_FF(TrainData=train_data, TestData=test_data, 
                    descriptors=descriptors, model=NN_model)
trainer.run()
