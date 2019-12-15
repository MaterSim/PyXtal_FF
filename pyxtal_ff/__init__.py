import os
from pyxtal_ff.models import NeuralNetwork
from pyxtal_ff.models.polynomialregression import PR
from pyxtal_ff.utilities import convert_to_descriptor
from pyxtal_ff.version import __version__


class PyXtal_FF():
    def __init__(self, TrainData=None, TestData=None, descriptors=None, 
                 model=None):
        """ PyXtal_FF is used for developing Machine Learning Interatomic 
        Potential.
        
        Parameters
        ----------
        TrainData: str (*)
            TrainData indicates the location the training data set file.
            After the training of Neural Network is over, TrainData will be 
            self-evaluated for the accuracy.
        
        TestData: str
            TestData is an optional argument. If the test data set is present, 
            PyXtal_FF will evaluate the accuracy of the developed potential 
            with the TestData.
        
        descriptors: dict
            The atom-centered descriptors parameters are defined here.
            The list of the descriptors keys:
            - type: str
                The type of atom-centered descriptors.
                + BehlerParrinello (Gaussian symmetry)
                + Bispectrum
            - Rc: float
                The radial cutoff of the descriptors.
            - derivative: bool
                If True, the derivative of the descriptors will be calculated.
            - N_train: int
                The number of crystal structures in training data set 
                to be converted into descriptors.
            - N_test: int
                The number of crystal structures in test data set 
                to be converted into descriptors.
            - ncpu: int
                The number of cpu core to use for converting crystal structures 
                into descriptors.
            - parameters: dict
                Example,
                + BehlerParrinello
                  {'G2': {'eta': [1.3, 2.], 'Rs': [.1, .2]},
                   'G4': {'eta': [.3, .7], 'lambda': [-1, 1], 'zeta': [.8, 2]}}
                + Bispectrum
                  {'lmax': 3, opt: 'polynomial', 'rfac': 1.}

        model: dict
            The Neural Network or Polynomial Regression parameters are defined.
            The list of the model keys:
            - algorithm: str (*NN and *PR)
                The desired machine learning algorithm for potential 
                development. Choose between ['PolynomialRegression', 'PR'] or
                ['NeuralNetwork', 'NN'].
            - system: list of str (*; *NN and *PR)
                A list of atomic species in the crystal system.
                e.g. ['Si', 'O']
            - hiddenlayers: list of int (*NN)
                [3, 3] contains 2 hidden layers with 3 nodes each.
            - activation: str or list of str (*NN)
                The activation function for the neural network model.
                Currently, there are tanh, sigmoid, and linear.
            - random_seed: int (*NN)
                If the Neural Network is initialized from the beginning,
                the random_seed is used to established the initial 
                Neural Network weights randomly.
            - batch_size: int (*NN)
                batch_size is used for online learning. The weights of the 
                Neural Network is updated for each batch_size.
            - force_coefficient: float (*NN and *PR)
                This parameter is used in the penalty function to scale 
                the force contribution relative to the energy.
            - alpha: float (*NN)
                L2 penalty (regularization term) parameter.
            - softmax_beta: float (*NN)
                The parameters for Softmax Energy Penalty function.
            - unit: str (*NN)
                The unit of energy ('eV' or 'Ha'). 
                The default unit of energy is 'eV'. If 'Ha' is used,
                Bohr is the unit length; otherwise, Angstrom is used.
            - logging: ? (*NN and *PR)
                ???
            - restart: str (*NN)
                To continue Neural Network training from where it was left off.
            - runner: str (*NN)
                CPU or GPU mode.
                CPU mode is 'numpy', and GPU mode is 'pytorch' or 'cupy'.
            - optimizer: dict (*NN)
                Define the optimization method used to update NN parameters.
            - path: str (*NN and *PR)
                The user defined path to store the NN results.
                path has to be ended with '/'.
            - order: int (*PR)
                Order is used to determined the polynomial order. 
                For order = 1, linear is employed, and quadratic is employed 
                for order = 2.
                
        (*) required.
        (*NN) for Neural Network algorithm only.
        (*PR) for Polynomial Regression algorithm only.
        """
        self.print_logo()
        
        # Checking descriptors' keys
        descriptors_keywords = ['type', 'Rc', 'derivative', 'N_train', 
                                'N_test', 'ncpu', 'parameters']
        if descriptors is not None:
            for key in descriptors.keys():
                if key not in descriptors_keywords:
                    msg = f"Don't recognize {key} in descriptors. "+\
                          f"Here are the keywords: {descriptors_keywords}."
                    raise NotImplementedError(msg)

        # Checking Neural Network' keys
        keywords = ['algorithm', 'system', 'hiddenlayers', 'activation', 
                    'random_seed', 'batch_size', 'force_coefficient', 
                    'alpha', 'unit', 'softmax_beta', 'logging', 'restart', 
                    'runner', 'optimizer', 'path', 'order', 'N_max',
                    'atoms_per_batch']
        for key in model.keys():
            if key not in keywords:
                msg = f"Don't recognize {key} in model. "+\
                      f"Here are the keywords: {keywords}."
                raise NotImplementedError(msg)

        # Set up default descriptors parameters
        _descriptors = {'type': 'Bispectrum',
                        'Rc': 6.0,
                        'derivative': False,
                        'N': None,
                        'N_train': None,
                        'N_test': None,
                        'ncpu': 1,
                        }
        
        # Convert data set(s) into descriptors and parse features.
        if descriptors is not None:
            _descriptors.update(descriptors)

        _parameters = {'lmax': 3, 'rfac': 1.0,
                       'normalize_U': False}
        if 'parameters' in descriptors:
            _parameters.update(descriptors['parameters'])
            _descriptors['parameters'] = _parameters

        _descriptors.update({'N': _descriptors['N_train']})

        # Create new directory to dump all the results
        if 'path' in model:
            self.path = model['path']
        else:
            _system = model['system']
            self.path = "-".join(_system) + "-"
            self.path += _descriptors['type'] + "/"
        if not os.path.exists(self.path):
            os.mkdir(self.path)

        self.print_descriptors(_descriptors)
        self.TrainFeatures, self.TrainDescriptors = convert_to_descriptor(
                                                        TrainData,
                                                        self.path+'Train.npy',
                                                        _descriptors,
                                                        ncpu=_descriptors['ncpu'])
        if TestData is not None:
            _descriptors.update({'N': _descriptors['N_test']}) 
            self.TestFeatures, self.TestDescriptors = convert_to_descriptor(
                                                        TestData,
                                                        self.path+'Test.npy',
                                                        _descriptors,
                                                        ncpu=_descriptors['ncpu'])
            self.EvaluateTest = True
        else:
            self.EvaluateTest = False
        
        # Create model
        pr_keywords = ['PolynomialRegression', 'PR']
        nn_keywords = ['NeuralNetwork', 'NN']
        if 'algorithm' not in model:
            model['algorithm'] = 'NN'

        if model['algorithm'] in pr_keywords:
            self.algorithm = 'PR'
        elif model['algorithm'] in nn_keywords:
            self.algorithm = 'NN'
        else:
            msg = f"{model['algorithm']} is not implemented."
            raise NotImplementedError(msg)
            
        # Instantiate model
        self._MODEL(model, _descriptors['type'])
        
        # Delete variable after used
        del(model)
        del(_descriptors)
       
        
    def _MODEL(self, model, descriptors_type):
        """ Model is created here. """
        # Polynomial regression doesn't applied to gaussian descriptors.
        #if self.algorithm == 'PR' and descriptors_type != 'Bispectrum':
        #    msg = "Polynomial Regression does not predict bispectrum!"
        #    raise NotImplementedError(msg)
            
        if self.algorithm == 'NN':
            _model = {'system': None,
                      'hiddenlayers': [6, 6],
                      'activation': ['tanh', 'tanh', 'linear'],
                      'random_seed': None,
                      'batch_size': None,
                      'force_coefficient': 0.03,
                      'alpha': 1e-4,
                      'softmax_beta': None,
                      'unit': 'eV',
                      'logging': None,
                      'restart': None,
                      'path': self.path,
                      'runner': 'numpy',
                      'optimizer': {},
                      'atoms_per_batch': 1000,
                      }
            _model.update(model) # Update model
            
            if len(_model['activation']) != len(_model['hiddenlayers']) + 1:
                msg = '\nWarning: Incompatible activation functions and hiddenlayers.'
                print(msg)
                print('hiddenlayers: ', _model['hiddenlayers'])
                print('activations: ', _model['activation'])
                _model['activation'] = ['tanh']*len(model['hiddenlayers'])+['linear']
                print('revised activations: ', _model['activation'])
    
            if 'parameters' not in _model['optimizer']:
                _model['optimizer']['parameters'] = None
            if 'derivative' not in _model['optimizer']:
                _model['optimizer']['derivative'] = True
            if 'method' not in _model['optimizer']:
                _model['optimizer']['method'] = 'L-BFGS-B'
            # Only do minibatch for SGD and ADAM
            if _model['optimizer']['method'] not in ['SGD', 'ADAM']:
                _model['batch_size'] = None
            self.model = NeuralNetwork(elements=_model['system'],
                                       hiddenlayers=_model['hiddenlayers'],
                                       activation=_model['activation'],
                                       random_seed=_model['random_seed'],
                                       batch_size=_model['batch_size'],
                                       atoms_per_batch=_model['atoms_per_batch'],
                                       force_coefficient=_model['force_coefficient'],
                                       alpha=_model['alpha'],
                                       softmax_beta=_model['softmax_beta'],
                                       unit=_model['unit'],
                                       logging=_model['logging'],
                                       restart=_model['restart'],
                                       path=_model['path'])
            self.runner = _model['runner']
            self.optimizer = _model['optimizer']

                
        elif self.algorithm == 'PR':
            _model = {'system': None,
                      'force_coefficient': 0.0001,
                      'order': 1,
                      'logging': None,
                      'path': self.path,
                      'N_max': None,
                      }
            _model.update(model)
            self.model = PR(elements=_model['system'],
                            force_coefficient=_model['force_coefficient'],
                            order =_model['order'],
                            path = _model['path'],
                            N_max = _model['N_max'],
                            )
            
        if _model['force_coefficient'] is None:
            self.use_force = False
        else:
            self.use_force = True
    
    
    def run(self):
        """ Invoke the pyxtal_ff run. """
        # Train
        if self.algorithm == 'NN':
            self.model.train(self.TrainDescriptors, self.TrainFeatures, 
                             runner=self.runner, optimizer=self.optimizer, 
                             use_force=self.use_force)
        elif self.algorithm == 'PR':
            self.model.train(self.TrainDescriptors, self.TrainFeatures,
                             use_force=self.use_force)
        
        # Evaluate Trained Data Set
        self.model.evaluate(self.TrainDescriptors, self.TrainFeatures,
                            figname='Train.png')
        
        # # Evaluate Test Data Set
        if self.EvaluateTest:
            self.model.evaluate(self.TestDescriptors, self.TestFeatures, 
                                figname='Test.png')
            

    def print_descriptors(self, _descriptors):
        """ Print the descriptors information. """
        print('The following parameters are used in descriptor calculation')
        keys = ['type', 'Rc', 'derivative']
        for key in keys:
            print('{:12s}: {:}'.format(key, _descriptors[key]))

        if _descriptors['type'] == 'Bispectrum':
            key_params = ['lmax', 'normalize_U']
        else:
            key_params = []

        for key in key_params:
            print('{:12s}: {:}'.format(key, _descriptors['parameters'][key]))
        print('\n')
        

    def print_logo(self):
        """ Print PyXtal_FF logo and version. """

        print("""
         ______       _    _          _         _______ _______ 
        (_____ \     \ \  / /        | |       (_______|_______)
         _____) )   _ \ \/ / |_  ____| |        _____   _____   
        |  ____/ | | | )  (|  _)/ _  | |       |  ___) |  ___)  
        | |    | |_| |/ /\ \ |_( ( | | |_______| |     | |      
        |_|     \__  /_/  \_\___)_||_|_(_______)_|     |_|      
               (____/      """)
        print('\n')
        print('------------------------(version', __version__,')----------------------\n')
        print('A Python package for Machine Learning Interatomic Force Field')
        print('The source code is available at https://github.com/qzhu2017/FF-project')
        print('Developed by Zhu\'s group at University of Nevada Las Vegas\n\n')
