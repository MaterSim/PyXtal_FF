import os
from pyxtal_ff.models.neuralnetwork import NeuralNetwork
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
            - epoch: int (*NN)
                A measure of the number of times all of the training vectors 
                are used once to update the weights.
            - device: str (*NN)
                The device used to train: 'cpu' or 'cuda'.
            - force_coefficient: float (*NN and *PR)
                This parameter is used in the penalty function to scale 
                the force contribution relative to the energy.
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
                    'random_seed', 'force_coefficient', 'unit', 'softmax_beta', 
                    'logging', 'restart', 'optimizer', 'path', 'order', 'N_max', 
                    'epoch', 'device', 'alpha', 'batch_size']
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
                    
        if self.algorithm == 'NN':
            _model = {'system': None,
                      'hiddenlayers': [6, 6],
                      'activation': ['Tanh', 'Tanh', 'Linear'],
                      'random_seed': None,
                      'epoch': 100,
                      'batch_size': None,
                      'device': 'cpu',
                      'force_coefficient': 0.03,
                      'softmax_beta': None,
                      'alpha': None,
                      'unit': 'eV',
                      'logging': None,
                      'restart': None,
                      'path': self.path,
                      'optimizer': {},
                      }
            _model.update(model)
            
            if len(_model['activation']) != len(_model['hiddenlayers']) + 1:
                msg = '\nWarning: Incompatible activation functions and hiddenlayers.'
                print(msg)
                print('hiddenlayers: ', _model['hiddenlayers'])
                print('activations: ', _model['activation'])
                _model['activation'] = ['Tanh']*len(model['hiddenlayers'])+['Linear']
                print('revised activations: ', _model['activation'])
    
            if 'parameters' not in _model['optimizer']:
                _model['optimizer']['parameters'] = {}
            if 'derivative' not in _model['optimizer']:
                _model['optimizer']['derivative'] = True
            if 'method' not in _model['optimizer']:
                _model['optimizer']['method'] = 'lbfgs'

            # If LBFGS is used, epoch is 1.
            if _model['optimizer']['method'] in ['lbfgs', 'LBFGS', 'lbfgsb']:
                if 'max_iter' in _model['optimizer']['parameters'].items():
                    if _model['epoch'] > _model['optimizer']['parameters']['max_iter']:
                        _model['optimizer']['parameters']['max_iter'] = _model['epoch']
                else:
                    _model['optimizer']['parameters']['max_iter'] = _model['epoch']
                _model['epoch'] = 1

            self.model = NeuralNetwork(elements=_model['system'],
                                       hiddenlayers=_model['hiddenlayers'],
                                       activation=_model['activation'],
                                       random_seed=_model['random_seed'],
                                       epoch=_model['epoch'],
                                       batch_size=_model['batch_size'],
                                       device=_model['device'],
                                       alpha=_model['alpha'],
                                       force_coefficient=_model['force_coefficient'],
                                       softmax_beta=_model['softmax_beta'],
                                       unit=_model['unit'],
                                       logging=_model['logging'],
                                       restart=_model['restart'],
                                       path=_model['path'])
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
                            N_max = _model['N_max'])
            
    
    def run(self):
        """ Invoke the pyxtal_ff to run. """
        # Train
        if self.algorithm == 'NN':
            self.model.train(self.TrainDescriptors, self.TrainFeatures, 
                             optimizer=self.optimizer)
        elif self.algorithm == 'PR':
            self.model.train(self.TrainDescriptors, self.TrainFeatures)
        
        # Evaluate Trained Data Set
        self.model.evaluate(self.TrainDescriptors, self.TrainFeatures,
                            figname='Train.png')
        
        # Evaluate Test Data Set
        if self.EvaluateTest:
            self.model.evaluate(self.TestDescriptors, self.TestFeatures, 
                                figname='Test.png')


    def print_descriptors(self, _descriptors):
        """ Print the descriptors information. """

        print('Descriptor parameters:')
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
        
        print("\n")
        print("""
               ______       _    _          _         _______ _______ 
              (_____ \     \ \  / /        | |       (_______|_______)
               _____) )   _ \ \/ / |_  ____| |        _____   _____   
              |  ____/ | | | )  (|  _)/ _  | |       |  ___) |  ___)  
              | |    | |_| |/ /\ \ |_( ( | | |_______| |     | |      
              |_|     \__  /_/  \_\___)_||_|_(_______)_|     |_|      
                     (____/      """)
        print("\n")
        print('          A Python package for Machine Learning Interatomic Force Field')
        print('           Developed by Zhu\'s group at University of Nevada Las Vegas')
        print('      The source code is available at https://github.com/qzhu2017/FF-project')
        print("\n")
        print('================================= version', __version__,'=================================\n\n')
