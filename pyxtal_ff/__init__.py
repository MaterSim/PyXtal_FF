import os
from pyxtal_ff.models.neuralnetwork import NeuralNetwork
from pyxtal_ff.models.polynomialregression import PR
from pyxtal_ff.models.gaussianprocess import GaussianProcess
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
                + BehlerParrinello (Gaussian symmetry function)
                + Bispectrum
                + SOAP
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
            - random_sample: bool
                To shuffle the samples randomly.
            - ncpu: int
                The number of cpu core to use for converting crystal structures 
                into descriptors.
            - parameters: dict
                Examples:
                + BehlerParrinello
                  {'G2': {'eta': [1.3, 2.], 'Rs': [.1, .2]},
                   'G4': {'eta': [.3, .7], 'lambda': [-1, 1], 'zeta': [.8, 2]}}
                + Bispectrum
                  {'lmax': 3, opt: 'polynomial', 'rfac': 1.}
                + SOAP
                  {'nmax': 1, 'lmax': 3}

        model: dict
            The Neural Network, Polynomial Regression, or Gaussian Process Regression 
            parameters are defined.

            The list of the model keys:
            - algorithm: str (*NN and *PR)
                The desired machine learning algorithm for potential 
                development. Choose between ['PolynomialRegression', 'PR'] or
                ['NeuralNetwork', 'NN'].
            - system: list of str (*; *NN, *PR and *GPR)
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
            - epoch: int (*NN, *GPR)
                A measure of the number of times all of the training vectors 
                are used once to update the weights.
            - device: str (*NN, *GPR)
                The device used to train: 'cpu' or 'cuda'.
            - force_coefficient: float (*NN, *PR, *GPR)
                This parameter is used in the penalty function to scale 
                the force contribution relative to the energy.
            - softmax_beta: float (*NN)
                The parameters for Softmax Energy Penalty function.
            - unit: str (*NN)
                The unit of energy ('eV' or 'Ha'). 
                The default unit of energy is 'eV'. If 'Ha' is used,
                Bohr is the unit length; otherwise, Angstrom is used.
            - logging: ? (*NN, *PR and *GPR)
                ???
            - restart: str (*NN)
                To continue Neural Network training from where it was left off.
            - optimizer: dict (*NN and *GPR)
                Define the optimization method used to update NN parameters.
            - path: str (*NN, *PR, *GPR)
                The user defined path to a directory for storing the ML results.
                Note: path has to be ended with '/'.
            - order: int (*PR)
                Order is used to determined the polynomial order.
                For order = 1, linear is employed, and quadratic is employed 
                for order = 2.
            - d_max: int (*PR)
                The maximum number of descriptors used.
            - alpha: float (*NN and *PR)
                L2 penalty (regularization term) parameter.
            - norm: int (*PR)
                This argument defines a model to calculate the regularization
                term. It takes only 1 or 2 as its value: Manhattan or Euclidean 
                norm, respectively. If alpha is None, norm is ignored.
            - noise: float (*GPR)
                The noise added to the Gaussian Kernel
            - kernel: str (*GPR)
                The kernel specifying the covariance function of the GPR.
                The current development allows "RBF" and "DotProduct".

        (*) required.
        (*NN) for Neural Network algorithm only.
        (*PR) for Polynomial Regression algorithm only.
        (*GPR) for Gaussian Process Regressor algorithm only.
        """
        self.print_logo()
        
        # Checking descriptors' keys
        descriptors_keywords = ['type', 'Rc', 'derivative', 'N_train', 
                                'N_test', 'random_sample', 'ncpu', 'parameters']
        if descriptors is not None:
            for key in descriptors.keys():
                if key not in descriptors_keywords:
                    msg = f"Don't recognize {key} in descriptors. "+\
                          f"Here are the keywords: {descriptors_keywords}."
                    raise NotImplementedError(msg)

        # Checking Neural Network' keys
        keywords = ['algorithm', 'system', 'hiddenlayers', 'activation', 
                    'random_seed', 'force_coefficient', 'unit', 'softmax_beta', 
                    'logging', 'restart', 'optimizer', 'path', 'order', 'd_max', 
                    'epoch', 'device', 'alpha', 'batch_size', 'noise', 'kernel',
                    'norm']
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
                        'random_sample': False,
                        'ncpu': 1,
                        }
        
        # Convert data set(s) into descriptors and parse features.
        if descriptors is not None:
            _descriptors.update(descriptors)

        _parameters = {'lmax': 3, 'rfac': 1.0, 'normalize_U': False}
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
        self._descriptors = _descriptors
        self.print_descriptors()
        if TrainData is not None:
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
        gr_keywords = ['GaussianProcessRegressor', 'GPR']
        if 'algorithm' not in model:
            model['algorithm'] = 'NN'

        if model['algorithm'] in pr_keywords:
            self.algorithm = 'PR'
        elif model['algorithm'] in nn_keywords:
            self.algorithm = 'NN'
        elif model['algorithm'] in gr_keywords:
            self.algorithm = 'GPR'
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
                      'norm': 2,
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
                      'alpha': None,
                      'norm': 2,
                      'd_max': None,
                      }
            _model.update(model)
            self.model = PR(elements=_model['system'],
                            force_coefficient=_model['force_coefficient'],
                            order=_model['order'],
                            path=_model['path'],
                            alpha=_model['alpha'],
                            norm=_model['norm'],
                            d_max=_model['d_max'])

        elif self.algorithm == 'GPR':
            _model = {'force_coefficient': 0.0001,
                      'path': self.path,
                      'system': None,
                      'epoch': 100,
                      'noise': 1e-10,
                      'kernel': 'RBF'
                      }
            _model.update(model)
                        
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
            
            self.model = GaussianProcess(elements=_model['system'],
                                         force_coefficient=_model['force_coefficient'],
                                         noise=_model['noise'],
                                         epoch=_model['epoch'],
                                         kernel=_model['kernel'],
                                         path=_model['path'],)
            self.optimizer = _model['optimizer']
            
    
    def run(self):
        """ Invoke the pyxtal_ff to run. """
        # Train
        if self.algorithm in ['NN', 'GPR']:
            self.model.train(self.TrainDescriptors, self.TrainFeatures, 
                             optimizer=self.optimizer)
        elif self.algorithm == 'PR':
            self.model.train(self.TrainDescriptors, self.TrainFeatures)
        
        # Evaluate Trained Data Set
            Train_stat = self.model.evaluate(\
            self.TrainDescriptors, self.TrainFeatures,figname='Train.png')
        
        # Evaluate Test Data Set
        if self.EvaluateTest:
            Test_stat = self.model.evaluate(\
            self.TestDescriptors, self.TestFeatures, figname='Test.png')
        else:
            Test_stat = None
        
        return (Train_stat, Test_stat)

    def print_descriptors(self):
        """ Print the descriptors information. """
        _descriptors = self._descriptors
        print('Descriptor parameters:')
        keys = ['type', 'Rc', 'derivative']
        for key in keys:
            print('{:12s}: {:}'.format(key, _descriptors[key]))

        if _descriptors['type'] == 'Bispectrum':
            key_params = ['lmax', 'normalize_U']
        elif _descriptors['type'] == 'SOAP':
            key_params = ['nmax', 'lmax']
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
