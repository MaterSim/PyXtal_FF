import os
import torch
from pyxtal_ff.version import __version__
from pyxtal_ff.utilities import convert_to_descriptor
from pyxtal_ff.models.polynomialregression import PR
from pyxtal_ff.models.neuralnetwork import NeuralNetwork
from pyxtal_ff.models.gaussianprocess import GaussianProcess


class PyXtal_FF():
    def __init__(self, descriptors=None, model=None, logo=True):
        """ PyXtal_FF develops Machine Learning Interatomic Potential.
        
        Parameters
        ----------
        descriptors: dict
            The atom-centered descriptors parameters are defined here.
            
            The list of the descriptors keys:
            - type: str
                The type of atom-centered descriptors.
                + BehlerParrinello (Gaussian symmetry)
                + Bispectrum
                + SOAP
            - Rc: float
                The radial cutoff of the descriptors.
            - force: bool
                If True, the derivative of the descriptors will be calculated.
            - stress: bool
                If True, the stress descriptors will be calculated.
            - N_train: int
                The number of crystal structures in training data set 
                to be converted into descriptors.
            - N_test: int
                The number of crystal structures in test data set 
                to be converted into descriptors.
            - random_sample: bool
                If True, the data will be selected after they are shuffled.
            - ncpu: int
                The number of cpu core to use for converting crystal structures 
                into descriptors.
            - compress: bool
                DXDR will be saved in the compressed version.
            - parameters: dict
                Example,
                + BehlerParrinello
                  {'G2': {'eta': [1.3, 2.], 'Rs': [.1, .2]},
                   'G4': {'eta': [.3, .7], 'lambda': [-1, 1], 'zeta': [.8, 2]}}
                + Bispectrum
                  {'lmax': 3, opt: 'polynomial', 'rfac': 1.}
                + SOAP
                  {'nmax': 1, 'lmax': 3}

        model: dict
            Machine learning parameters are defined here.

            The list of the model keys:
            - algorithm: str (*NN and *PR)
                The desired machine learning algorithm for potential 
                development. Choose between ['PolynomialRegression', 'PR'] or
                ['NeuralNetwork', 'NN'].
            - system: list of str (*NN, *PR, and *GPR)
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
            - epoch: int (*NN and *GPR)
                A measure of the number of times all of the training vectors 
                are used once to update the weights.
            - device: str (*NN and *GPR)
                The device used to train: 'cpu' or 'cuda'.
            - force_coefficient: float (*NN, *PR, and *GPR)
                This parameter is used as the penalty parameter to scale 
                the force contribution relative to the energy.
            - stress_coefficient: float (*NN, *PR, and *GPR)
                This parameter is used as the balance parameter scaling
                the stress contribution relative to the energy.
            - stress_group: list of strings, not bool! (*NN, *PR, and *GPR)
                Only the intended group will be considered in stress training.
            - softmax_beta: float (*NN)
                The parameters for Softmax Energy Penalty function.
            - unit: str (*NN)
                The unit of energy ('eV' or 'Ha'). 
                The default unit of energy is 'eV'. If 'Ha' is used,
                Bohr is the unit length; otherwise, Angstrom is used.
            - logging: ? (*NN, *PR, and *GPR)
                ???
            - restart: str (*NN)
                To continue Neural Network training from where it was left off.
            - optimizer: dict (*NN and *GPR)
                Define the optimization method used to update NN parameters.
            - path: str (*NN, *PR, *GPR)
                The user defined path to store the NN results.
                path has to be ended with '/'.
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
        if logo:
            self.print_logo()
        
        # Checking the keys in descriptors
        descriptors_keywords = ['type', 'Rc', 'force', 'stress', 'N_train', 
                                'N_test', 'random_sample', 'ncpu', 'parameters',
                                'compress']
        if descriptors is not None:
            for key in descriptors.keys():
                if key not in descriptors_keywords:
                    msg = f"Don't recognize {key} in descriptors. "+\
                          f"Here are the keywords: {descriptors_keywords}."
                    raise NotImplementedError(msg)

        # Set up default descriptors parameters
        self._descriptors = {'system': model['system'],
                             'type': 'Bispectrum',
                             'Rc': 5.0,
                             'force': True,
                             'stress': True,
                             'N': None,
                             'N_train': None,
                             'N_test': None,
                             'random_sample': False,
                             'ncpu': 1,
                             'compress': False,
                             }
        
        # Update the default based on user-defined descriptors
        if descriptors is not None:
            self._descriptors.update(descriptors)
        
            _parameters = {'lmax': 3, 'rfac': 1.0, 'normalize_U': False}
            if 'parameters' in descriptors:
                _parameters.update(descriptors['parameters'])
                self._descriptors['parameters'] = _parameters

        # Create new directory to dump all the results.
        # E.g. for default 'Si-O-Bispectrum/'
        if 'path' in model:
            self.path = model['path']
        else:
            _system = model['system']
            self.path = "-".join(_system) + "-"
            self.path += self._descriptors['type'] + "/"
        if not os.path.exists(self.path):
            os.mkdir(self.path)
        if logo:
            self.print_descriptors(self._descriptors)
        
        # Checking the keys in model.
        keywords = ['algorithm', 'system', 'hiddenlayers', 'activation', 
                    'random_seed', 'force_coefficient', 'unit', 'softmax_beta', 
                    'logging', 'restart', 'optimizer', 'path', 'order', 'd_max', 
                    'epoch', 'device', 'alpha', 'batch_size', 'noise', 'kernel',
                    'norm', 'stress_coefficient', 'stress_group']
        for key in model.keys():
            if key not in keywords:
                msg = f"Don't recognize {key} in model. "+\
                      f"Here are the keywords: {keywords}."
                raise NotImplementedError(msg)
        
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
        
        self._model = model

        # Assertion for x and dxdr compression.	
        if self._descriptors['compress']:
            assert self.algorithm == 'PR',\
            f"The compress must be 'False' to employ {self.algorithm} algorithm."	

            if 'order' in model.keys():	
                assert model['order'] == 1,\
                f"The compress must be 'False' to employ quadratic regression."

    
    def run(self, mode='train', TrainData=None, TestData=None, mliap=None):
        """ Command PyXtal_FF to run in 2 modes:
        1. train
            In train mode, PyXtal_FF needs TrainData and/or TestData to be defined.
            
            TrainData: str
                TrainData indicates the location the training data set file.
                After the training of Neural Network is over, TrainData will be 
                self-evaluated for the accuracy.
            
            TestData: str
                TestData is an optional argument. If the test data set is present, 
                PyXtal_FF will evaluate the accuracy of the developed potential 
                with the TestData.
        

        2. predict
            In predict mode, PyXtal_FF need the saved machine learning interatomic
            potential.

            mliap: str
                The machine learning interatomic potential.
        """
        if mode == 'train':
            # Instantiate model
            self._MODEL(self._model, self._descriptors['type'])

            # Calculate descriptors.
            self._descriptors.update({'N': self._descriptors['N_train']})
            self.TrainFeatures, self.TrainDescriptors = convert_to_descriptor(
                                                        TrainData,
                                                        self.path+'Train_',
                                                        self._descriptors)

            if TestData is not None:
                EvaluateTest = True
                self._descriptors.update({'N': self._descriptors['N_test']}) 
                self.TestFeatures, self.TestDescriptors = convert_to_descriptor(
                                                          TestData,
                                                          self.path+'Test_',
                                                          self._descriptors)
            else:
                EvaluateTest = False
            
            print("==================================== Training ====================================\n")
            self.model.train(self.TrainDescriptors, self.TrainFeatures, 
                             optimizer=self.optimizer)
            self.model.save_checkpoint(des_info=self._descriptors)
            print("==================================================================================\n")
            
            print(f"============================= Evaluating Training Set ============================\n")
            train_stat = self.model.evaluate(self.TrainDescriptors, self.TrainFeatures,
                                             figname='Train.png')
            print("==================================================================================\n")

            if EvaluateTest:
                print("============================= Evaluating Testing Set =============================\n")
                test_stat =  self.model.evaluate(self.TestDescriptors, self.TestFeatures,
                                                 figname='Test.png')
                print("==================================================================================\n")
            else:
                test_stat = None

            return (train_stat, test_stat)
        
        elif mode == 'predict':
            self._model['algorithm'] = torch.load(mliap)['algorithm']
            self.algorithm = self._model['algorithm']
            self._MODEL(self._model, self._descriptors['type'])
            self._descriptors = self.model.load_checkpoint(filename=mliap)
            self._descriptors['force'] = True
            self._descriptors['stress'] = True

    
    def _MODEL(self, model, descriptors_type):
        """ Machine learning model is created here. """
                    
        if self.algorithm == 'NN':
            _model = {'system': None,
                      'hiddenlayers': [6, 6],
                      'activation': ['Tanh', 'Tanh', 'Linear'],
                      'random_seed': None,
                      'epoch': 100,
                      'batch_size': None,
                      'device': 'cpu',
                      'force_coefficient': 0.03,
                      'stress_coefficient': None,
                      'stress_group': None,
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

            # If LBFGS is used, epoch is 1. Also, batch_size = None.
            if _model['optimizer']['method'] in ['lbfgs', 'LBFGS', 'lbfgsb']:
                if 'max_iter' in _model['optimizer']['parameters'].items():
                    if _model['epoch'] > _model['optimizer']['parameters']['max_iter']:
                        _model['optimizer']['parameters']['max_iter'] = _model['epoch']
                else:
                    _model['optimizer']['parameters']['max_iter'] = _model['epoch']
                _model['epoch'] = 1
                _model['batch_size'] = None

            self.model = NeuralNetwork(elements=_model['system'],
                                       hiddenlayers=_model['hiddenlayers'],
                                       activation=_model['activation'],
                                       random_seed=_model['random_seed'],
                                       epoch=_model['epoch'],
                                       batch_size=_model['batch_size'],
                                       device=_model['device'],
                                       alpha=_model['alpha'],
                                       force_coefficient=_model['force_coefficient'],
                                       stress_coefficient=_model['stress_coefficient'],
                                       stress_group=_model['stress_group'],
                                       softmax_beta=_model['softmax_beta'],
                                       unit=_model['unit'],
                                       logging=_model['logging'],
                                       restart=_model['restart'],
                                       path=_model['path'])
            self.optimizer = _model['optimizer']
                
        elif self.algorithm == 'PR':
            _model = {'system': None,
                      'force_coefficient': 0.0001,
                      'stress_coefficient': None,
                      'stress_group': None,
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
                            stress_coefficient=_model['stress_coefficient'],
                            stress_group=_model['stress_group'],
                            order=_model['order'],
                            path=_model['path'],
                            alpha=_model['alpha'],
                            norm=_model['norm'],
                            d_max=_model['d_max'])
            self.optimizer = None

        elif self.algorithm == 'GPR':
            _model = {'force_coefficient': 0.0001,
                      'stress_coefficient': None,
                      'stress_group': None,
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
                                         stress_coefficient=_model['stress_coefficient'],
                                         stress_group=_model['stress_group'],
                                         epoch=_model['epoch'],
                                         path=_model['path'],)
            self.optimizer = _model['optimizer']


    def print_descriptors(self, _descriptors):
        """ Print the descriptors information. """

        print('Descriptor parameters:')
        keys = ['type', 'Rc', 'force', 'stress']
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
        print('================================= version', __version__,'=================================\n')
