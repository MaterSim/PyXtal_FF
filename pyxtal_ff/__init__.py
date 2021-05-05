import os
import torch
from pyxtal_ff.version import __version__
from pyxtal_ff.utilities import Database
from pyxtal_ff.models.polynomialregression import PR
from pyxtal_ff.models.neuralnetwork import NeuralNetwork

class PyXtal_FF():
    def __init__(self, descriptors=None, model=None, logo=True):
        """ PyXtal_FF develops Machine Learning Interatomic Potential.
        
        Parameters
        ----------
        descriptors: dict
            The atom-centered descriptors parameters are defined here.
            
            The list of the descriptors keys:
            - type: str (SO4)
                The type of atom-centered descriptors.
                + ACSF (BehlerParrinello Gaussian symmetry)
                + wACSF (weighted Gaussian symmetry)
                + EAD (embeded atom density)
                + SO4 (bispectrum)
                + SO3 (smoothed powerspectrum)
                + SNAP (similar to SO4 but the weighting and Rc schemes are adopted from LAMMPS)
            - Rc: float/dictionary
                The radial cutoff of the descriptors. Dictionary form is particularly for SNAP.
            - weights: dictionary
                The relative species weights.
            - N_train: int
                The number of crystal structures in training data set 
                to be converted into descriptors.
            - N_test: int
                The number of crystal structures in test data set 
                to be converted into descriptors.
            - ncpu: int
                The number of cpu core to use for converting crystal structures 
                into descriptors.
            - stress: bool (False)
                Compute rdxdr (needed for stress calculation) or not
            - force: bool (True)
                Compute dxdr (needed for force calculation) or not
            - cutoff: str
                The cutoff function.
            - parameters: dict
                Example,
                + BehlerParrinello
                    {'G2': {'eta': [.3, 2.], 'Rs': [1., 2.]}, 
                     'G4': {'eta': [.3, .7], 'lambda': [-1, 1], 'zeta': [.8, 2]}}
                + EAD
                    {'L': 2, 'eta': [.3, .7], 'Rs': [.1, .2]}
                + SO4/Bispectrum
                    {'lmax': 3}
                + SO3
                    {'nmax': 1, 'lmax': 3, 'alpha': 2.0}
                + SNAP
                    {'weights': {'Si': 1.0, 'O': 2.0},
                     'Rc': {'Si': 4.0, 'O': 5.0}
            - zbl: dict
                {'inner': 4.0, 'outer': 4.5}

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
            - restart: str (*NN)
                To continue Neural Network training from where it was left off.
            - optimizer: dict (*NN and *GPR)
                Define the optimization method used to update NN parameters.
            - path: str (*NN, *PR, *GPR)
                The user defined path to store the NN results.
                path has to be ended with '/'.
            - memory: str (*NN)
                There are two options: 'in' or 'out'. 'in' will use load all
                descriptors to memory as 'out' will call from disk as needed.
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
        descriptors_keywords = ['type', 'Rc', 'weights', 'N_train', 'N_test', 'cutoff',
                                'force', 'stress', 'ncpu', 'parameters', 'base_potential']
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
                             'weights': None,
                             'N': None,
                             'N_train': None,
                             'N_test': None,
                             'ncpu': 1,
                             'force': True,
                             'stress': True,
                             'cutoff': 'cosine',
                             'base_potential': False,
                             }
        
        # Update the default based on user-defined descriptors
        if descriptors is not None:
            self._descriptors.update(descriptors)
            if 'type' in descriptors and descriptors['type'] in ['EAD', 'ead']:
                _parameters = {'L': 3, 'eta': [0.1], 'Rs': [1.]}
            elif 'type' in descriptors and descriptors['type'] in ['SO3', 'SOAP']:
                _parameters = {'nmax': 3, 'lmax': 3, 'alpha': 2.0}
            else:
                _parameters = {'lmax': 3, 'rfac': 0.99363, 'normalize_U': False}
            if 'parameters' in descriptors:
                _parameters.update(descriptors['parameters'])
                self._descriptors['parameters'] = _parameters

            # Check for the SNAP type
            if self._descriptors['type'] in ['SNAP', 'snap']:
                if not isinstance(self._descriptors['weights'], dict):
                    msg = "The weights for SNAP type must be defined as a dictionary."
                    raise ValueError(msg)
                #if not isinstance(self._descriptors['Rc'], dict):
                #    msg = "The Rc for SNAP type must be defined as a dictionary."
                #    raise ValueError(msg)

        # Create new directory to dump all the results.
        # E.g. for default 'Si-O-Bispectrum/'
        if 'path' in model:
            self.path = model['path']
        else:
            _system = model['system']
            self.path = "-".join(_system) + "-"
            self.path += self._descriptors['type'] + "/"

        if logo:
            if not os.path.exists(self.path):
                os.mkdir(self.path)
            self.print_descriptors(self._descriptors)
        
        # Checking the keys in model.
        keywords = ['algorithm', 'system', 'hiddenlayers', 'activation', 
                    'random_seed', 'force_coefficient', 'unit', 'softmax_beta', 
                    'restart', 'optimizer', 'path', 'order', 'd_max', 
                    'epoch', 'device', 'alpha', 'batch_size', 'noise', 'kernel',
                    'norm', 'stress_coefficient', 'stress_group', 'memory']
        for key in model.keys():
            if key not in keywords:
                msg = f"Don't recognize {key} in model. "+\
                      f"Here are the keywords: {keywords}."
                raise NotImplementedError(msg)
        
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
        
        self._model = model

    def todict(self):
        return {"descriptor": self._descriptors, "model": self._model}

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
            assert TrainData is not None, "TrainData can't be None for train mode."

            # Instantiate model
            self._MODEL(self._model)

            # Calculate descriptors.
            self._descriptors.update({'N': self._descriptors['N_train']})
            if not os.path.exists(self.path+'Train_db.dat') and not os.path.exists(self.path+'Train_db.db'):
                trainDB = Database(name=self.path+'Train_db')
                trainDB.store(TrainData, self._descriptors, True, self.path+'ase.db')
            else:
                trainDB = Database(name=self.path+'Train_db')
                trainDB.store(TrainData, self._descriptors, False)
            trainDB.close()

            if TestData is not None:
                EvaluateTest = True
                self._descriptors.update({'N': self._descriptors['N_test']}) 
                if not os.path.exists(self.path+'Test_db.dat'):
                    testDB = Database(name=self.path+'Test_db')
                    testDB.store(TestData, self._descriptors, True, self.path+'ase.db')
                else:
                    testDB = Database(name=self.path+'Test_db')
                    testDB.store(TestData, self._descriptors, False)
                testDB.close()

            else:
                EvaluateTest = False
            
            print("=========================== Training =============================\n")

            self.model.train('Train_db', optimizer=self.optimizer)
            self.model.save_checkpoint(des_info=self._descriptors)
            
            print("==================================================================\n")
            
            print(f"==================== Evaluating Training Set ====================\n")

            train_stat = self.model.evaluate('Train_db', figname='Train.png')
            
            print("==================================================================\n")

            if EvaluateTest:
                print("================= Evaluating Testing Set =====================\n")

                test_stat =  self.model.evaluate('Test_db', figname='Test.png')
                
                print("==============================================================\n")
            else:
                test_stat = None

            return (train_stat, test_stat)
        
        elif mode == 'predict':
            self._model['algorithm'] = torch.load(mliap)['algorithm']
            self.algorithm = self._model['algorithm']
            self._MODEL(self._model)
            self._descriptors = self.model.load_checkpoint(filename=mliap)

    
    def _MODEL(self, model):
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
                      'restart': None,
                      'path': self.path,
                      'memory': 'in',
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

                # In full batch-LBFGS, epoch is 1. 
            if _model['optimizer']['method'] in ['lbfgs', 'LBFGS', 'lbfgsb']:
                if _model['batch_size'] is None: #full batch
                    _model['optimizer']['parameters']['max_iter'] = _model['epoch']
                    _model['epoch'] = 1
                else:
                    _model['optimizer']['parameters']['max_iter'] = 20

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
                                       restart=_model['restart'],
                                       path=_model['path'],
                                       memory=_model['memory'])
            self.optimizer = _model['optimizer']
                
        elif self.algorithm == 'PR':
            _model = {'system': None,
                      'force_coefficient': 0.0001,
                      'stress_coefficient': None,
                      'stress_group': None,
                      'order': 1,
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


    def print_descriptors(self, _descriptors):
        """ Print the descriptors information. """

        print('Descriptor parameters:')
        keys = ['type', 'Rc', 'cutoff']
        for key in keys:
            print('{:12s}: {:}'.format(key, _descriptors[key]))

        if _descriptors['type'] in ['SO4', 'Bispectrum']:
            key_params = ['lmax', 'normalize_U']
        elif _descriptors['type'] in ['SO3', 'SOAP']:
            key_params = ['nmax', 'lmax', 'alpha']
        elif _descriptors['type'] in ['SNAP', 'snap']:
            key_params = ['lmax', 'rfac']
        elif _descriptors['type'] == 'EAD':
            key_params = ['L', 'eta', 'Rs']
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
        print('        A Python package for Machine Learning Interatomic Force Field')
        print('         Developed by Zhu\'s group at University of Nevada Las Vegas')
        print('    The source code is available at https://github.com/qzhu2017/FF-project')
        print("\n")
        print('=========================== version', __version__,'=============================\n')
