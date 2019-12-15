Step by Step instruction
========================

This page illustrates how to run the simulation from the scratch. 

.. code-block:: Python

    from pyxtal_ff import PyXtal_FF

Define the source of data
-------------------------
.. code-block:: Python

    TrainData = "pyxtal_ff/datasets/SiO2/OUTCAR_SiO2"
    
Choosing the desrcriptor
------------------------
Two types of descriptors are available (see `Atomic Descriptors <_background.html#atomic-descriptors>`_). 
Each of them needs some additional parameters to be defined as follows.

- ``BehlerParrinello`` 

.. code-block:: Python

    parameters = {'G2': {'eta': [0.003214, 0.035711, 0.071421, 
                               0.124987, 0.214264, 0.357106],
                         'Rs': [0]},
                  'G4': {'lambda': [-1, 1],
                       'zeta': [1],
                       'eta': [0.000357, 0.028569, 0.089277]}
                 }

- ``Bispectrum``

.. code-block:: Python

    parameters = {'lmax': 3, 'opt': 'polynomial', 'rfac': 1.0}


Last, these parameters should be passed to the ``function``.

.. code-block:: Python

    function = {'type': 'BehlerParrinello',
                'derivative': False,
                'parameters': parameters,
               }

.. _defOptim:

Defining your optimizer
-----------------------

The optimizer is defined by a dictionary which contains 3 keys: 

- ``method`` 
- ``derivative``
- ``parameters``

Currently, the ``method`` options are 

- ``L-BFGS-B`` (from ``scipy.optimize.minimize`` [1]_)
- ``CG`` (from ``scipy.optimize.minimize`` [1]_)
- ``BFGS`` (from ``scipy.optimize.minimize`` [1]_)
- ``SGD`` (built-in)
- ``ADAM`` (built-in)

The ``derivative`` key is optional boolean, which is True by default.
If False, the chosen method will calculate the numerical approximation of the jacobian, which is useful check if the jacobian from the NN code is correct. However, we advise that one should not set this option as False for the production runs. If ``SGD`` or ``ADAM`` is chosen, ``derivative`` has to be True.

To comply with Scipy optimizer, the parameters keys such as ``maxiter``, ``gtol``, and ``ftol`` are defined in ``option`` dictionary.

The following list describes the default for several optimizers.

L-BFGS-B
~~~~~~~~
.. code-block:: Python

    optimizer = {'method': 'L-BFGS-B',
                 'parameters': {'tol': 1e-10,
                                'options': {'maxiter': 1000,
                                            'gtol': 1e-8,
                                            'disp': False,
                                            }
                                }
                }

ADAM
~~~~
.. code-block:: Python
    
    optimizer = {'method': 'ADAM',
                 'parameters': {'tol': 1e-10,
                                'options': {'maxiter': 50000},
                                'lr_init': 0.001,
                                'beta1': 0.9,
                                'beta2': 0.999,
                                'epsilon': 10E-8,
                                't': 0,
                               }
                }
                       
SGD
~~~
.. code-block:: Python

    optimizer = {'method': 'SGD',
                 'derivative': True,
                 'parameters': {'tol': 1e-10,
                                'options': {'maxiter': 2000},
                                'lr_init': 0.001,
                                'lr_method': 'constant',
                                'power_t': 0.5,
                                'momentum': 0.9,
                                'nesterovs_momentum': True,
                               }
                }

Note: ``SGD`` and ``ADAM`` parameters are consistent with scikit-learn [2]_.
If no optimizer is defined, ``L-BFGS-B`` with a maximum iteration of 100 will be used.

Setting the NN parameters
-------------------------
.. code-block:: Python

    model = {'system' : ['Si','O'],
             'hiddenlayers': [30, 30],
             'activation': ['tanh', 'tanh', 'linear'], 
             'batch_size': None,
             'force_coefficient': 0.05,
             'alpha': 1e-5,
             'logging': logging,
             'runner': 'numpy',
             'restart': None, #'O-Si-BehlerParrinello/30-30-parameters.json'
             'optimizer': optimizer,
             }

- ``system``: a list of elements involved in the training, *list*, e.g., ['Si', 'O'] 
- ``hiddenlayers``: the nodes information used in the training, *list or dict*, default: [6, 6],
- ``activation``: activation functions used in each layer, *list or dict*, default: ['tanh', 'tanh', 'linear'],
- ``batch_size``: the number of samples (structures) used for each iteration of NN; *int*, default: all structures,
- ``force_coefficient``: parameter to scale the force contribution relative to the energy in the loss function; *float*, default: 0.03,
- ``alpha``: L2 penalty (regularization term) parameter; *float*, default: 1e-5,
- ``runner``: backend to train the NN; *string*, default: ``numpy`` (other options include ``cupy`` and ``pytorch``).
- ``restart``: dcontinuing Neural Network training from where it was left off. *string*, default: None.
- ``optimizer``: optimizers used in NN training. The default is ``L-BFGS-B optimizer`` with 100 iterations.

Note that a lot of them have the default parameters. So the simplest case to define the model is to just define the ``system`` key:

.. code-block:: Python

    model = {'system' : ['Si','O']}

Also, you can just pick the values from a previous run by defining the ``restart`` key:

.. code-block:: Python

    model = {'restart': 'Si-O-BehlerParrinello/30-30-parameters.json'}



Invoking the simulation
-----------------------
Finally, one just need to load the defined data, descriptors and NN model to PyXtal_FF and execute the ``run`` function.

.. code-block:: Python

    trainer = PyXtal_FF(TrainData=TrainData, TestData=TestData,
                 descriptors=descriptor, model=model)
    trainer.run()

.. [1] https://docs.scipy.org/doc/scipy-0.13.0/reference/generated/scipy.optimize.minimize.html
.. [2] https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html
