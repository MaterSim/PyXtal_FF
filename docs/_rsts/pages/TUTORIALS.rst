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
Four types of descriptors are available (see `Atomic Descriptors <_background.html#atomic-descriptors>`_). 
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

    descriptor = {'type': 'BehlerParrinello',
                  'parameters': parameters,
                  'Rc': 5.0,
                 }

- ``EAMD``

.. code-block:: Python

    parameters = {'L': 2, 'eta': [0.36],
                  'Rs': [0.  , 0.75, 1.5 , 2.25, 3.  , 3.75, 4.5]}
    
    descriptor = {'type': 'EAMD',
                  'parameters': parameters,
                  'Rc': 5.0,
                  }
    

- ``Bispectrum``

.. code-block:: Python

    descriptor = {'Rc': 5.0,
                  'parameters': {'lmax': 3},
                 }


- ``SOAP``

.. code-block:: Python

    descriptor = {'type': 'SOAP',
                  'Rc': 5.0,
                  'parameters': {'lmax': 4, 'nmax': 3},
                  'ncpu': 4,
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
Usually, one only needs to specify the ``method``.
If no optimizer is defined, ``L-BFGS-B`` with a maximum iteration of 100 will be used.

Setting the NN parameters
-------------------------
.. code-block:: Python

    model = {'system' : ['Si','O'],
             'hiddenlayers': [30, 30],
             'activation': ['tanh', 'tanh', 'linear'], 
             'batch_size': None,
             'epoch': 1000,
             'force_coefficient': 0.05,
             'alpha': 1e-5,
             'path': 'SiO2-BehlerParrinello/',
             'restart': None, #'SiO2-BehlerParrinello/30-30-checkpoint.pth',
             'optimizer': {'method': 'lbfgs'},
             }

- ``system``: a list of elements involved in the training, *list*, e.g., ['Si', 'O'] 
- ``hiddenlayers``: the nodes information used in the training, *list or dict*, default: [6, 6],
- ``activation``: activation functions used in each layer, *list or dict*, default: ['tanh', 'tanh', 'linear'],
- ``batch_size``: the number of samples (structures) used for each iteration of NN; *int*, default: all structures,
- ``force_coefficient``: parameter to scale the force contribution relative to the energy in the loss function; *float*, default: 0.03,
- ``stress_coefficient``: balance parameter to scale the stress contribution relative to the energy. *float*, default: None,
- ``alpha``: L2 penalty (regularization term) parameter; *float*, default: 1e-5,
- ``restart``: dcontinuing Neural Network training from where it was left off. *string*, default: None.
- ``optimizer``: optimizers used in NN training. 
- ``epoch``: A measure of the number of times all of the training vectors are used once to update the weights. *int*, default: 100.

Note that a lot of them have the default parameters. So the simplest case to define the model is to just define the ``system`` key:

.. code-block:: Python

    model = {'system' : ['Si','O']}

Also, you can just pick the values from a previous run by defining the ``restart`` key:

.. code-block:: Python

    model = {'restart': 'Si-O-BehlerParrinello/30-30-parameters.json'}


Setting the linear regression models
------------------------------------
.. code-block:: Python

    model = {'algorithm': 'PR',
             'system' : ['Si'],
             'force_coefficient': 1e-4,
             'order': 1,
             'alpha': 0,
            }

- ``alpha``: L2 penalty (regularization term) parameter; *float*, default: 1e-5,
- ``order``: linear regression (1) or quadratic fit (2)



Invoking the simulation
-----------------------
Finally, one just need to load the defined data, descriptors and NN model to PyXtal_FF and execute the ``run`` function.

.. code-block:: Python

    ff = PyXtal_FF(descriptors=descriptor, model=model)
    ff.run(TrainData=TrainData, TestData=TestData,)

.. [1] https://docs.scipy.org/doc/scipy-0.13.0/reference/generated/scipy.optimize.minimize.html
.. [2] https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html
