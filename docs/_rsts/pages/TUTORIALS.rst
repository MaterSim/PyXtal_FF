Step by Step instruction
========================

This page illustrates how to run the simulation from the scratch. 

.. code-block:: Python

    from pyxtal_ff import PyXtal_FF

Define the source of data
-------------------------
.. code-block:: Python

    TrainData = "pyxtal_ff/datasets/SiO2/OUTCAR_SiO2"
    
At the moment, we accept the various formats:

- ``ase.db``
- ``json``
- ``OUTCAR``
 
In principle, one can easily write a utility function to follow the style as shown in the `utility section <https://pyxtal-ff.readthedocs.io/en/latest/pyxtal_ff.utilities.html#pyxtal_ff.utilities.parse_json>`_.

Among all different formats, we recommend the use of `ase.db <https://wiki.fysik.dtu.dk/ase/ase/db/db.html>`_. Following ase db, you use need to add the following additional tags to each ``atoms`` object,

.. code-block:: Python

 
    from ase.db import connect

    # Suppose you have the following variables
    # - struc: ase atoms objects
    # - eng: total DFT energy
    # - forces: DFT Forces: N*3 array
    # - stress: DFT Stress: 1*6 stress [in GPa, xx, yy, zz, xy, xz, yz]
    # - db_name: the filename to store the information and pass to pyxtal_ff
    
    data = {'dft_energy': eng,      
            'dft_force': forces,  
            'dft_stress': stress,    
            #'group': group,
           }

    with connect(db_name) as db:  
        db.write(struc, data=data)
        
Note that different codes arrange the stress tensor in different order and unit. For PyXtal\_FF, we strictly use ``GPa`` and the order of ``[xx, yy, zz, xy, xz, yz]``.
    
Choosing the descriptor
------------------------
Four types of descriptors are available (see `Atomic Descriptors <_background.html#atomic-descriptors>`_). 
Each of them needs some additional parameters to be defined as follows.

- ``BehlerParrinello`` (ACSF, wACSF)

.. code-block:: Python

    parameters = {'G2': {'eta': [0.003214, 0.035711, 0.071421, 
                               0.124987, 0.214264, 0.357106],
                         'Rs': [0]},
                  'G4': {'lambda': [-1, 1],
                       'zeta': [1],
                       'eta': [0.000357, 0.028569, 0.089277]}
                 }

    descriptor = {'type': 'ACSF',
                  'parameters': parameters,
                  'Rc': 5.0,
                 }

The ``wACSF`` is also supported. In this case, the number of descriptors will linearly dependent on the number of atoms in the system.

- ``EAD``

.. code-block:: Python

    parameters = {'L': 2, 'eta': [0.36],
                  'Rs': [0.  , 0.75, 1.5 , 2.25, 3.  , 3.75, 4.5]}
    
    descriptor = {'type': 'EAD',
                  'parameters': parameters,
                  'Rc': 5.0,
                  }
    

- ``SO4``

.. code-block:: Python

    descriptor = {'type': 'SO4',
                  'Rc': 5.0,
                  'parameters': {'lmax': 3},
                 }


- ``SO3``

.. code-block:: Python

    descriptor = {'type': 'SO3',
                  'Rc': 5.0,
                  'parameters': {'lmax': 4, 'nmax': 3},
                 }


.. _defOptim:

Defining your optimizer
-----------------------

The optimizer is defined by a dictionary which contains 2 keys: 

- ``method`` 
- ``parameters``

Currently, the ``method`` options are 

- ``L-BFGS-B`` 
- ``SGD`` 
- ``ADAM`` 

If ``SGD`` or ``ADAM`` is chosen, the default learning rate is 1e-3.
Usually, one only needs to specify the ``method``.
If no optimizer is defined, ``L-BFGS-B`` will be used.

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
    ff.run(TrainData=TrainData, TestData=TestData)

