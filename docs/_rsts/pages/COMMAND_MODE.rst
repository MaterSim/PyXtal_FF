Run PyXtal_FF as an executable
==============================

:: 

    Usage: pyxtal_ff [options]
    
    Options:
      -h, --help            show this help message and exit
      -s dataset, --source=dataset
                            path of data source, string
      -l HIDDENLAYER, --hiddenlayer=HIDDENLAYER
                            hidden layer, ints separated by comma, e.g. 6, 6
      -a ACTIVATION, --activation=ACTIVATION
                            activation function, e.g., tanh, tanh
      -d, --derivative      calculate dxdr or not, default: False
      -e ELEMENTS, --elements=ELEMENTS
                            elements separated by comma, e.g., Si, O
      -r RUNNER, --runner=RUNNER
                            backend for NN training, numpy, cupy or torch
      -i ITERATION, --iteration=ITERATION
                            number of iterations for NN training, default: 100
      -o OPTIMIZATION, --optimization=OPTIMIZATION
                            optimization method used in trainning
      -t TYPE, --type=TYPE  descriptor type: default: Bispectrum
      -c CUTOFF, --cutoff=CUTOFF
                            cut off for descriptor calc: default: 6.0
    

``$ pyxtal_ff -s ../pyxtal_ff/datasets/Si/UCSD/test.json -e Si``

::

            ______       _    _          _         _______ _______ 
            (_____ \     \ \  / /        | |       (_______|_______)
             _____) )   _ \ \/ / |_  ____| |        _____   _____   
            |  ____/ | | | )  (|  _)/ _  | |       |  ___) |  ___)  
            | |    | |_| |/ /\ \ |_( ( | | |_______| |     | |      
            |_|     \__  /_/  \_\___)_||_|_(_______)_|     |_|      
                   (____/      
    
    
    ------------------------(version 0.0.1 )----------------------
    
    A Python package for Machine learning of interatomic force field
    The source code is available at https://github.com/qzhu2017/FF-project
    Developed by Zhu's group at University of Nevada Las Vegas
    
    
    The following parameters are used in descriptor calculation
    type        : Bispectrum
    Rc          : 6.0
    derivative  : False
    lmax        : 3
    rfac        : 1.0
    opt         : polynomial

    ...
    ...

    Epoch:  119, Loss:       0.00010262
    Epoch:  120, Loss:       0.00009914
    Epoch:  121, Loss:       0.00009614
    Epoch:  122, Loss:       0.00008880
    Epoch:  123, Loss:       0.00008545
    The training time: 9.67 s
    ==================== Training is Completed ===================
    ======================== Evaluating ==========================
    Energy R2  : 0.999394
    Energy MAE : 0.005685
    
    Export the figure to: Si-Bispectrum/Train.png
    =================== Evaluation is Completed ==================

``$ pyxtal_ff -s ../pyxtal_ff/datasets/Si/UCSD/test.json -e Si -d -i 200``

::

    

             ______       _    _          _         _______ _______
            (_____ \     \ \  / /        | |       (_______|_______)
             _____) )   _ \ \/ / |_  ____| |        _____   _____
            |  ____/ | | | )  (|  _)/ _  | |       |  ___) |  ___)
            | |    | |_| |/ /\ \ |_( ( | | |_______| |     | |
            |_|     \__  /_/  \_\___)_||_|_(_______)_|     |_|
                   (____/
    
    
    ------------------------(version 0.0.1 )----------------------
    
    A Python package for Machine learning of interatomic force field
    The source code is available at https://github.com/qzhu2017/FF-project
    Developed by Zhu's group at University of Nevada Las Vegas
    
    
    The following parameters are used in descriptor calculation
    type        : Bispectrum
    Rc          : 6.0
    derivative  : True
    lmax        : 3
    rfac        : 1.0
    opt         : polynomial
    
    
    Computing the descriptors -----------
    load the precomputed values from  /anaconda3/lib/python3.6/site-packages/pyxtal_ff-0.0.1-py3.6.egg/pyxtal_ff/descriptors/Wigner_coefficients.npy
      25 out of   25
    Computed descriptors for 25 entries
    Saved the descriptors to  Si-Bispectrum/Train.npy
    
    
    ========================== Training ==========================
    Restart: None
    Runner: numpy
    Batch: 25
    Optimizer: L-BFGS-B
    Jacobian: True
    
    Epoch:    1, Loss:     110.56659557
    Epoch:    2, Loss:   16608.98301313
    ...
    ...
    Epoch:  230, Loss:       0.05970341
    Epoch:  231, Loss:       0.05969028
    Epoch:  232, Loss:       0.05966836
    Epoch:  233, Loss:       0.05965355
    Epoch:  234, Loss:       0.05964018
    Epoch:  235, Loss:       0.05963363
    Epoch:  236, Loss:       0.05963006
    Epoch:  237, Loss:       0.05961346
    The training time: 122.51 s
    ==================== Training is Completed ===================
    ======================== Evaluating ==========================
    Energy R2  : 0.079187
    Energy MAE : 0.282949
    Force R2   : -0.168294
    Force MAE  : 0.615777
    
    Export the figure to: Si-Bispectrum/Train.png
    =================== Evaluation is Completed ==================


