Installation
=======================

Dependencies
------------
`PyXtal_FF` is entirely based on Python 3. *Thus Python 2 will not be supported!*. To make it work, several major Python libraries are required. 

- `NumPy>=1.13.3 <https://www.scipy.org/scipylib/download.html>`_  
- `SciPy>=1.1.0 <https://www.scipy.org/install.html>`_  
- `Matplotlib>=2.0.0 <https://matplotlib.org>`_
- `Sklearn>=0.20.0 <http://scikit-learn.github.io/stable>`_
- `Numba>=0.44.1 <https://numba.pydata.org>`_
- `ase>=3.18.0 <https://wiki.fysik.dtu.dk/ase/>`_


Optionally, the performance of code has been optimized based on GPU accerlation for neural network training.

- `Pytorch>=1.2 <https://pytorch.org>`_ 
- `Cupy>=6.0.0 <https://cupy.chainer.org>`_ 


To install
------------

To install it, one can simply type ``pip install pyxtal_ff`` or make a copy of the source code, and then install it manually.
::

    $ git clone https://github.com/qzhu2017/PyXtal_FF.git
    $ cd FF-project
    $ python setup.py install

This will install the module. The code can be used within Python via

.. code-block:: Python

  import pyxtal_ff
  print(pyxtal_ff.__version__)


