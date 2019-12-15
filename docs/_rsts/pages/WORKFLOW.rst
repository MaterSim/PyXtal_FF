List of Important Classes & Functions
======================================
This is a reference for important classes and functions used in PyXtal_FF. Not all are listed here.

.. todo:: Please let me know which classes or functions to ADD or REMOVE if not listed or not needed.

Atomic Descriptors (*pyxtal_ff.descriptors*)
--------------------------------------------

descriptors.BehlerParrinello & descriptors.SO4_Bispectrum functions

.. table::
   :widths: auto

   =====================================   ===================================================================
   *pyxtal_ff.descriptors* **functions**
   =====================================   ===================================================================
   BehlerParrinello.calculate      	        Initiates calculation of BP Symmetry Funcs. (see :ref:`descBP`)
   SO4_Bispectrum.calculate        	        Initiates calculation of Bisepectrum Coeff. (see :ref:`descBI`)
   =====================================   ===================================================================

Neural Network (*pyxtal_ff.models*) 
-----------------------------------

Root classes and functions from the pyxtal_ff.models module

.. table::
   :widths: auto
   
   ====================================  =====================================================================
   *pyxtal_ff.models* **root classes**                         
   ====================================  =====================================================================
      models.NeuralNetwork		         Neural network base class for Energy and Force Predictions (see `pyxtal_ff.models.NeuralNetwork <../../pyxtal_ff.models.html#pyxtal_ff.models.NeuralNetwork>`_) 
      models.LossFunction	             Loss function between prediction and target (see `pyxtal_ff.models.LossFunction <../../pyxtal_ff.models.html#pyxtal_ff.models.LossFunction>`_)
      models.Regressor                   Optimzation methods used in NN training (see :ref:`defOptim`) 
      models.optimize                    Built-in SGD & ADAM optimizers (see :ref:`modelsOptim`)
   ====================================  =====================================================================

Some useful functions used during NN training from models.NeuralNetwork  

.. table::
   :widths: auto

   ==============================================  =====================================================================
   *pyxtal_ff.models.NueralNetwork* **functions**                            
   ==============================================  =====================================================================
      train			                    Initiate NN training (see `NeuralNetwork.train <../../html/pyxtal_ff.models.html#pyxtal_ff.models.NeuralNetwork.train>`_) 	
      evaluate		                            Compare predicted and test data (see `NeuralNetwork.evaluate <../../pyxtal_ff.models.html#pyxtal_ff.models.NeuralNetwork.evaluate>`_)
      export_nn_parameters		            Save NN parameters to .json file (see `NeuralNework.export_nn_parameters <../../pyxtal_ff.models.html#pyxtal_ff.models.NeuralNetwork.export_nn_parameters>`_)
      import_nn_parameters		            Load NN parameters to .json file (see `NeuralNework.import_nn_parameters <../../pyxtal_ff.models.html#pyxtal_ff.models.NeuralNetwork.import_nn_parameters>`_)
      get_random_weights                            Initialized random weights for training (see `NeuralNework.get_random_weights <../../pyxtal_ff.models.html#pyxtal_ff.models.NeuralNetwork.get_random_weights>`_)
      get_scalings                                  Scales NN output to range of true energy (see `NeuralNework.get_scalings <../../pyxtal_ff.models.html#pyxtal_ff.models.NeuralNetwork.scalings>`_)
   ==============================================  =====================================================================

