Background and Theory
=========================
The recent emergence of machine learning techniques have greatly influenced the methods used in solving material science problems. In particular, constructing a potential energy surface (PES) of a given system is a remarkable task. Traditionally, the quantum mechanical simulations based on density functional theory (DFT) has been widely used to describe accurate PES. Despite of the availability of supercomputers, the computational cost using DFT method is still a bottleneck to many important problems in materials science. On the other hand, a force field or interatomic potential method offers much faster computations. This potential contains a set of parameters, which is fitted to DFT and/or experimental data, used to compute the potential energy of a system based on classical physics. While trying to describe quantum mechanical system, the for    ce field method sacrifices accuracy in the gain of faster computation.

Recently, machine learning based IAP (MLIAP) is proposed to solve the inaccuracy issue while retaining the speed. The power of MLIAP is illustrated by the applications to a range of materials. MLIAP is proven not only to yield nearly quantum mechanic accuracy, but also allows the investigation of materials properties at larger time and size scales with molecular dynamics (MD) simulations. Among various ML models, the artificial neural network and Gaussian process regression are currently favored in ML potentials development.

PyXtalFF involves two important components: `crystal descriptors` and `force field training`. We focus on two types of crystal descriptors, `Behler-Parrinello` and `bispectrum coefficient`. For the force field training, we consider the artificial neural networks at the moment.
 
Atomic Descriptors
------------------
Descriptors---representations of a crystal structure---play an essential part in constructing MLIAP. Due to the periodic boundary conditions, Cartesian coordinates poorly describe the structural environment. While the energy of a crystal structure remains unchanged, the Cartesian coordinates change as transnational or rotational operation is applied to it [1]_. Thus, physically meaningful descriptors must withhold the change as the alterations are performed to the structural environment. In another words, the descriptors need to be invariant with respect to translation and rotational operations, and the exchanges of any equivalent atom. 


Gaussian Symmetry Function
^^^^^^^^^^^^^^^^^^^^^^^^^^
Behler-Parrinello method---atom-centered descriptors---utilizes a set of symmetry functions [2]_. The symmetry functions map two atomic Cartesian coordinates to a distribution of distances between atom (radial functions) or three atomic Cartesian coordinates to a distribution of bond angles (angular functions). These mappings are invariant with respect to translation, rotation, and permutation of atoms of the system. Therefore, the energy of the system will remain unchanged under these mapping of symmetry functions.
 
In PyXtal_FF, two types of Gaussian functions (:math:`\textbf{G}^2` and :math:`\textbf{G}^4`) are supported:

.. math::
    \textbf{G}^{2}_i = \sum_{j\neq i} e^{-\boldsymbol{\eta} \otimes (\textbf{R}_{ij}-\textbf{R}_s)^2} \textbf{f}_c(\textbf{R}_{ij}; R_c)

.. math::
    \textbf{G}^{4}_i = 2^{1-\boldsymbol{\zeta}}\sum_{j\neq i} \sum_{k \neq i, j} (1+\boldsymbol{\lambda} \otimes \cos \boldsymbol{\theta}_{ijk})^{\boldsymbol{\zeta}}  e^{-\boldsymbol{\eta} \otimes (\textbf{R}_{ij}^2 + \textbf{R}_{ik}^2 + \textbf{R}_{jk}^2)} \textbf{f}_c(\textbf{R}_{ij})  \textbf{f}_c(\textbf{R}_{ik}) \textbf{f}_c(\textbf{R}_{jk})
    

where :math:`\eta, \textbf{R}_s, \lambda, \zeta` represent the column vectors of parameters, :math:`i, j, k` denotes the atomic indices. :math:`\textbf{f}_c(\textbf{R}_{ij}; R_c)` is a cutoff function to ensure that the values go to zero smoothly.

.. math::
    f_c(\boldsymbol{r}) = \begin{cases}
    \frac{1}{2}\left[\cos\left(\frac{\pi| \boldsymbol{r_i} |}{R_c}\right) + 1\right],& |\boldsymbol{r_i}| \leq R_c\\
    0,              & |\boldsymbol{r_i}| > R_c
    \end{cases}

In order to calculate the force, :math:`\frac{\partial G}{\partial r}` are also needed (to add the derivatives).

Bispectrum Coefficients
^^^^^^^^^^^^^^^^^^^^^^^
Similar to Gaussian symmetry functions, SO(4) bispectrum can be used to represent the local atomic environments. It was first introduced by Bartok [3]_. Later, Thompson *et al.* proposed the spectral neighbor analysis method (SNAP) method and demonstrated that the SO(4) bispectrum could achieve satisfactory accuracy based on the simple linear [4]_ and quadratic regressions [5]_. Following the original work, the expression of SO(4) bispectrum is formed by the expansion coefficients of 4D hyperspherical harmonics:

.. math::
    B_{i}^{l_1,l_2,l} = \sum_{m, m'=-l}^{l} c^{l}_{m',m} 
    \sum_{m_1, m_1'=-l_1}^{l_1} \sum_{m_2, m_2'=-l_2}^{l_2}c^{l_1}_{m_1',m_1} c^{l_2}_{m_2',m_2} H^{l, m, m'}_{l_1,m_1,m_1',l_2,m_2,m_2'}

where :math:`H^{l_1, l_2, l}_{m_1',m_2',m',m_1,m_2,m}` is the analog to the Clebsch-Gordan coefficients on the 3-sphere. In application, it is the product of two ordinary Clebsch-Gordan coefficients on the 2-sphere. :math:`c^{l,m}_{l_1, m_1, l_2, m_2}` are the expansion coefficients from the hyperspherical harmonics (:math:`U^{l}_{m',m}`) functions that are projected from the atomic neighborhood density within a cutoff radius onto the surface of four-dimensional sphere:

.. math::
    \rho = \sum_{l=0}^{+\infty}\sum_{m=-l}^{+l}\sum_{m'=-l}^{+l}c^l_{m',m}U^{l}_{m',m}

where the expansion coefficients are defined as

.. math::
    c^l_{m',m} = \left<U^l_{m',m}|\rho\right>

Expression of Energy and Forces
--------------------------------
With invariant descriptors (:math:`\delta`) for each atom, the total energy as a sum of each atomic energy can be written in the following form:

.. math::
    E_s = \sum_i^{\textrm{all atoms}} \textrm{E}_i(\delta_i) 

:math:`\delta_i` is a function of atomic environment (:math:`\textbf{R}_i`) within a cutoff distance (:math:`R_c`).

The atomic energy contributions depend on the local structural environment within a cutoff radius with respect to the center atom *i*. Furthermore, accurate representation of PES is also dependent on the contributions of forces. The force acted on atom j can be expressed by the negative gradient of the energy with respect to its atomic positions (:math:`\boldsymbol{r}_j`):

.. math::
     \boldsymbol{F}_j=-\sum_i ^{\textrm{all atoms}} \frac{\partial E_i(\boldsymbol{X}_{ij})}{\partial \boldsymbol{X}_{ij}} \cdot \frac{\partial
    \boldsymbol{X}_{ij}}{\partial \boldsymbol{r}_j}

The functional forms of *E* and *F* are fully dependent on the regression algorithm. Generalized linear regression and neural network (NN) regression will be discussed in the following sections. 

Regression Techniques
----------------------
The objective in force field fitting is to obtain an explicit (or implicit) functional form which leads to the minimum error compared the energy and forces from quantum calculations. 

Objective Loss Function
^^^^^^^^^^^^^^^^^^^^^^^
We can define the objective function as follows,

.. math::
    \Delta = \frac{1}{2s}\sum_{i=1}^s\Bigg[\bigg(\frac{E_i - E^{\textrm{Ref}}_i}{N_{\textrm{atom}}^i}\bigg)^2 +
             \frac{\beta} {3N_{\textrm{atom}}^i}\sum_{j=1}^{3N_{\textrm{atom}}^i}
    (F_{i, j} - F_{i, j}^{\textrm{Ref}})^2 \Bigg]
    + \frac{\alpha}{2s} \sum_{i=1}^{m} (\boldsymbol{w}^i)^2

where *s* is the total number of structures, *i* loops over all structures, and *j* loops over all atoms for each structure *i* in all directions. :math:`N^{\textrm{atom}}_i` is the total number of atoms in the structure *i*. Here, :math:`\beta` is acting as the balance parameters, as the number of force components is much larger than the number of energies. The cost function compares the predicted values obtained from the regression (:math:`E_i` and :math:`F_{i, j}`) to the true values of :math:`E^{\textrm{Ref}}` and :math:`F_{i, j}^{\textrm{Ref}}`. Then, the optimum solution can be solved by finding the **w** leading to the zero partial derivative of :math:`\Delta` with respect to each element in **w**. 

Linear Regression
^^^^^^^^^^^^^^^^^^

Neural Networks
^^^^^^^^^^^^^^^

Gaussian Process Regression
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
We can define the objective function as follows,
Type of nns, optimazation techniques, etc.


References
----------

.. [1] A. P. Bartok, R. Kondor and G. Csanyi, Phys. Rev. B 87, 184115 (2013)
.. [2] J. Behler and M. Parrinello,  Phys. Rev. Lett. 98, 146401 (2007)
.. [3] A. P Bartok, M. C Payne, R. Kondor and G. Csanyi,  Phys. Rev. Lett. 104, 136403 (2010)
.. [4] A. P. Thompson, et. al., J. Comput. Phys. 285, 316–330 (2015) 
.. [5] M. A. Wood and A. P. Thompson, J. Chem. Phys. 148, 241721 (2018).



