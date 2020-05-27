Background and Theory
=========================
The recent emergence of machine learning techniques have greatly influenced methods used in solving problems in material science. In particular, constructing a potential energy surface (PES) of a given system is a remarkable task. Traditionally, the PES could only be calculated to sufficient accuracy through quantum mechanical simulations based on density functional theory (DFT). Even when utilizing supercomputing resources DFT methods still are not feasible for use in many important materials science. On the other hand, a force field or interatomic potential method offers an avenue to much faster computations, with a sacrifice in accuracy. A potential is fit to DFT and/or experimental data by optimizing the potential parameters to the data which can then be used to compute the potential energy of a system.

Recently, machine learning based interatomic potentials (MLIAP) have been used to achieve greater accuracies issue while still maintaining costs orders of magnitude less than DFT methods. MLIAPs have proven not only to yield nearly quantum mechanic accuracy, but also allows the investigation of materials properties at larger time and size scales with molecular dynamics (MD) simulations. MLIAPs are most often fit using linear regression, neural network regression, and Gaussian process regression.

PyXtalFF involves two important components: descriptors and force field training. We focus on four types of descriptors, Behler-Parrinello Symmetry Functions, Embedded Atom Descriptors, SO(4) Bispectrum Components and Smooth SO(3) power spectrum. For the force field training.

For all of the regression techniques, the force field training involves fitting of energy, force, and stress simultaneously, although PyXtal_FF allows the fitting of force or stress to be optional. The energy can be written in the sum of atomic energies, in which is a functional (:math:`\mathscr{F}`) of the descriptor (:math:`\boldsymbol{X}_i`):

.. math::

   E_\textrm{total} = \sum_{i=1}^{N} E_i = \sum_{i=1}^{N} \mathscr{F}_i(\boldsymbol{X}_i)

Specifically, the functional represents regression techniques such as neural network or generalized linear regressions.

Since neural network and generalized linear regressions have well-defined functional forms, analytic derivatives can be derived by applying the chain rule to obtain the force at each atomic coordinate, :math:\boldsymbol{r}_m:

.. math::
   
   \boldsymbol{F}_m=-\sum_{i=1}^{N}\frac{\partial \mathscr{F}_i(\boldsymbol{X}_{i})}{\partial \boldsymbol{X}_{i}} \cdot \frac{\partial\boldsymbol{X}_{i}}{\partial \boldsymbol{r}_m}

Force is an important property to accurately describe the local atomic environment especially in geometry optimization and MD simulation. Finally, the stress tensor is acquired through the virial stress relation:
   
.. math::

   \boldsymbol{S}=-\sum_{m=1}^N \boldsymbol{r}_m \otimes \sum_{i=1}^{N} \frac{\partial \mathscr{F}_i(\boldsymbol{X}_{i})}{\partial \boldsymbol{X}_{i}} \cdot \frac{\partial \boldsymbol{X}_{i}}{\partial \boldsymbol{r}_m}
 
Atomic Descriptors
------------------
Descriptor---a representation of a crystal structure---plays an essential role in constructing MLFF. Due to periodic boundary conditions, Cartesian coordinates poorly describe the structural environment. While the energy of a crystal structure remains unchanged, the Cartesian coordinates change as translational or rotational operation is applied to the structure [1]_. Thus, physically meaningful descriptor must withhold the energy change as the alterations are performed to the structural environment. In another words, the descriptor needs to be invariant with respect to translation and rotational operations, and the exchanges of any equivalent atom. To ensure the descriptor mapping from the atomic positions smoothly approaching zero beyond the :math:`R_c`, a cutoff function (:math:`f_c`) is included to most decriptor mapping schemes, here the exception is the Smooth SO(3) Power Spectrum:

.. math::
    f_c(r) = \begin{cases}
        \frac{1}{2}\cos\left(\pi \frac{r}{R_c}\right) + \frac{1}{2} & r \leq R_c\\
        0              & r > R_c
    \end{cases}

In the following, the types of descriptors will be explained in details.

Behler-Parrinello Symmetry Function
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Behler-Parrinello method---atom-centered descriptors---utilizes a set of symmetry functions [2]_. The symmetry functions map two atomic Cartesian coordinates to a distribution of distances between atom (radial functions) or three atomic Cartesian coordinates to a distribution of bond angles (angular functions). These mappings are invariant with respect to translation, rotation, and permutation of atoms of the system. Therefore, the energy of the system will remain unchanged under these mapping of symmetry functions.
 
PyXtal_FF supports three types of symmetry functions:

.. math::
    G^{(2)}_i = \sum_{j\neq i} e^{-\eta (R_{ij}-\mu)^2} \cdot f_c(R_{ij})

.. math::
    G^{(4)}_i = 2^{1-\zeta}\sum_{j\neq i} \sum_{k \neq i, j} [(1+\lambda \cos \theta_{ijk})^{\zeta} \cdot e^{-\eta (R_{ij}^2 + R_{ik}^2 + R_{jk}^2)} \cdot f_c(R_{ij}) \cdot f_c(R_{ik}) \cdot f_c(R_{jk})]

.. math::
    G^{(5)}_i = 2^{1-\zeta}\sum_{j\neq i} \sum_{k \neq i, j} [(1+\lambda \cos \theta_{ijk})^{\zeta} \cdot e^{-\eta (R_{ij}^2 + R_{ik}^2)} \cdot f_c(R_{ij}) \cdot f_c(R_{ik})]

where :math:`\eta` and :math:`R_s` are defined as the width and the shift of the symmetry function. As for :math:`G^{(4)}` and :math:`G^{(5)}`, they are a few of many ways to capture the angular information via three-body interactions (:math:`\theta_{ijk}`). :math:`\zeta` determines the strength of angular information. Finally, :math:`\lambda` values are set to +1 and -1, for inverting the shape of the cosine function.

Embedded Atom Density
^^^^^^^^^^^^^^^^^^^^^

Embedded atom density (EAD) descriptor [3]_ is inspired by embedded atom method (EAM)---description of atomic bonding by assuming each atom is embedded in the uniform electron cloud of the neighboring atoms. The EAM generally consists of a functional form in a scalar uniform electron density for each of the "embedded" atom plus the short-range nuclear repulsion potential. Given the uniform electron gas model, the EAM only works for metallic systems, even so the EAM can severely underperform in predicting the metallic systems. Therefore, the density can be modified by including the square of the linear combination the atomic orbital components:

.. math::
    \rho_i(R_{ij}) = \sum_{l_x, l_y, l_z}^{l_x+l_y+l_z=L} \frac{L!}{l_x!l_y!l_z!} \bigg(\sum_{j\neq i}^{N} Z_j  \Phi(R_{ij})\bigg)^2

where :math:`Z_j` represents the atomic number of neighbor atom :math:`j`. :math:`L` is the quantized angular momentum, and :math:`l_{x,y,z}` are the quantized directional-dependent angular momentum. For example, :math:`L=2` corresponds to the :math:`d` orbital. Lastly, the explicit form of :math:`\Phi` is:

.. math::
    \Phi(R_{ij}) = x^{l_x}_{ij}  y^{l_y}_{ij}  z^{l_z}_{ij} \cdot e^{-\eta (R_{ij}-\mu)^2} \cdot f_c(R_{ij})

According to quantum mechanics, :math:`\rho` follows the similar procedure in determining the probability density of the states, i.e. the Born rule.

Furthermore, EAMD can be regarded as the improved Gaussian symmetry functions. EAMD has no classification between the radial and angular term. The angular or three-body term is implicitly incorporated in when :math:`L>0`. By definition, the computation cost for calculating EAMD is cheaper than angular symmetry functions by avoiding the extra sum of the :math:`k` neighbors. In term of usage, the parameters :math:`\eta` and :math:`\mu` are similar to the strategy used in the Gaussian symmetry functions, and the maximum value for :math:`L` is 3, i.e. up to :math:`f` orbital.

SO(4) Bispectrum Components
^^^^^^^^^^^^^^^^^^^^^^^^^^^

The SO(4) bispectrum components are another type of atom-centered descriptor based on triple correlation of the atomic neighbor density function on the 3-sphere. The distribution of atoms in an atomic environment can be represented as a sum of delta functions, this is known as the atomic neighbor density function.

.. math::
    \rho(\boldsymbol{r}) = \delta(\boldsymbol{r}) + \sum_j \delta(\boldsymbol{r}-\boldsymbol{r_j})

Then this function can mapped to the 3 sphere by mapping the atomic coordinates :math:`(x,y,z)` to the 3-sphere by the following relations:

.. math::
    \theta = arccos\left(\frac{z}{r}\right)
    
.. math::
    \phi = arctan\left(\frac{y}{x}\right)
    
.. math::
    \omega = \pi \frac{r}{r_{cut}}
    
Using this mapping, the Atomic Neighbor Density Function is then expanded on the 3-sphere using the Wigner-D matrix elements, the harmonic functions on the 3-sphere.  The resulting expansion coefficients are given by:

.. math::
    c^l_{m',m} = D^{l}_{m',m}(\boldsymbol{0}) + \sum_j D^{l}_{m',m}(\boldsymbol{r}_j)
    
The triple correlation of the Atomic Neighbor Density Function on the 3-sphere is then given by a third order product of the expansion coefficients by the Fourier theorem.

.. math::
    B_{l_1,l_2,l} = \sum_{m',m = -l}^{l}c^{l}_{m',m}\sum_{m_1',m_1 = -l_1}^{l_1}c^{l_1}_{m_1',m_1}\times \sum_{m_2',m_2 = -l_2}^{l_2}c^{l_2}_{m_2',m_2}C^{ll_1l_2}_{mm_1m_2}C^{ll_1l_2}_{m'm_1'm_2'},
    
Where C is a Clebsch-Gordan coefficient.
    
Smooth SO(3) Power Spectrum
^^^^^^^^^^^^^^^^^^^^^^^^^^
Now instead of considering a hyperdimensional space, we can derive a similar descriptor by taking the auto correlation of the atomic neighbor density function through expansions on the 2-sphere and a radial basis on a smoothened atomic neighbor density function,

.. math::
   \rho ' = \sum_i e^{-\alpha|\bm{r}-\bm{r}_i|^2}
   
This function is then expanded on the 2-sphere using Spherical Harmonics and a radial basis :math:`g_n(r)` orthonormalized on the interval :math:`(0, r_\textrm{cut})`.

.. math::
    c_{nlm} = <g_n Y_{lm}|\rho '> = 4\pi e^{-alpha r_i^2} Y^*_{lm}(\bm{r}_i)\int_0^{r_{\textrm{cut}}}r^2 g_n(r) I_l(2\alpha r r_i) e^{-alpha r^2}dr

Where :math:`I_l` is a modified spherical bessel function of the first kind.  The autocorrelation or power spectrum is obtained through the following sum.

.. math::
    p_{n_1 n_2 l} = \sum_{m=-l}^{+l}c_{n_1lm} c^*_{n_2 l m}
    

Force Field Training
--------------------

Here, we reveal the functional form (:math:`\mathscr{F}`) presented in equation above. The functional form is essentially regarded as the regression model. Each regression model is species-dependent, i.e. as the the number of species increases, the regression parameters will increase. This is effectively needed to describe the presence of other chemical types in complex system. Hence, explanation for the regression models will only consider single-species for the sake of simplicity.

Furthermore, it is important to choose differentiable functional as well as its derivative due to the existence of force (:math:`F`) and stress (:math:`S`) contribution along with the energy (:math:`E`) in the loss function:

.. math::
    \Delta = \frac{1}{2M}\sum_{i=1}^M\Bigg[\bigg(\frac{E_i - E^{\textrm{Ref}}_i}{N_{\textrm{atom}}^i}\bigg)^2 + \frac{\beta_f} {3N_{\textrm{atom}}^i}\sum_{j=1}^{3N_{\textrm{atom}}^i} (F_{i, j} - F_{i, j}^{\textrm{Ref}})^2 + \frac{\beta_s} {6} \sum_{p=0}^{2} \sum_{q=0}^{p} (S_{pq} - S_{pq}^{\textrm{Ref}})^2 \Bigg]

where M is the total number of structures in the training pool, and :math:`N^{\textrm{atom}}_i` is the total number of atoms in the :math:`i`-th structure. The superscript :math:`\textrm{Ref}` corresponds to the target property. :math:`\beta_f` and :math:`\beta_s` are the force and stress coefficients respectively. They scale the importance between energy, force, and stress contribution as the force and stress information can overwhelm the energy information due to their sizes. Additionally, a regularization term can be added to induce penalty on the entire parameters preventing overfitting:

.. math::
    \Delta_\textrm{p} = \frac{\alpha}{2M} \sum_{i=1}^{m} (\boldsymbol{w}^i)^2

where :math:`\alpha` is a dimensionless number that controls the degree of regularization.

Generalized Linear Regression
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This regression methodology is a type of polynomial regression. Essentially, the quantum-mechanical energy, forces, and stress can be expanded via Taylor series with atom-centered descriptors as the independent variables:

.. math::
    E_{\textrm{total}} = \gamma_0 + \boldsymbol{\gamma} \cdot \sum^{N}_{i=1}\boldsymbol{X}_i + \frac{1}{2}\sum^{N}_{i=1}\boldsymbol{X}_i^T\cdot \boldsymbol{\Gamma} \cdot \boldsymbol{X}_i

where :math:`N` is the total atoms in a structure. :math:`\gamma_0` and :math:`\boldsymbol{\gamma}` are the weights presented in scalar and vector forms. :math:`\boldsymbol{\Gamma}` is the symmetric weight matrix (i.e. :math:`\boldsymbol{\Gamma}_{12} = \boldsymbol{\Gamma}_{21}`) describing the quadratic terms. In this equation, we only restricted the expansion up to polynomial 2 due to to enormous increase in the weight parameters.

In consequence, the force on atom :math:`j` and the stress matrix can be derived, respectively:

.. math::
    \boldsymbol{F}_m = -\sum^{N}_{i=1} \bigg(\boldsymbol{\gamma} \cdot \frac{\partial \boldsymbol{X}_i}{\partial \boldsymbol{r}_m} + \frac{1}{2} \bigg[\frac{\partial \boldsymbol{X}_i^T}{\partial \boldsymbol{r}_m} \cdot \boldsymbol{\Gamma} \cdot \boldsymbol{X}_i + \boldsymbol{X}_i^T \cdot \boldsymbol{\Gamma} \cdot \frac{\partial \boldsymbol{X}_i}{\partial \boldsymbol{r}_m} \bigg]\bigg)

.. math::
    \boldsymbol{S} = -\sum_{m=1}^N \boldsymbol{r}_m \otimes \sum^{N}_{i=1} \bigg(\boldsymbol{\gamma} \cdot \frac{\partial \boldsymbol{X}_i}{\partial \boldsymbol{r}_m} + \frac{1}{2} \bigg[\frac{\partial \boldsymbol{X}_i^T}{\partial \boldsymbol{r}_m} \cdot \boldsymbol{\Gamma} \cdot \boldsymbol{X}_i + \boldsymbol{X}_i^T \cdot \boldsymbol{\Gamma} \cdot \frac{\partial \boldsymbol{X}_i}{\partial \boldsymbol{r}_m} \bigg]\bigg)

Notice that the energy, force, and stress share the weights parameters :math:`\{\gamma_0, \boldsymbol{\gamma}_1, ..., \boldsymbol{\gamma}_N, \boldsymbol{\Gamma}_{11}, \boldsymbol{\Gamma}_{12}, ..., \boldsymbol{\Gamma}_{NN}\}`. Therefore, a reliable MLP must satisfy the three conditions in term of energy, force, and stress.

Neural Network Regression
^^^^^^^^^^^^^^^^^^^^^^^^^

Another type of regression model is neural network regression. Due to the set-up of the algorithm, neural network is suitable for training large data sets. Neural network gains an upper hand from generalized linear regression in term of the flexibility of the parameters.

A mathematical form to determine any node value can be written as:

.. math::
    X^{l}_{n_i} = a^{l}_{n_i}\bigg( b^{l-1}_{n_i} + \sum^{N}_{n_j=1} W^{l-1, l}_{n_j, n_i} \cdot X^{l-1}_{n_j} \bigg)

The value of a neuron (:math:`X_{n_i}^l`) at layer :math:`l` can determined by the relationships between the weights (:math:`W^{l-1, l}_{n_j, n_i}`), the bias (:math:`b^{l-1}_{n_i}`), and all neurons from the previous layer (:math:`X^{l-1}_{n_j}`). :math:`W^{l-1, l}_{n_j, n_i}` specifies the connectivity of neuron :math:`n_j` at layer :math:`l-1` to the neuron :math:`n_i` at layer :math:`l`. :math:`b^{l-1}_{n_i}` represents the bias of the previous layer that belongs to the neuron :math:`n_i`. These connectivity are summed based on the total number of neurons (:math:`N`) at layer :math:`l-1`. Finally, an activation function (:math:`a_{n_i}^l`) is applied to the summation to induce non-linearity to the neuron (:math:`X_{n_i}^l`). :math:`X_{n_i}` at the output layer is equivalent to an atomic energy, and it represents an atom-centered descriptor at the input layer. The collection of atomic energy contributions are summed to obtain the total energy of the structure.

.. [1] Albert P Bartok, Risi Kondor and Gabor Csanyi, “On representing chemical environments,” Phys. Rev. B 87, 184115 (2013)
.. [2] Jorg Behler and Michele Parrinello, “Generalized neural-network representation of high-dimensional potential-energy surfaces,” Phys. Rev. Lett. 98, 146401 (2007)
.. [3] Zhang, C. Hu, B. Jiang, "Embedded atom neural network potentials: Efficient and accurate machine learning with a physically inspired representation," The Journal of Physical Chemistry Letters 10 (17) (2019) 4962–4967 (2019).
.. [4] Albert P Bartok, Mike C Payne, Risi Kondor and Gabor Csanyi, “Gaussian approximation potentials: The accuracy of quantum mechan-ics, without the electrons,” Phys. Rev. Lett. 104, 136403 (2010)
.. [5] A.P. Thompson, L.P. Swiler, C.R. Trott, S.M. Foiles and G.J. Tucker, “Spectral neighbor analysis method for automated generation ofquantum-accurate interatomic potentials,” J. Comput. Phys. 285, 316–330 (2015)  
