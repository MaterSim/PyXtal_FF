<!--- [![Build Status](https://travis-ci.org/qzhu2017/PyXtal_FF.svg?branch=master)](https://travis-ci.org/qzhu2017/PyXtal_FF) --->
[![Test Status](https://github.com/qzhu2017/PyXtal_FF/workflows/tests/badge.svg)](https://github.com/qzhu2017/PyXtal_FF/actions)
[![Documentation Status](https://readthedocs.org/projects/pyxtal-ff/badge/?version=latest)](https://pyxtal-ff.readthedocs.io/en/latest/?badge=latest)
[![PyPI version](https://badge.fury.io/py/pyxtal-ff.svg)](https://badge.fury.io/py/pyxtal-ff)
[![Downloads](https://pepy.tech/badge/pyxtal-ff)](https://pepy.tech/project/pyxtal-ff)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.3839987.svg)](https://doi.org/10.5281/zenodo.3839987)




         ______       _    _          _         _______ _______ 
        (_____ \     \ \  / /        | |       (_______|_______)
         _____) )   _ \ \/ / |_  ____| |        _____   _____   
        |  ____/ | | | )  (|  _)/ _  | |       |  ___) |  ___)  
        | |    | |_| |/ /\ \ |_( ( | | |_______| |     | |      
        |_|     \__  /_/  \_\___)_||_|_(_______)_|     |_|      
               (____/  
               
A Python package for Machine learning of interatomic force field.
PyXtal FF is an open-source Python library for developing machine learning interatomic potential of materials. 

The aim of PyXtal\_FF is to promote the application of atomistic simulations by providing several choices of structural descriptors and machine learning regressions in one platform. Based on the given choice of structural descriptors including 
- atom-centered symmetry functions 
- embedded atom density
- SNAP
- SO4 bispectrum
- SO3 power spectrum 

PyXtal\_FF can train the MLPs with either the linear regression or neural networks model, by simultaneously minimizing the errors of energy/forces/stress tensors in comparison with the data from the ab-initio simulation.

See the [documentation page](https://pyxtal-ff.readthedocs.io/en/latest/) for more background materials.

One can also quickly checkout the [example](https://github.com/qzhu2017/PyXtal_FF/tree/master/examples) section to see how to train and apply the force fields for productive simulations.

**This is an ongoing project.**

## Relevant works

[1]. Yanxon H, Zagaceta D, Tang B, Matteson D, Zhu Q* (2020)\
[PyXtal\_FF: a Python Library for Automated Force Field Generation](http://arxiv.org/abs/2007.13012)

[2]. Zagaceta D, Yanxon H, Zhu Q* (2020) \
[Spectral Neural Network Potentials for Binary Alloys](http://arxiv.org/abs/2005.04332)

[3]. Yanxon H, Zagaceta D, Wood B, Zhu Q* (2019) \
[On Transferability of Machine Learning Force Fields: A Case Study on Silicon](https://arxiv.org/pdf/2001.00972.pdf)

[4]. Fredericks S, Sayre D, Zhu Q *(2019) \
[PyXtal: a Python Library for Crystal Structure Generation and Symmetry Analysis](https://arxiv.org/pdf/1911.11123.pdf)
