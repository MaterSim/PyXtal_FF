# PyXtal Silicon

These DFT-quality data sets are randomly generated based on symmetry constraints with [PyXtal](https://github.com/qzhu2017/PyXtal) (an open source Python package). The data sets are presented in JSON format.

The main data sets are Si4.json, Si6.json, Si8.json, and Si16.json. The integers suggest the total atoms in a unit cell. 

Si.json is a compilation of Si4.json, Si6.json, Si8.json, and Si16.json. Si_train.json consists of structures with <= -4 eV in Si.json. Si_train.json has 5352 structures, while Si.json has 6887 structures.

Phase diagram of Si_train.json:
<p align="center">
  <img width="600" height="450" src="https://github.com/qzhu2017/FF-project/blob/master/pyxtal_ff/datasets/Si/PyXtal/PES.png">
</p>
