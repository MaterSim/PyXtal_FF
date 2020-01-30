from distutils.core import setup
import setuptools  # noqa
from os import path

this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

exec(open('pyxtal_ff/version.py').read())

setup(
    name="pyxtal_ff",
    version=__version__,
    author="Qiang Zhu, Howard Yanxon, David Zagaceta",
    author_email="qiang.zhu@unlv.edu",
    description="Python code for force field training of crystals",
    long_description=long_description,
    long_description_content_type = 'text/markdown',
    url="https://github.com/qzhu2017/PyXtal-FF",
    packages=['pyxtal_ff', 
              'pyxtal_ff.datasets', 
              'pyxtal_ff.descriptors', 
              'pyxtal_ff.models', 
              'pyxtal_ff.models.optimizers', 
              'pyxtal_ff.utilities'],
    scripts=['scripts/pyxtal_ff'],
    package_data={'pyxtal_ff.datasets': ['*.json'],
                  'pyxtal_ff.descriptors': ['*.npy'],
                 },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        'numpy>=1.13.3', 
        'scipy>=1.1.0', 
        'matplotlib>=2.0.0',
        'numba>=0.44.1',
        'ase>=3.18.0'],
    python_requires='>=3.6.1',
    license='MIT',
)
