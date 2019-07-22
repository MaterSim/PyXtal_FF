from monty.serialization import loadfn
import json
from mapp.descriptors.fbehlerparrinello import BehlerParrinello
from mapp.models.fneuralnetwork import fNeuralNetwork

no_of_structures = 2

data = loadfn("mapp/datasets/Si/training.json")[:no_of_structures]

#descriptors = loadfn("BehlerParrinello.json")

features = {}
structures = [d['structure'] for d in data]
structures = []
for i, d in enumerate(data):
    structures.append(d['structure'])
    features[i] = {}
    features[i]['energy'] = d['outputs']['energy']/len(d['structure'])
    features[i]['forces'] = d['outputs']['forces']

symmetry = {'G2': {'eta': [0.036, 0.071,]},
            'G4': {'lambda': [-1, 1], 'zeta':[1], 'eta': [0.036, 0.071,]}}

descriptors = BehlerParrinello(symmetry, Rc=5.2, derivative=True)

model = fNeuralNetwork(elements=['Si'])
model.train(structures, descriptors=descriptors, features=features, save=True)
