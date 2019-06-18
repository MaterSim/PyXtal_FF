import json

from mapp.models.snap import Snap

dataset = "mapp/datasets/Si/Si4.json"

with open(dataset) as f:
    dataset = json.load(f)

print(len(dataset))

