from .model.snap import Snap

directory = "/datasets/Mo/training/"
files = ["AIMD_NPT.json"]
profile = dict(Mo=dict(r=0.5, w=1.0))


for file in files:
    with open(directory+file) as f:
        datas = json.load(f)

for struc in data:
    lat = struc['structure']['lattice']['matrix']
    species = []
    positions = []
    for site in struc['structure']['sites']:
        species.append(site['label'])
        positions.append(site['xyz'])
    fcoords = np.dot(positions, np.linalg.inv(lat))
    structure = Structure(lat, species, fcoords)
    structures.append(structure)


