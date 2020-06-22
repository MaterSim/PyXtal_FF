from eamd import EAMD as new_EAMD
from old_eamd import EAMD as old_EAMD
from ase.build import bulk

a = 5.0
Rc1 = 3.5
params = {'L': 2, 'eta': [0.036, 0.071], 'Rs': [0., 0.5]}

si = bulk('Si', 'diamond', a=a, cubic=True)

old = old_EAMD(params, Rc1, derivative=True, stress=True).calculate(si)
new = new_EAMD(params, Rc1, derivative=True, stress=True).calculate(si)

a, b, c, d = old['dxdr'].shape
seq = new['seq']

count = 0
for i in range(a):
    for j in range(b):
        if ([i, j] == seq[count]).all():
            print([i, j], "   ", seq[count])
            assert (old['dxdr'][i, j] == new['dxdr'][count]).all(), f"old and new are not equal, {i, j}"
            count += 1
        else:
            assert (old['dxdr'][i, j] == 0.).all(), "Array is not zero."

print("NEW and OLD are the same")
