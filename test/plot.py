import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

bispectrums = pd.read_csv("test/bispectrum.csv")

energies = pd.read_csv("test/energy.csv")

y = energies.iloc[:,0].values
X = bispectrums.iloc[:294,:].values

X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.4, 
                                                    random_state=13)

lr = LinearRegression().fit(X_train, y_train)
y_pred = lr.predict(X_test)

mae_all = mean_absolute_error(y_test, y_pred)

mae = []
for i in range(1, 31):
    X_red = np.delete(X, i, axis=1)
    X_trainr, X_testr, y_trainr, y_testr = train_test_split(X_red, y, 
                                                            test_size=0.4, 
                                                            random_state=13)
    lr_red = LinearRegression().fit(X_trainr, y_trainr)
    y_predr = lr_red.predict(X_testr)
    mae.append(mean_absolute_error(y_testr, y_predr))
    

mae_total = mae - mae_all
x_axis = ['0-0-0', '1-0-1', '1-1-2', '2-0-2', '2-1-3', '2-2-2', '2-2-4', 
          '3-0-3', '3-1-4',	'3-2-3', '3-2-5', '3-3-4', '3-3-6',	'4-0-4', 
          '4-1-5', '4-2-4', '4-2-6', '4-3-5', '4-4-4', '4-4-6', '5-0-5',
          '5-1-6', '5-2-5', '5-3-6', '5-4-5', '5-5-6', '6-0-6', '6-2-6',
          '6-4-6', '6-6-6']
y_pos = np.arange(len(x_axis))


plt.bar(y_pos, mae_total)
plt.xticks(y_pos, x_axis, rotation=90)
plt.ylabel('Δmae(eV)')
plt.xlabel('j1-j2-jmax')

plt.savefig('mae.png', dpi=2000, bbox_inches='tight')
