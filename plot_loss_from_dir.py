import os
import pandas as pd
from matplotlib import pyplot as plt
import sys

logs_dir = sys.argv[1]
loss = sys.argv[2]

loss_file = os.path.join(logs_dir, f"{loss}.csv")

if not os.path.exists(loss_file):
    print(f"File {loss_file} not found")

d = pd.read_csv(loss_file)
_max = d[loss].max()
d['norm'] = d[loss] / _max

d[loss].hist()
plt.xlabel(loss)
plt.show()

