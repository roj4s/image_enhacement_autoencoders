import pandas as pd
import sys
from matplotlib import pyplot as plt

log_file_addr = sys.argv[1]
d = pd.read_csv(log_file_addr)
d.plot(x='epoch', y=['training_loss', 'test_loss'])
plt.yscale('log')
plt.grid()
plt.show()
