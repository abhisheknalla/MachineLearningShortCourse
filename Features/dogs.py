import numpy as np
import matplotlib.pyplot as plt

greyhounds = 500
labradors = 500
pugs = 500

grey_height = 28 + 4 * np.random.randn(greyhounds)
lab_height = 24 + 4 * np.random.randn(labradors)
pug_height = 12 + 4 * np.random.randn(pugs)

plt.hist([grey_height, lab_height, pug_height], stacked =True, color=['g','b','y'], rwidth=0.5)
plt.xlabel('height')
plt.ylabel('# of dogs')
plt.show()
