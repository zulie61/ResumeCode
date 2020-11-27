import numpy as np
import matplotlib.pylab as plt

def sigmoid_function(x):
	return 1/(1+np.exp(-x))
x = np.linspace(-5, 5)
y = sigmoid_function(x)

plt.plot(x, y)
plt.show()