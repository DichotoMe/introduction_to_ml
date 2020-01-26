import matplotlib.pyplot as plt

from mglearn.datasets import make_wave

X, y = make_wave(n_samples=40)
plt.plot(X, y, 'o')
plt.ylim(-3, 3)
plt.xlabel("Feature")
plt.ylabel("Target")

plt.show()