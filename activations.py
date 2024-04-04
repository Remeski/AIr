import numpy as np

class ReLU:
  def prime(self, X):
    return (X > 0) * 1

  def calc(self, X):
    return np.maximum(0, X)

class LeakyReLU:
  def prime(self, X):
    dx = np.ones_like(X)
    dx[X < 0] = 0.01
    return dx
    # return (X > 0) * 1

  def calc(self, X):
    return np.maximum(0.01*X, X)

class Linear:
  def prime(self, X):
    return np.ones(X.shape)

  def calc(self, X):
    return X

class Sigmoid:
  def prime(self, X):
    return self.calc(X)**2*np.exp(-np.array(X))

  def calc(self, X):
    return 1 / (1 + np.exp(-np.array(X)))

name_to_class = {
  "ReLU": ReLU,
  "LeakyReLU": LeakyReLU,
  "Linear": Linear,
  "Sigmoid": Sigmoid
}
