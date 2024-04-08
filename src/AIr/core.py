import numpy as np
from . import activations
import math
import time

np.random.seed(0)

class Layer:
  def __init__(self, n_input, n_output, activation="ReLU", weights=None, biases=None):
    self.n_input = n_input
    self.n_output = n_output

    self.activation = activations.name_to_class[activation]()
    self.activation_name = activation

    if (weights is not None) and (np.array(weights).shape == (n_input, n_output)):
      self.weights = np.array(weights)
    else:
      self.weights = np.random.randn(n_input, n_output)

    if (biases is not None) and (np.array(biases).shape == (1, n_output)):
      self.biases = np.array(biases)
    else:
      self.biases = np.zeros((1,n_output))

  def forward(self, input):
    self.Z = np.dot(np.array(input), np.array(self.weights)) + np.array(self.biases) 
    self.output = self.activation.calc(self.Z)

# schema: { n_input: INPUT_NODES, 
#           layers: [ { n: N_NODES, 
#             activation: ACTIVATION_FUNCTION_NAME,
#           }, ...],
#         } 
# The last layer in layers will be the output layer
class NeuralNetwork:
  def __init__(self, schema, layers=None):
    self.schema = schema
    self.n_input = schema["n_input"]
    self.layers_schema = schema["layers"]
    if layers is not None and self.validate_layers(schema, layers):
      self.layers = layers
    else:
      self.layers = []
      for i,l in enumerate(self.layers_schema):
        input = self.n_input
        if i > 0:
          input = self.layers[i-1].n_output
        layer = Layer(input, l["n"], activation=l.get("activation"))
        self.layers.append(layer)

  def validate_layers(self, schema, layers):
    schema_layers = schema["layers"]
    if len(layers) != len(schema_layers):
      return False
    for ls, l in zip(schema_layers, layers): 
      if l.n_output != ls["n"]:
        return False
      if l.activation_name != ls.get("activation"):
        return False
    return True

  def forward(self, input):
    self.input = input
    output = input
    for l in self.layers:
      l.forward(output)
      output = l.output
    self.output = output

  def backprop(self, Y):
    dW = []
    dB = []

    M = len(Y)

    for l in self.layers:
      dW.append(np.zeros(l.weights.shape))
      dB.append(np.zeros(l.weights.shape))

    ls = self.layers

    delta = self.loss_prime(Y) * ls[-1].activation.prime(ls[-1].Z)
    dB[-1] = 1/M * np.sum(delta, axis=0)
    dW[-1] = 1/M * np.dot(ls[-2].output.T, delta)

    for i in range(2, len(self.layers)):
      delta = ls[-i].activation.prime(ls[-i].Z) * np.dot(delta, ls[-i+1].weights.T)
      dB[-i] = 1/M * np.sum(delta, axis=0)
      dW[-i] = 1/M * np.dot(ls[-i-1].output.T, delta) 

    delta = ls[0].activation.prime(ls[0].Z) * np.dot(delta, ls[1].weights.T)
    dB[0] = 1/M * np.sum(delta, axis=0)
    dW[0] = 1/M * np.dot(np.array(self.input).T, delta)

    return dW, dB

  def loss(self, Y):
    return 1/(2*len(Y)) * np.sum((self.output-Y)**2)

  def loss_prime(self,Y):
    return 1/len(Y) * (self.output-Y)

  # Batch is a tuple ([X], [Y]) where [X] is a matrix thats rows are one set of inputs
  #                                   [Y] is a matrix thats rows are the sets expected output 
  # If mini_batch_size is not specified, it will use batch GD.
  def train(self, batch, debug=False, silent=False, eta=0.1, gamma=0, mini_batch_size=0, epoch=1, fn=lambda self: None):
    X,Y = batch

    batch_size = len(Y)

    if debug:
      print("Starting things")
      print("X:")
      print(X)
      print("Y:")
      print(Y)
      print("length:")
      print(batch_size)

    iterations = epoch

    if mini_batch_size > 0:
      iterations = epoch * (batch_size // mini_batch_size) 

    self.prev_dW = np.zeros(len(self.layers))
    self.prev_dB = np.zeros(len(self.layers))

    if iterations < 1:
      raise Exception(f"Invalid options, iterations: {iterations}")

    for i in range(iterations):
      if mini_batch_size > 0:
        start_index = i % (batch_size // mini_batch_size) * mini_batch_size
        end_index = start_index + mini_batch_size
        X,Y = (batch[0][start_index:end_index], batch[1][start_index:end_index])

      if debug:
        print("X:")
        print(X)
        print("Y:")
        print(Y)

      self.forward(X)

      dW, dB = self.backprop(Y)

      if debug:
        print("dW:")
        print(dW)
        print("dB:")
        print(dB)

      _prev_dW = []
      _prev_dB = []

      for l, dw, db, dw2, db2 in zip(self.layers, dW, dB, self.prev_dW, self.prev_dB):
        w_inc = -(eta * dw + gamma * dw2)
        b_inc = -(eta * db + gamma * db2)

        _prev_dW.append(w_inc)
        _prev_dB.append(b_inc)

        l.weights += w_inc
        l.biases += b_inc

      self.prev_dW = _prev_dW
      self.prev_dB = _prev_dB

      fn(self)

      self.cur_loss = self.loss(Y)

      if debug:
        print("loss:")
        print(self.cur_loss)
        time.sleep(2)

      elif not silent:
        print(f"Done: {math.floor(round((i+1)/iterations, 2)*100)}%, loss at {round(self.cur_loss,9)}", end="\r" if i != iterations-1 else "\n")

  def run(self, input):
    self.forward(input)
    return self.output
  
  def test_loss(self, batch):
    self.forward(batch[0])
    return self.loss(batch[1])

  def store(self, file_path: str):
    file_path = file_path.rstrip(".npz")
    store_arrays = []
    for l in self.layers:
      store_arrays.append(l.weights)
      store_arrays.append(l.biases)
    np.savez(file_path, schema=self.schema, *store_arrays)


  @staticmethod
  def load_from_file(file_path: str):
    file_path = file_path if file_path.rfind(".npz") != -1 else file_path + ".npz"
    npz = np.load(file_path, allow_pickle=True)
    schema = npz["schema"][()]
    layers_schema = schema["layers"]

    layers = []
    if (len(npz.files) - 1) % 2 != 0:
      print("Loaded values seem to me corrupted")

    for i,l in enumerate(layers_schema):
      input = schema["n_input"]
      if i > 0:
        input = layers[i-1].n_output
      layer = Layer(input, l.get("n"), activation=l.get("activation"), weights=npz["arr_" + str(2*i)], biases=npz["arr_" + str(2*i+1)])
      layers.append(layer)

    return NeuralNetwork(schema, layers)
