import os
import random
import time
import numpy as np
from .core import NeuralNetwork

# TODO: come up with a more clever way to do this
def split_dataset(dataset, split):
  test_X = []
  test_Y = []
  X = []
  Y = []
  for x,y in zip(dataset[0], dataset[1]):
    if random.random() > split:
      test_X.append(x)
      test_Y.append(y)
      continue
    X.append(x)
    Y.append(y)
  return ((X,Y), (test_X, test_Y))

def convert_to_stamp(input: int, n=3):
  if len(str(input)) > n:
      return input
  return (n-len(str(input)))*"0"+str(input)

class Trainer:
  def __init__(self, schema=None, file_path=None):
    if file_path is not None and os.path.isfile(file_path if file_path.rfind(".npz") != -1 else file_path + ".npz"):
      self.file_path = file_path.replace(".npz", "")
      self.network = NeuralNetwork.load_from_file(file_path)
    elif schema is None:
      print("Specify schema or file_path")
      return
    else:
      self.file_path = file_path + "-000" if file_path is not None else "model-000"
      self.network = NeuralNetwork(schema)
  

  def calculate_eta(self):
    eta = self.eta if self.eta is not None else 0.1

    if self.loss < 10**(-3):
      eta *= 1/2

    return eta 

  def calculate_gamma(self):
    return 0 if self.gamma is None else self.gamma

  def calculate_epoch(self):
    return 1000 if self.epoch is None else self.epoch

  # dataset = (X, Y), where X is list of input and Y is expected output
  # split = determines how much (roughly) of dataset is used for training
  # epoch = number of iterations over whole dataset
  # eta = learning rate, if None => automatic learning (online)
  # gamma = momentum, if None => automatic
  # mini_batch_size = if None => automatic
  # noise = added randomness to dataset
  def train(self, dataset, split=0.9, debug=False, silent=False, epoch=None, eta=None, gamma=None, mini_batch_size=0, noise=0.0):
    self.eta = eta
    self.gamma = gamma
    self.epoch = epoch

    training_batch, test_batch = split_dataset(dataset, split)

    if len(training_batch[0]) < 1:
      print("No training samples")
      return

    self.loss = 100

    try:
        self.training_count = int(self.file_path[-3:]) 
    except:
        self.training_count = 0

    try:
      print(f"[train-{convert_to_stamp(self.training_count)}] Training with dataset size {len(dataset[0])}, training batch size {len(training_batch[0])}, test batch size {len(test_batch[0])}") 
      while True:
        self.eta = self.calculate_eta()
        self.gamma = self.calculate_gamma()
        self.epoch = self.calculate_epoch()

        k = 3

        self.training_count += 1

        print(f"[train-{convert_to_stamp(self.training_count)}] Starting with mini batch size {mini_batch_size}, eta {self.eta}, gamma {self.gamma} and epoch {self.epoch}") 

        self.network.train((np.array(training_batch[0])+noise*np.random.choice([-1, 1], size=np.array(training_batch[0]).shape)*np.random.rand(*np.array(training_batch[0]).shape), training_batch[1]),
                           epoch=self.epoch,
                           eta=self.eta,
                           gamma=self.gamma,
                           mini_batch_size=mini_batch_size,
                           silent=silent,
                           debug=debug)

        print(f"[train-{convert_to_stamp(self.training_count)}] Finished with loss {self.network.cur_loss}")

        self.loss = self.network.cur_loss

        if len(test_batch) > 0:
          print(f"[perf-{convert_to_stamp(self.training_count)}] Evaluating")
          self.loss = self.network.test_loss(test_batch)
          print(f"[perf-{convert_to_stamp(self.training_count)}] loss is {self.loss}")
          if self.loss < 10**(-k):
            c = input(f"[perf-{convert_to_stamp(self.training_count)}] Do you want to continue? Y/n")
            if c.lower() != "n":
              k += 3
            else:
              break
        if debug:
          time.sleep(2)

    except KeyboardInterrupt:
      print("Stopping")

    print(f"Finished training session")

    self.network.store(file_path=self.file_path[:-4] + f"-{convert_to_stamp(self.training_count)}")

