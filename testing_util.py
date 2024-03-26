import matplotlib
matplotlib.use("WebAgg")
import matplotlib.pyplot as plt
import numpy as np

import tests.AI as testing
import core as AI

def plot_netw(netw, nrange=0, prange=20, precision=1000):
  if netw.schema["n_input"] != 1 or netw.schema["layers"][-1]["n"] != 1:
    print("Invalid network for 2d plotting")
    return
  xpoints = [x / precision for x in range(nrange*precision, prange*precision) ]
  ypoints = [netw.run(x)[0][0] for x in xpoints ]

  plt.plot(np.array(xpoints), np.array(ypoints))
  plt.show()
