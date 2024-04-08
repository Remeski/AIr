import matplotlib
matplotlib.use("QtAgg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np


def plot_netw(netw, nrange=0, prange=7, precision=10000):
  if netw.schema["n_input"] != 1 or netw.schema["layers"][-1]["n"] != 1:
    print("Invalid network for 2d plotting")
    return
  xpoints = [x / precision for x in range(nrange*precision, prange*precision) ]
  ypoints = [netw.run(x)[0][0] for x in xpoints ]

  plt.plot(np.array(xpoints), np.array(ypoints))
  plt.show(block=False)


class Animator:
  def __init__(self):
    self.fig = plt.figure()
    self.ax = self.fig.add_subplot(111)
    self.losses = []

    self.ani = animation.FuncAnimation(self.fig, self.animate, frames=1000, interval=100)
    plt.show(block=False)


  def loss_point(self, netw):
    self.losses.append(netw.cur_loss)

  def animate(self, i):
    if len(self.losses) > 0:
      plt.ylim(0, self.losses[-1])
      plt.xlim(0, len(self.losses))
    self.ax.clear()
    self.ax.plot(self.losses)

  def stop(self):
    self.ani.pause()



