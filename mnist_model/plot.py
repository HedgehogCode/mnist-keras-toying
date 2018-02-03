import matplotlib.pyplot as plt
import numpy as np

def plot_some(imgs, labels, count = 4):
    fig = plt.figure()
    for i in range(count):
        ax = plt.subplot(1, count, i+1)
        ax.set_title("Label: %s" % int(labels[i]))
        ax.imshow(np.reshape(imgs[i], (28,28)))
    plt.show()
