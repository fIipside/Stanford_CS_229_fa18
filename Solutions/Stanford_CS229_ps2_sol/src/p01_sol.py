import matplotlib.pyplot as plt
import numpy as np
import util as util

x_train_a, y_train_a = util.load_csv('../data/ds1_a.csv', add_intercept=True)
x_train_b, y_train_b = util.load_csv('../data/ds1_b.csv', add_intercept=True)

def plot(x, y, save_path):
    # Plot dataset
    plt.figure()
    plt.plot(x[y == 1, -2], x[y == 1, -1], 'bx', linewidth=2)
    plt.plot(x[y == -1, -2], x[y == -1, -1], 'go', linewidth=2)

    # Add labels and save to disk
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.savefig(save_path)

plot(x_train_a, y_train_a, "output\p01_a.png")
plot(x_train_b, y_train_b, "output\p01_b.png")