from __future__ import print_function
import matplotlib.pyplot as plt
import sys
import numpy as np
# from matplotlib.colors import

reload(sys)
sys.setdefaultencoding('utf8')


def plot(name_df, ids, X, Y, clusters, pops = None):
    # Get names for plotting
    name = []
    k = 0  # For logging
    for id in ids:
        name.append(str(name_df.loc[id][1]))
        k += 1
        if k % 1000 == 0:
            print("Found names of {}/{} vectors".format(k, len(ids)))

    # Plot svg

    # Helper function to make sure points are not on top of each other
    def rand_jitter(arr):
        stdev = .01 * (max(arr) - min(arr))
        return arr + np.random.randn(len(arr)) * stdev

    def jitter(x, y, marker='o', cmap=None, norm=None, vmin=None, vmax=None, alpha=None, linewidths=None,
               verts=None, hold=None, **kwargs):
        return plt.scatter(rand_jitter(x), rand_jitter(y), marker=marker, cmap=cmap, norm=norm, vmin=vmin,
                           vmax=vmax, alpha=alpha, linewidths=linewidths, verts=verts, hold=hold, **kwargs)

    col = plt.cm.get_cmap('tab20')
    colors = [col.colors[i] for i in clusters]

    jitter(X, Y, s=[2]*len(X), c=colors, alpha=0.5)

    for label, x, y in zip(name, X, Y):
        plt.text(x, y, label, horizontalalignment='center',
                 verticalalignment='bottom', fontsize=2, color='black')

    return plt
