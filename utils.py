import matplotlib.pyplot as plt
from pennylane import numpy as np


def distances_between_centers(centers):
    c_distances = np.zeros((len(centers), len(centers)))
    for i, v_1 in enumerate(centers.values()):
        for j, v_2 in enumerate(centers.values()):
            if i == j:
                continue
            c_distances[i, j] = np.linalg.norm(np.array(v_1) - np.array(v_2))
    return c_distances


def davies_bouldin_index(n, values, c_distances):
    intra_cls_distances = np.zeros((n))
    for i in range(n):
        rows = np.where(values[:, 0] == i)
        intra_cls_distances[i] = np.average(values[rows][:, -1])
    dbis = []
    for i in range(n):
        fractions = []
        for j in range(n):
            if i == j:
                continue
            d = c_distances[i, j]
            fractions.append((intra_cls_distances[i] +
                              intra_cls_distances[j]) / d)
        dbis.append(max(fractions))
    return float(np.average(dbis))


def plot_2d(classes, values, centers, step, show=False, save=True):
    colors = [("tomato",  "red"),
              ("deepskyblue", "blue"),
              ("chartreuse", "green"),
              ("gold", "orange"),
              ("violet", "fuchsia"),
              ]
    plt.rcParams["figure.figsize"] = (6, 6)

    for index, center in enumerate(centers.values()):
        rows = np.where(values[:, 0] == index)

        marker_color, center_color = colors.pop(0)
        plt.scatter(values[rows][:, 1], values[rows][:, 2],
                    color=marker_color, alpha=0.4, label=str(classes[index]))
        plt.plot(center[0], center[1], color=center_color,
                 alpha=0.9, ms=13, marker="*", markeredgecolor="black")

    plt.title("step " + str(step))
    plt.ylim(-1, 1)
    plt.xlim(-1, 1)
    plt.legend()
    if show:
        plt.show()
    if save:
        plt.savefig("./images/" + str(step) + ".png")
    plt.clf()


def plot_3d(classes, values, centers, step, show=False, save=True):
    colors = [("tomato",  "red"),
              ("deepskyblue", "blue"),
              ("chartreuse", "green"),
              ("gold", "orange"),
              ("violet", "fuchsia"),
              ]
    plt.rcParams["figure.figsize"] = (6, 6)

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    for index, center in enumerate(centers.values()):
        rows = np.where(values[:, 0] == index)

        marker_color, center_color = colors.pop(0)
        ax.scatter(values[rows][:, 1], values[rows][:, 2], values[rows][:, 3],
                    color=marker_color, alpha=0.4, label=str(classes[index]))
        ax.plot(center[0], center[1], center[2], color=center_color,
                 alpha=0.9, ms=13, marker="*", markeredgecolor="black")

    plt.title("step " + str(step))
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    plt.legend()
    if show:
        plt.show()
    if save:
        plt.savefig("./images/" + str(step) + ".png")
    plt.clf()
