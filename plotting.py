import matplotlib.pyplot as plt
from pennylane import numpy as np


GG = 100


def plot_2d(classes, values, centers, step, clf, accuracy, dbi, show=False, save=True, cont=True):
    colors = [("tomato",  "red"),
              ("deepskyblue", "blue"),
              ("chartreuse", "green"),
              ("gold", "orange"),
              ("violet", "fuchsia"),
              ]
    plt.figure(figsize=(6,6))

    for index, center in enumerate(centers.values()):
        rows = np.where(values[:, 0] == index)

        marker_color, center_color = colors.pop(0)
        plt.scatter(values[rows][:, 1], values[rows][:, 2],
                    color=marker_color, alpha=0.4, label=str(classes[index]))
        plt.plot(center[0], center[1], color=center_color,
                 alpha=0.9, ms=13, marker="*", markeredgecolor="black")

    if cont:
        grid = np.linspace(-1, 1, GG)
        xx, yy = np.meshgrid(grid, grid)

        zz = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        zz = zz.reshape(xx.shape)

        ax = plt.gca()
        ax.contour(xx, yy, zz, len(classes)-2, colors='black')

    plt.title(f"Step: {step} | Acc: {accuracy:.3f} | DBI: {dbi:.3f}")
    plt.ylim(-1, 1)
    plt.xlim(-1, 1)
    plt.legend()

    if show:
        plt.show()
    if save:
        plt.savefig("./images/" + str(step) + ".png")
    plt.close()


def plot_3d(classes, values, centers, step, accuracy, dbi, show=False, save=True):
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

    plt.title(f"Step: {step} | Acc: {accuracy:.3f} | DBI: {dbi:.3f}")
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    plt.legend()
    if show:
        plt.show()
    if save:
        plt.savefig("./images/" + str(step) + ".png")
    plt.close()


def plot_curves(accuracys, dbis, loss, title):
    plt.figure(figsize=(12,8))
    plt.plot(accuracys[:, 0], accuracys[:, 1], label="Accuracy")
    plt.plot(dbis[:, 0], dbis[:, 1], label="Davis Bouldin Index")
    plt.plot(loss[:, 0], loss[:, 1], label="Loss")
    plt.title(title)
    plt.ylim(ymax = 3)
    plt.legend()
    plt.show()
