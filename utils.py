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
