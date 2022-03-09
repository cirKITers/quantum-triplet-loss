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


def cross_entropy_with_logits(predictions, targets, epsilon=1e-12):
    """
    Computes cross entropy between targets (encoded as one-hot vectors)
    and predictions. 
    Input: predictions (N, k) ndarray
           targets (N, k) ndarray        
    Returns: scalar
    """
    predictions = np.clip(predictions, epsilon, 1. - epsilon)
    predictions = np.exp(predictions)/sum(np.exp(predictions))
    N = predictions.shape[0]
    ce = -np.sum(targets*np.log(predictions+1e-9))/N
    return ce


def labels_to_one_hot(label: np.array):
    dummy = np.zeros((label.size, label.max()+1))
    dummy[np.arange(label.size), label] = 1
    return dummy