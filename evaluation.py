import numpy as np
from sklearn.svm import SVC
from utils import davies_bouldin_index, distances_between_centers
from plotting import plot_2d, plot_3d


def evaluate(dataset, train_x, train_y, test_x, test_y,
             qNode, params, step, classes, output_qubits,
             show=False, save=True, cont=True):

    svm = SVC(kernel="linear")
    if dataset == "mnist":
        clf = svm.fit([qNode(params, x) for x in train_x],
                    [np.argmax(y) for y in train_y]
                    )
        test_data = [qNode(params, x) for x in test_x]
        test_target = [np.argmax(y) for y in test_y]
        accuracy = clf.score(test_data, test_target)
    else: 
        clf = svm.fit([qNode(params, x) for x in train_x],
                  train_y
                  )
        test_data = [qNode(params, x) for x in test_x]
        accuracy = clf.score(test_data, test_y)
    
    print("Accuracy", accuracy)

    # this will store:
    # label (0 - len(CLASSES)), x_output, y_output, distance_to_center
    values = np.zeros((len(test_y), output_qubits+2))

    # store label and output
    for index, (label, datum) in enumerate(zip(test_y, test_x)):
        output = qNode(params, datum)
        values[index, 0] = np.argmax(label) if dataset == "mnist" \
                                            else label
        for i in range(len(output)):
            values[index, 1+i] = output[i]

    # calculate centers, key is label
    centers = {}
    for cls in range(len(classes)):
        rows = np.where(values[:, 0] == cls)
        center = np.average(values[rows][:, 1:(1+output_qubits)], axis=0)
        centers[cls] = center

    # calculate distance to center
    for i in range(len(test_y)):
        values[i, -1] = np.linalg.norm(values[i, 1:(1+output_qubits)]
                                       - centers[int(values[i, 0])])

    c_distances = distances_between_centers(centers)
    print("Distances between centers\n", c_distances)

    dbi = davies_bouldin_index(len(classes), values, c_distances)
    print("Davies Bouldin Index:", dbi)

    if show or save:
        if output_qubits == 2:
            plot_2d(classes, values, centers, step, clf,
                    accuracy, show, save, cont, dbi)
        elif output_qubits == 3:
            plot_3d(classes, values, centers, step, accuracy, show, save, dbi)

    return accuracy, dbi


def evaluate_classical(test_x, test_y, qNode, params):
    correct = 0
    for x, y in zip(test_x, test_y):
        prediction = qNode(params, x)
        if np.argmax(prediction[:, 1]) == np.argmax(y):
            correct += 1
    return correct/len(test_y)
