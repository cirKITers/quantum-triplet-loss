import random
import pennylane as qml
from pennylane import numpy as np
from sklearn.svm import SVC
from utils import davies_bouldin_index, distances_between_centers
from plotting import plot_2d, plot_3d, plot_curves
from data import load_mnist, mnist_apn_generator


SEED = 1337
np.random.seed(SEED)
random.seed(SEED)

QUBITS = 4
DATA_QUBITS = 4
OUTPUT_QUBITS = 2
LAYERS = 5

TRAIN_SIZE = 2000
TEST_SIZE = 400
CLASSES = (3, 6)

STEPS = 2501
TEST_EVERY = 250

START_STEPSIZE = 0.005
UPDATE_SZ_EVERY = 35000
SZ_FACTOR = 0.1

ALPHA = 1.0

SHOTS = None


def circuit(params, data):
    qml.templates.embeddings.AngleEmbedding(
            features=data, wires=range(DATA_QUBITS), rotation="X"
        )

    for layer in range(LAYERS):
        for wire in range(QUBITS):
            qml.RX(params[layer][wire][0], wires=wire)
            qml.RY(params[layer][wire][1], wires=wire)
        for wire in range(0, QUBITS - 1, 2):
            qml.CZ(wires=[wire, wire + 1])
        for wire in range(1, QUBITS - 1, 2):
            qml.CZ(wires=[wire, wire + 1])
    # return [qml.expval(qml.PauliZ(i)) for i in range(OUTPUT_QUBITS)]
    return [qml.expval(qml.PauliZ(2*x) @ qml.PauliZ((2*x)+1)) for x in range(OUTPUT_QUBITS)]


def triplet_loss(params, qNode, anchor, positive, negative, alpha):
    a_value = qNode(params, anchor)
    p_value = qNode(params, positive)
    n_value = qNode(params, negative)

    return max((np.linalg.norm(a_value-p_value)**2 -
               np.linalg.norm(a_value - n_value)**2 + alpha),
               0.0)


def train(dataset: str):
    assert(dataset in ["mnist", "bc"])
    dev = qml.device('default.qubit', wires=QUBITS, shots=SHOTS)
    qNode = qml.QNode(func=circuit, device=dev)

    stepsize = START_STEPSIZE
    optimizer = qml.AdamOptimizer(stepsize)

    def cost_fn(params):
        return triplet_loss(params, qNode, anchor, positive, negative, ALPHA)

    params = np.random.uniform(low=-np.pi, high=np.pi, size=(LAYERS, QUBITS, 2))

    if dataset == "mnist":
        train_x, train_y, test_x, test_y = load_mnist(seed=SEED,
                                                      train_size=TRAIN_SIZE,
                                                      test_size=TEST_SIZE,
                                                      classes=CLASSES,
                                                      wires=QUBITS,
                                                      )

        apn_generator = mnist_apn_generator(train_x,
                                            train_y,
                                            n_cls=len(CLASSES))

    accuracys = []
    dbis = []
    losses = []
    current_losses = []
    gradients = []

    for step in range(STEPS):
        anchor, positive, negative = next(apn_generator)

        params, c = optimizer.step_and_cost(cost_fn, params)

        print(f"step {step:{len(str(STEPS))}}| cost {c:8.5f}")

        current_losses.append(c)
        if len(current_losses) > 24:
            losses.append((step, np.average(current_losses)))
            current_losses = []

        # if step % 100 == 0:
        g, _ = optimizer.compute_grad(cost_fn, (params,), {}, None)
        gradients.append(np.var(g))
            # print("Gradients", np.var(g))

        if step % TEST_EVERY == 0:
            accuracy, dbi = evaluate(train_x, train_y, test_x, test_y, 
                                     qNode, params, step)
            accuracys.append((step, accuracy))
            dbis.append((step, dbi))
            print("Accuracys:\n", accuracys)

        # if (step+1) % UPDATE_SZ_EVERY == 0:
        #     stepsize *= SZ_FACTOR
        #     optimizer.stepsize = stepsize
        #     print("Updated stepsize to", stepsize)

    print("Accuracys:\n", accuracys)
    print("Maximum: ", max(accuracys))

    print("DBIs:\n", dbis)
    print("Minimum:", min(dbis))

    print("Gradients Avg: ", np.average(gradients))

    plot_curves(np.array(accuracys),
                np.array(dbis),
                np.array(losses),
                f"Qubits: {QUBITS}, Layers: {LAYERS}, Classes: {CLASSES}, Output_dim: {OUTPUT_QUBITS}"
                )


def evaluate(train_x, train_y, test_x, test_y, 
             qNode, params, step, show=False, save=True, cont=True):

    svm = SVC(kernel="linear")
    clf = svm.fit([qNode(params, x) for x in train_x],
                  [np.argmax(y) for y in train_y]
                  )
    test_data = [qNode(params, x) for x in test_x]
    test_target = [np.argmax(y) for y in test_y]
    accuracy = clf.score(test_data, test_target)

    print("Accuracy", accuracy)

    # this will store:
    # label (0 - len(CLASSES)), x_output, y_output, distance_to_center
    values = np.zeros((len(test_y), OUTPUT_QUBITS+2))

    # store label and output
    for index, (label, datum) in enumerate(zip(test_y, test_x)):
        output = qNode(params, datum)
        values[index, 0] = np.argmax(label)
        for i in range(len(output)):
            values[index, 1+i] = output[i]

    # calculate centers, key is label
    centers = {}
    for cls in range(len(CLASSES)):
        rows = np.where(values[:, 0] == cls)
        center = np.average(values[rows][:, 1:(1+OUTPUT_QUBITS)], axis=0)
        centers[cls] = center

    # calculate distance to center
    for i in range(len(test_y)):
        values[i, -1] = np.linalg.norm(values[i, 1:(1+OUTPUT_QUBITS)]
                                       - centers[int(values[i, 0])])

    c_distances = distances_between_centers(centers)
    print("Distances between centers\n", c_distances)

    dbi = davies_bouldin_index(len(CLASSES), values, c_distances)
    print("Davies Bouldin Index:", dbi)

    if OUTPUT_QUBITS == 2:
        plot_2d(CLASSES, values, centers, step, clf, accuracy, dbi, show, save, cont)
    elif OUTPUT_QUBITS == 3:
        plot_3d(CLASSES, values, centers, step, accuracy, dbi, show, save)

    return accuracy, dbi


if __name__ == "__main__":
    train("mnist")
