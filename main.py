import random
import pennylane as qml
from pennylane import numpy as np
from maskit.datasets import load_data
from collections import Counter
from utils import davies_bouldin_index, distances_between_centers, plot_2d, plot_3d

SEED = 1337
np.random.seed(SEED)
random.seed(SEED)


WIRES = 4
LAYERS = 5
TRAIN_SIZE = 2000
TEST_SIZE = 400
CLASSES = (3, 4, 6)

STEPS = 5001
TEST_EVERY = 250

START_STEPSIZE = 0.01
UPDATE_SZ_EVERY = 2000
SZ_FACTOR = 0.5

START_ALPHA = 1.0
UPDATE_ALPHA_EVERY = 5002
ALPHA_FACTOR = 1

OUTPUT_QUBITS = 2


def circuit(params, data):
    qml.templates.embeddings.AngleEmbedding(
            features=data, wires=range(WIRES), rotation="X"
        )

    for layer in range(LAYERS):
        for wire in range(WIRES):
            qml.RX(params[layer][wire][0], wires=wire)
            qml.RY(params[layer][wire][1], wires=wire)
        for wire in range(0, WIRES - 1, 2):
            qml.CZ(wires=[wire, wire + 1])
        for wire in range(1, WIRES - 1, 2):
            qml.CZ(wires=[wire, wire + 1])
    # return [qml.expval(qml.PauliZ(i)) for i in [0, 1]]
    return qml.expval(qml.PauliZ(0) @ qml.PauliX(1)), qml.expval(qml.PauliZ(2) @ qml.PauliX(3))

def triplet_loss(params, qNode, anchor, positive, negative, alpha):
    a_value = qNode(params, anchor)
    p_value = qNode(params, positive)
    n_value = qNode(params, negative)
    # print(a_value, p_value, n_value)
    return max((np.linalg.norm(a_value-p_value)**2 -
               np.linalg.norm(a_value - n_value)**2 + alpha),
               0.0)


def train():
    dev = qml.device('default.qubit', wires=WIRES, shots=None)
    qNode = qml.QNode(func=circuit, device=dev)

    stepsize = START_STEPSIZE
    optimizer = qml.AdamOptimizer(stepsize)

    def cost_fn(params):
        return triplet_loss(params, qNode, anchor, positive, negative, alpha)

    params = np.random.uniform(low=-np.pi, high=np.pi, size=(LAYERS, WIRES, 2))
    alpha = START_ALPHA

    data = load_data("mnist", shuffle=SEED,
                     train_size=TRAIN_SIZE,
                     test_size=TEST_SIZE,
                     classes=CLASSES)
    occurences_train = [np.argmax(x) for x in data.train_target]
    occurences_test = [np.argmax(x) for x in data.test_target]
    print("Train", Counter(occurences_train))
    print("Test", Counter(occurences_test))

    images = {}
    for cls in range(len(CLASSES)):
        images[cls] = []

    for index, label in enumerate(data.train_target):
        images[np.argmax(label)].append(data.train_data[index])

    dbis = []

    for step in range(STEPS):
        # if step < 2000:
        #     pos, neg = np.random.choice(range(len(CLASSES)), size=2, replace=False, p=[0.45, 0.45, 0.1])
        # else:
        pos, neg = random.sample(range(len(CLASSES)), 2)
        
        anchor, positive = random.sample(images[int(pos)], 2)
        negative = random.choice(images[int(neg)])
            
        params, c = optimizer.step_and_cost(cost_fn, params)

        print(f"step {step:{len(str(STEPS))}}| cost {c:8.5f}")

        if step % TEST_EVERY == 0:
            dbi = evaluate(data, qNode, params, step)
            dbis.append(dbi)

        # if (step+1) % UPDATE_SZ_EVERY == 0:
        #     stepsize *= SZ_FACTOR
        #     optimizer.stepsize = stepsize
        #     print("Updated stepsize to", stepsize)

        # if (step+1) % UPDATE_ALPHA_EVERY == 0:
        #     alpha *= ALPHA_FACTOR
        #     print("Updated alpha to", alpha)

    print("DBIs:\n", dbis)
    print("Minimum:", min(dbis))


def evaluate(data, qNode, params, step, show=False, save=True):

    # this will store:
    # label (0 - len(CLASSES)), x_output, y_output, distance_to_center
    values = np.zeros((len(data.test_target), OUTPUT_QUBITS+2))

    # store label and output
    for index, (label, datum) in enumerate(zip(data.test_target, data.test_data)):
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
    for i in range(len(data.test_target)):
        values[i, -1] = np.linalg.norm(values[i, 1:(1+OUTPUT_QUBITS)] - centers[int(values[i, 0])])

    c_distances = distances_between_centers(centers)
    print("Distances between centers\n", c_distances)

    dbi = davies_bouldin_index(len(CLASSES), values, c_distances)
    print("Davies Bouldin Index:", dbi)

    if OUTPUT_QUBITS == 2:
        plot_2d(CLASSES, values, centers, step, show, save)
    elif OUTPUT_QUBITS == 3:
        plot_3d(CLASSES, values, centers, step, show, save)

    return dbi


if __name__ == "__main__":
    train()
