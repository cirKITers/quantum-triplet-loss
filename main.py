import random
import pennylane as qml
import matplotlib.pyplot as plt
from pennylane import numpy as np
from maskit.datasets import load_data


np.random.seed(1337)
random.seed(1337)


WIRES = 4
LAYERS = 4
STEPS = 2001
ALPHA = 1
TRAIN_SIZE = 1000
TEST_SIZE = 200


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
    return qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1))


def triplet_loss(params, qNode, anchor, positive, negative):
    a_value = qNode(params, anchor)
    p_value = qNode(params, positive)
    n_value = qNode(params, negative)

    return max(np.linalg.norm(a_value-p_value)**2 - np.linalg.norm(a_value - n_value)**2 + ALPHA, 0.0)


def train():
    dev = qml.device('default.qubit', wires=WIRES, shots=1000)
    qNode = qml.QNode(func=circuit, device=dev)

    optimizer = qml.AdamOptimizer(stepsize=0.001)
    def cost_fn(params):
        return triplet_loss(params, qNode, anchor, positive, negative)
    params = np.random.uniform(low=-np.pi, high=np.pi, size=(LAYERS, WIRES, 2))

    data = load_data("mnist", shuffle=False, target_length=2, train_size=TRAIN_SIZE, test_size=TEST_SIZE, classes=(6, 9))
    data_six = []
    data_nine = []

    for index, label in enumerate(data.train_target):
        if label[0] == 1:
            data_six.append(data.train_data[index])
        else:
            data_nine.append(data.train_data[index])

    for s in range(STEPS):
        if s % 2 == 0:
            anchor = random.choice(data_six)
            positive = random.choice(data_six)
            negative = random.choice(data_nine)
        else:
            anchor = random.choice(data_nine)
            positive = random.choice(data_nine)
            negative = random.choice(data_six)

        params, c = optimizer.step_and_cost(cost_fn, params)

        print(s, c)

        if s % 100 == 0:
            plot_test_set(data, qNode, params, s)


def plot_test_set(data, qNode, params, s):
    x_six = []
    y_six = []
    x_nine = []
    y_nine = []
    for label, datum in zip(data.test_target, data.test_data):
        values = qNode(params, datum)
        if label[0] == 1:
            x_six.append(values[0])
            y_six.append(values[1])
        else:
            x_nine.append(values[0])
            y_nine.append(values[1])

    plt.rcParams["figure.figsize"] = (6,6)
    plt.scatter(x_six, y_six, color="red")
    plt.scatter(x_nine, y_nine, color="blue")
    plt.title("step " + str(s))
    plt.ylim(-1, 1)
    plt.xlim(-1, 1)
    plt.savefig("./images/" + str(s) + ".png")
    # plt.show()
    plt.clf()


if __name__ == "__main__":
    train()