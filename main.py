import random
import pennylane as qml
import matplotlib.pyplot as plt
from pennylane import numpy as np
from maskit.datasets import load_data

SEED = 1337
np.random.seed(SEED)
random.seed(SEED)


WIRES = 4
LAYERS = 5
STEPS = 5001
ALPHA = 0.5
TRAIN_SIZE = 2000
TEST_SIZE = 400
TEST_EVERY = 250
CLASSES = (2, 4, 7, 8)


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
    # print(a_value, p_value, n_value)
    return max(np.linalg.norm(a_value-p_value)**2 - np.linalg.norm(a_value - n_value)**2 + ALPHA, 0.0)


def train():
    dev = qml.device('default.qubit', wires=WIRES, shots=None)
    qNode = qml.QNode(func=circuit, device=dev)

    optimizer = qml.AdamOptimizer(stepsize=0.005)
    def cost_fn(params):
        return triplet_loss(params, qNode, anchor, positive, negative)
    params = np.random.uniform(low=-np.pi, high=np.pi, size=(LAYERS, WIRES, 2))

    data = load_data("mnist", shuffle=SEED, train_size=TRAIN_SIZE, test_size=TEST_SIZE, classes=CLASSES)

    images = {}
    for cls in range(len(CLASSES)):
        images[cls] = []

    for index, label in enumerate(data.train_target):
        images[np.argmax(label)].append(data.train_data[index])

    for s in range(STEPS):
        pos, neg = random.sample(range(len(CLASSES)), 2)
        
        anchor, positive = random.sample(images[pos], 2)
        negative = random.choice(images[neg])

        params, c = optimizer.step_and_cost(cost_fn, params)

        print(s, c)

        if s % TEST_EVERY == 0:
            plot_test_set(data, qNode, params, s)


def plot_test_set(data, qNode, params, s):
    colors = ["darkred", "blue", "green", "orange", "black", "lightblue", "lightgreen"]
    images = {}
    plt.rcParams["figure.figsize"] = (6,6)

    for cls in range(len(CLASSES)):
        images[cls] = []

    for label, datum in zip(data.test_target, data.test_data):
        values = qNode(params, datum)
        images[np.argmax(label)].append(values)

    for index, (cls, values) in enumerate(images.items()):
        x = []
        y = []
        for value in values:
            x.append(value[0])
            y.append(value[1])

        plt.scatter(x, y, color=colors.pop(0), alpha=0.6, label=str(CLASSES[index]))
    
    plt.title("step " + str(s))
    plt.ylim(-1, 1)
    plt.xlim(-1, 1)
    plt.legend()
    # plt.show()
    plt.savefig("./images/" + str(s) + ".png")
    plt.clf()


if __name__ == "__main__":
    train()