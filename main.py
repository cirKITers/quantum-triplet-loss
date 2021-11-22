import random
import pennylane as qml
from pennylane import numpy as np
from maskit.datasets import load_data


wires = 4
layers = 4
steps = 5000
alpha = 0.5

def circuit(params, data):
    qml.templates.embeddings.AngleEmbedding(
        features=data, wires=range(wires), rotation="X"
    )

    for layer in range(layers):
        for wire in range(wires):
            qml.RX(params[layer][wire][0], wires=wire)
            qml.RY(params[layer][wire][1], wires=wire)
        for wire in range(0, wires - 1, 2):
            qml.CZ(wires=[wire, wire + 1])
        for wire in range(1, wires - 1, 2):
            qml.CZ(wires=[wire, wire + 1])
    return qml.expval(qml.PauliZ(0))


def triplet_loss(params, qNode, anchor, positive, negative):
    a_value = qNode(params, anchor)
    p_value = qNode(params, positive)
    n_value = qNode(params, negative)

    return max((a_value - p_value)**2 - (a_value - n_value)**2 + alpha, 0.0)


def train():
    dev = qml.device('default.qubit', wires=wires, shots=1000)
    qNode = qml.QNode(func=circuit, device=dev)

    optimizer = qml.AdamOptimizer()
    def cost_fn(params):
        return triplet_loss(params, qNode, anchor, positive, negative)
    params = np.random.uniform(low=-np.pi, high=np.pi, size=(layers, wires, 2))

    data = load_data("mnist", shuffle=False, target_length=2, train_size=1000, test_size=500)
    data_six = []
    data_nine = []

    for index, label in enumerate(data.train_target):
        if label[0] == 1:
            data_six.append(data.train_data[index])
        else:
            data_nine.append(data.train_data[index])

    for s in range(steps):
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


    a, b, c, d = 0, 0, 0, 0

    for label, datum in zip(data.test_target, data.test_data):
        value = qNode(params, datum)
        print(label, value)
        if label[0] == 1 and value > 0:
            a += 1
        if label[0] == 1 and value < 0:
            b += 1
        if label[0] == 0 and value > 0:
            c += 1
        if label[0] == 0 and value < 0:
            d += 1


    print(a, b, c, d)

if __name__ == "__main__":
    train()