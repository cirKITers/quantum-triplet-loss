import random
import pennylane as qml
from pennylane import numpy as np
from maskit.datasets import load_data
from collections import Counter
from utils import cross_entropy_with_logits


SEED = 1337
np.random.seed(SEED)
random.seed(SEED)

QUBITS = 4
DATA_QUBITS = 4
CLASSES = (3, 6)
OUTPUT_QUBITS = len(CLASSES)
LAYERS = 20

TRAIN_SIZE = 2000
TEST_SIZE = 400

STEPS = 1001
TEST_EVERY = 250

START_STEPSIZE = 0.005
UPDATE_SZ_EVERY = 35000
SZ_FACTOR = 0.1

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

    return [qml.probs(wires=x) for x in range(OUTPUT_QUBITS)]


def loss(params, qNode, x, y):
    logits = qNode(params, x)[:, 1]
    ce_loss = cross_entropy_with_logits(predictions=logits, targets=y)
    return ce_loss


def train():
    dev = qml.device('default.qubit', wires=QUBITS, shots=SHOTS)
    qNode = qml.QNode(func=circuit, device=dev)

    stepsize = START_STEPSIZE
    optimizer = qml.AdamOptimizer(stepsize)

    def cost_fn(params):
        return loss(params, qNode, x, y)

    params = np.random.uniform(low=-np.pi, high=np.pi, size=(LAYERS, QUBITS, 2))

    data = load_data("mnist", shuffle=SEED,
                     train_size=TRAIN_SIZE,
                     test_size=TEST_SIZE,
                     classes=CLASSES,
                     wires=DATA_QUBITS)
    occurences_train = [np.argmax(x) for x in data.train_target]
    occurences_test = [np.argmax(x) for x in data.test_target]
    print("Train", Counter(occurences_train))
    print("Test", Counter(occurences_test))

    accuracys = []
    losses = []
    current_losses = []
    gradients = []

    step = 0
    while step < STEPS:
        for x, y in zip(data.train_data, data.train_target[:, :len(CLASSES)]):
            params, c = optimizer.step_and_cost(cost_fn, params)
            print(f"step {step:{len(str(STEPS))}}| cost {c:8.5f}")

            current_losses.append(c)
            if len(current_losses) > 24:
                losses.append((step, np.average(current_losses)))
                current_losses = []

            # if step % 100 == 0:
            g, _ = optimizer.compute_grad(cost_fn, (params,), {}, None)
            gradients.append(np.var(g))
            print("Gradients", np.var(g))

            if step % TEST_EVERY == 0:
                accuracy = evaluate(data, qNode, params, step)
                accuracys.append((step, accuracy))
                print(f"Accuracy in step {step}: {accuracy}")
                print("Accuracys:\n", accuracys)

            # if (step+1) % UPDATE_SZ_EVERY == 0:
            #     stepsize *= SZ_FACTOR
            #     optimizer.stepsize = stepsize
            #     print("Updated stepsize to", stepsize)

            step += 1
            if step >= STEPS:
                break

    print("Accuracys:\n", accuracys)
    print("Maximum: ", max(accuracys))

    print("Gradients Avg: ", np.average(gradients))


def evaluate(data, qNode, params):
    correct = 0
    for x, y in zip(data.test_data, data.test_target):
        prediction = qNode(params, x)
        if np.argmax(prediction[:, 1]) == np.argmax(y):
            correct += 1
    return correct/len(data.test_data)


if __name__ == "__main__":
    train()
