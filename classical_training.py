import random
import pennylane as qml
from pennylane import numpy as np
from utils import cross_entropy_with_logits, labels_to_one_hot
from data import load_mnist
from data import load_breast_cancer_lju
from data import load_moons_dataset
from evaluation import evaluate_classical


SEED = 1337
np.random.seed(SEED)
random.seed(SEED)

QUBITS = 9
DATA_QUBITS = 9
CLASSES = (3, 6)
OUTPUT_QUBITS = len(CLASSES)
LAYERS = 5

TRAIN_SIZE = 150
TEST_SIZE = 100

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


def train(dataset="minst"):
    assert(dataset in ["mnist", "bc", "moons"])
    dev = qml.device('default.qubit', wires=QUBITS, shots=SHOTS)
    qNode = qml.QNode(func=circuit, device=dev)

    stepsize = START_STEPSIZE
    optimizer = qml.AdamOptimizer(stepsize)

    def cost_fn(params):
        return loss(params, qNode, x, y)

    params = np.random.uniform(low=-np.pi, high=np.pi,
                               size=(LAYERS, QUBITS, 2)
                               )

    if dataset == "mnist":
        train_x, train_y, test_x, test_y = load_mnist(seed=SEED,
                                                      train_size=TRAIN_SIZE,
                                                      test_size=TEST_SIZE,
                                                      classes=CLASSES,
                                                      wires=QUBITS
                                                      )
        train_y = train_y[:, :len(CLASSES)]
        test_y = test_y[:, :len(CLASSES)]
    elif dataset == "bc":
        train_x, train_y, test_x, test_y = load_breast_cancer_lju(TRAIN_SIZE,
                                                                  TEST_SIZE
                                                                  )
        train_y = labels_to_one_hot(train_y)
        test_y = labels_to_one_hot(test_y)
    elif dataset == "moons":
        train_x, train_y, test_x, test_y = load_moons_dataset(TRAIN_SIZE,
                                                              TEST_SIZE
                                                              )
        train_y = labels_to_one_hot(train_y)
        test_y = labels_to_one_hot(test_y)

    accuracys = []
    losses = []
    current_losses = []
    gradients = []

    step = 0
    while step < STEPS:
        for x, y in zip(train_x, train_y):
            params, c = optimizer.step_and_cost(cost_fn, params)
            print(f"step {step:{len(str(STEPS))}}| cost {c:8.5f}")

            current_losses.append(c)
            if len(current_losses) > 24:
                losses.append((step, np.average(current_losses)))
                current_losses = []

            if step % 100 == 0:
                g, _ = optimizer.compute_grad(cost_fn, (params,), {}, None)
                gradients.append(np.var(g))
                print("Gradients", np.var(g))

            if step % TEST_EVERY == 0:
                accuracy = evaluate_classical(test_x, test_y, qNode, params)
                accuracys.append((step, accuracy))
                print(f"Accuracy in step {step}: {accuracy}")
                print("Accuracys:\n", accuracys)

            step += 1
            if step >= STEPS:
                break

    print("Accuracys:\n", accuracys)
    print("Maximum: ", max(np.array(accuracys)[:, 1]))

    print("Gradients Avg: ", np.average(gradients))


if __name__ == "__main__":
    train("bc")
