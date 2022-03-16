import random
import pennylane as qml
import json
from pennylane import numpy as np
from time import localtime, strftime
from utils import cross_entropy_with_logits, labels_to_one_hot
from data import load_mnist
from data import load_breast_cancer_lju
from data import load_moons_dataset
from evaluation import evaluate_classical


with open('hyperparameters.json') as json_file:
    hp = json.load(json_file)
print(hp)

starting_time = strftime("%Y-%m-%d_%H-%M-%S", localtime())

np.random.seed(hp["seed"])
random.seed(hp["seed"])


def circuit(params, data):
    qml.templates.embeddings.AngleEmbedding(
            features=data, wires=range(hp["data_qubits"]), rotation="X"
        )

    for layer in range(hp["layers"]):
        for wire in range(hp["qubits"]):
            qml.RX(params[layer][wire][0], wires=wire)
            qml.RY(params[layer][wire][1], wires=wire)
        for wire in range(0, hp["qubits"] - 1, 2):
            qml.CZ(wires=[wire, wire + 1])
        for wire in range(1, hp["qubits"] - 1, 2):
            qml.CZ(wires=[wire, wire + 1])

    return [qml.probs(wires=x) for x in range(hp["output_qubits"])]


def loss(params, qNode, x, y):
    logits = qNode(params, x)[:, 1]
    ce_loss = cross_entropy_with_logits(predictions=logits, targets=y)
    return ce_loss


def train():
    assert(hp["dataset"] in ["mnist", "bc", "moons"])
    dev = qml.device('default.qubit', wires=hp["qubits"], shots=hp["shots"])
    qNode = qml.QNode(func=circuit, device=dev)

    stepsize = hp["start_stepsize"]
    optimizer = qml.AdamOptimizer(stepsize)

    def cost_fn(params):
        return loss(params, qNode, x, y)

    params = np.random.uniform(low=-np.pi, high=np.pi,
                               size=(hp["layers"], hp["qubits"], 2)
                               )

    if hp["dataset"] == "mnist":
        train_x, train_y, test_x, test_y = load_mnist(seed=hp["seed"],
                                                      train_size=hp["train_size"],
                                                      test_size=hp["test_size"],
                                                      classes=hp["classes"],
                                                      wires=hp["output_qubits"]
                                                      )
        train_y = train_y[:, :len(hp["classes"])]
        test_y = test_y[:, :len(hp["classes"])]
    elif hp["dataset"] == "bc":
        train_x, train_y, test_x, test_y = load_breast_cancer_lju(hp["train_size"],
                                                                  hp["test_size"]
                                                                  )
        train_y = labels_to_one_hot(train_y)
        test_y = labels_to_one_hot(test_y)
    elif hp["dataset"] == "moons":
        train_x, train_y, test_x, test_y = load_moons_dataset(hp["train_size"],
                                                              hp["test_size"]
                                                              )
        train_y = labels_to_one_hot(train_y)
        test_y = labels_to_one_hot(test_y)

    accuracys = []
    losses = []
    gradients = []

    step = 0
    while step < hp["steps"] + 1:
        for x, y in zip(train_x, train_y):
            params, c = optimizer.step_and_cost(cost_fn, params)
            print(f"step {step:{len(str(hp['steps']))}}| cost {c:8.5f}")

            losses.append(c)

            if step % hp["grads_every"] == 0:
                g, _ = optimizer.compute_grad(cost_fn, (params,), {}, None)
                gradients.append(np.var(g))
                print("Gradients", np.var(g))

            # if step % hp["test_every"] == 0:
            #     accuracy = evaluate_classical(test_x, test_y, qNode, params)
            #     accuracys.append((step, accuracy))
            #     print(f"Accuracy in step {step}: {accuracy}")
            #     print("Accuracys:\n", accuracys)

            step += 1
            if step >= hp["steps"] + 1:
                break

    if accuracys:
        print("Accuracys:\n", accuracys)
        print("Maximum: ", max(np.array(accuracys)[:, 1]))

    print("Gradients Avg: ", np.average(gradients))
    return np.average(gradients)
    # with open(f"./trainings/{starting_time}_classical.json", "w") as json_file:
    #     json.dump(hp, json_file)
    # np.savez(f"./trainings/{starting_time}_classical.npz",
    #          accuracys=accuracys,
    #          losses=losses,
    #          gradients=gradients,
    #          params=params
    #          )

if __name__ == "__main__":
    n_layers = [5, 10, 15, 20, 25, 30]
    n_qubits = [4, 6, 8, 10, 12]
    seeds = [1]

    gradients = np.zeros((len(n_layers), len(n_qubits), len(seeds)))

    for il, l in enumerate(n_layers):
        for iq, q in enumerate(n_qubits):
            for js, s in enumerate(seeds):
                hp["layers"] = l
                hp["qubits"] = q
                hp["seeds"] = s
                print(l, q, s)
                grad = train()
                gradients[il, iq, js] = grad

    np.savez(f"./trainings/{starting_time}_grads_cl.npz", grads=gradients,
                                                          n_layers=np.array(n_layers),
                                                          n_qubits=np.array(n_qubits),
                                                          seeds=np.array(seeds))
    print(gradients)

    # train()
