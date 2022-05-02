import random
import pennylane as qml
import json
from pennylane import numpy as np
from time import localtime, strftime
from plotting import plot_curves
from data import load_mnist, mnist_apn_generator
from data import load_mnist_ae
from data import load_breast_cancer_lju, bc_apn_generator
from data import load_moons_dataset, moons_apn_generator
from evaluation import evaluate


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
    # return [qml.expval(qml.PauliZ(i)) for i in range(hp["output_qubits"])]
    return [qml.expval(qml.PauliZ(2*x) @ qml.PauliZ((2*x)+1))
            for x in range(hp["output_qubits"])]


def triplet_loss(params, qNode, anchor, positive, negative, alpha):
    a_value = qNode(params, anchor)
    p_value = qNode(params, positive)
    n_value = qNode(params, negative)

    dist_a_p = np.linalg.norm(a_value - p_value)**2
    dist_a_n = np.linalg.norm(a_value - n_value)**2

    return max(dist_a_p - dist_a_n + alpha, 0.0)


def train():
    assert(hp["dataset"] in ["mnist", "mnist_ae", "bc", "moons"])
    dev = qml.device('default.qubit', wires=hp["qubits"], shots=hp["shots"])
    qNode = qml.QNode(func=circuit, device=dev)

    stepsize = hp["start_stepsize"]
    optimizer = qml.AdamOptimizer(stepsize)

    def cost_fn(params):
        return triplet_loss(params, qNode, anchor,
                            positive, negative, hp["alpha"])

    params = np.random.uniform(low=-np.pi, high=np.pi,
                               size=(hp["layers"], hp["qubits"], 2)
                               )

    if hp["dataset"] == "mnist":
        train_x, train_y, test_x, test_y = load_mnist(seed=hp["seed"],
                                                      train_size=hp["train_size"],
                                                      test_size=hp["test_size"],
                                                      classes=hp["classes"],
                                                      wires=hp["data_qubits"]
                                                      )

        apn_generator = mnist_apn_generator(train_x,
                                            train_y,
                                            n_cls=len(hp["classes"])
                                            )
    elif hp["dataset"] == "mnist_ae":
        train_x, train_y, test_x, test_y = load_mnist_ae(train_size=hp["train_size"],
                                                         test_size=hp["test_size"],
                                                         classes=hp["classes"],
                                                         wires=hp["data_qubits"]
                                                         )

        apn_generator = mnist_apn_generator(train_x,
                                            train_y,
                                            n_cls=len(hp["classes"])
                                            )
    elif hp["dataset"] == "bc":
        train_x, train_y, test_x, test_y = load_breast_cancer_lju(hp["train_size"],
                                                                  hp["test_size"]
                                                                  )
        apn_generator = bc_apn_generator(train_x,
                                         train_y
                                         )
    elif hp["dataset"] == "moons":
        train_x, train_y, test_x, test_y = load_moons_dataset(hp["train_size"],
                                                              hp["test_size"]
                                                              )
        apn_generator = moons_apn_generator(train_x,
                                            train_y
                                            )

    accuracys = []
    dbis = []
    losses = []
    gradients = []

    for step in range(hp["steps"] + 1):
        anchor, positive, negative = next(apn_generator)

        params, c = optimizer.step_and_cost(cost_fn, params)

        print(f"step {step:{len(str(hp['steps']))}}| cost {c:8.5f}")

        losses.append(c)

        if step % hp["grads_every"] == 0:
            g, _ = optimizer.compute_grad(cost_fn, (params,), {}, None)
            gradients.append(np.average(np.abs(g)))
            # print("Gradients", g[0][0, 0, 0])
            # return g[0][0, 0, 0]

        if step % hp["test_every"] == 0:
            accuracy, dbi = evaluate(hp["dataset"], train_x, train_y,
                                     test_x, test_y,
                                     qNode, params, step,
                                     hp["classes"], hp["output_qubits"]
                                     )
            accuracys.append((step, accuracy))
            dbis.append((step, dbi))
            print("Accuracys:\n", accuracys)

        # if (step+1) % hp["update_sz_every"] == 0:
        #     stepsize *= hp["sz_factor"]
        #     optimizer.stepsize = stepsize
        #     print("Updated stepsize to", stepsize)

    if accuracys:
        print("Accuracys:\n", accuracys)
        print("Maximum: ", max(np.array(accuracys)[:, 1]))

    if dbis:
        print("DBIs:\n", dbis)
        print("Minimum:", min(np.array(dbis)[:, 1]))

    if gradients:
        print("Gradients Avg: ", np.average(gradients))

    with open(f"./trainings/{starting_time}.json", "w") as json_file:
        json.dump(hp, json_file)
    np.savez(f"./trainings/{starting_time}.npz",
             accuracys=accuracys,
             dbis=dbis,
             losses=losses,
             gradients=gradients,
             params=params
             )

    # plot_curves(np.array(accuracys),
    #             np.array(dbis),
    #             np.array(losses),
    #             f"Qubits: {hp['qubits']}, " +
    #             f"Layers: {hp['layers']}, " +
    #             f"Classes: {hp['classes']}, " +
    #             f"Output_dim: {hp['output_qubits']}"
    #             )
    

if __name__ == "__main__":
    # n_layers = [5, 25, 50, 75, 100]
    # n_qubits = [4, 6, 8, 10, 12]
    # seeds = list(range(100))

    # gradients = np.zeros((len(n_layers), len(n_qubits), len(seeds)))

    # for il, l in enumerate(n_layers):
    #     for iq, q in enumerate(n_qubits):
    #         for js, s in enumerate(seeds):
    #             hp["layers"] = l
    #             hp["qubits"] = q
    #             hp["seeds"] = s
    #             print(l, q, s)
    #             grad = train()
    #             gradients[il, iq, js] = grad

    # np.savez(f"./trainings/{starting_time}_grads.npz", grads=gradients,
    #                                                    n_layers=np.array(n_layers),
    #                                                    n_qubits=np.array(n_qubits),
    #                                                    seeds=np.array(seeds))
    # print(gradients)

    train()
