import numpy as np
import random
from collections import Counter
from sklearn.datasets import load_breast_cancer, make_moons
from datasets.breast_cancer.bc_features import *
from maskit.datasets import load_data

np.random.seed(1337)
BC_PATH = "./datasets/breast_cancer/breast-cancer.data"


def load_breast_cancer_lju(train_size=150, test_size=100, shuffle=True):
    file = open(BC_PATH, 'r')
    data = []
    for line in file:
        line = line.split(",")
        if "?" in line:
            continue
        datum = []
        for index, f in enumerate(line):
            datum.append(bc_features[index][f])
        data.append(datum)
    assert(train_size + test_size <= len(data))
    data = np.array(data)
    if shuffle:
        np.random.shuffle(data)
    x = data[:, 1:]
    y = data[:, 0]

    return x[:train_size], \
        y[:train_size], \
        x[train_size:train_size+test_size], \
        y[train_size:train_size+test_size]


def load_breast_cancer_skl(train_size=300, test_size=100):
    data = load_breast_cancer()
    assert(train_size + test_size <= data.data.shape[0])
    x = data.data
    y = data.target
    for i in range(x.shape[1]):
        x[:, i] = np.interp(x[:, i],
                            (x[:, i].min(), x[:, i].max()),
                            (0 + 1e-10, np.pi - 1e-10)
                            )

    pmt = np.random.permutation(len(x))
    x, y = x[pmt], y[pmt]
    return x[:train_size], \
        y[:train_size], \
        x[train_size:train_size+test_size], \
        y[train_size:train_size+test_size]


def load_mnist(seed, train_size, test_size, classes, wires):
    data = load_data("mnist", shuffle=seed,
                     train_size=train_size,
                     test_size=test_size,
                     classes=classes,
                     wires=wires
                     )
    occurences_train = [np.argmax(x) for x in data.train_target]
    occurences_test = [np.argmax(x) for x in data.test_target]
 
    print("Train", Counter(occurences_train))
    print("Test", Counter(occurences_test))

    return (
            data.train_data,
            data.train_target,
            data.test_data,
            data.test_target,
            )


def mnist_apn_generator(train_x, train_y, n_cls):
    images = {}
    for cls in range(n_cls):
        images[cls] = []

    for index, label in enumerate(train_y):
        images[np.argmax(label)].append(train_x[index])

    while True:
        pos, neg = random.sample(range(n_cls), 2)

        anchor, positive = random.sample(images[int(pos)], 2)
        negative = random.choice(images[int(neg)])

        yield anchor, positive, negative


def bc_apn_generator(train_x, train_y):
    _, unique_elements = np.unique(train_x, return_index=True, axis=0)

    train_x = train_x[unique_elements, :]
    train_y = train_y[unique_elements]

    train_y = np.expand_dims(train_y, axis=1)
    data = np.concatenate((train_y, train_x), axis=1)

    mask_0 = (data[:, 0] == 0)
    mask_1 = (data[:, 0] == 1)
    data_0 = data[mask_0, :]
    data_1 = data[mask_1, :]

    while True:
        # same distribution as dataset
        # anchor_cls = data[np.random.randint(0, data.shape[0], 1)][0][0]

        # 50:50 distribution
        anchor_cls = random.choice([0, 1])

        if anchor_cls == 0:
            anc, pos = random.sample(range(data_0.shape[0]), 2)
            anchor, positive = data_0[anc], data_0[pos]
            negative = data_1[np.random.randint(0, data_1.shape[0], 1)][0]
        elif anchor_cls == 1:
            anc, pos = random.sample(range(data_1.shape[0]), 2)
            anchor, positive = data_1[anc], data_1[pos]
            negative = data_0[np.random.randint(0, data_0.shape[0], 1)][0]

        yield anchor[1:], positive[1:], negative[1:]


def load_moons_dataset(train_size=300, test_size=100):
    X, Y = make_moons(n_samples=train_size+test_size, shuffle=True, noise=0.15, random_state=1337)
    X = (X - np.min(X))/np.ptp(X)*np.pi

    return X[:train_size], Y[:train_size], X[train_size:], Y[train_size:]


def moons_apn_generator(train_x, train_y):
    train_y = np.expand_dims(train_y, axis=1)
    data = np.concatenate((train_y, train_x), axis=1)

    mask_0 = (data[:, 0] == 0)
    mask_1 = (data[:, 0] == 1)
    data_0 = data[mask_0, :]
    data_1 = data[mask_1, :]

    while True:
        # 50:50 distribution
        anchor_cls = random.choice([0, 1])

        if anchor_cls == 0:
            anc, pos = random.sample(range(data_0.shape[0]), 2)
            anchor, positive = data_0[anc], data_0[pos]
            negative = data_1[np.random.randint(0, data_1.shape[0], 1)][0]
        elif anchor_cls == 1:
            anc, pos = random.sample(range(data_1.shape[0]), 2)
            anchor, positive = data_1[anc], data_1[pos]
            negative = data_0[np.random.randint(0, data_0.shape[0], 1)][0]

        yield anchor[1:], positive[1:], negative[1:]


if __name__ == "__main__":
    # dataset = load_breast_cancer_skl()
    # for d in dataset:
    #     print(d.shape)
    # train_x, train_y, test_x, test_y = dataset
    # print(train_x[0])

    # dataset = load_breast_cancer_lju(shuffle=False)
    # for d in dataset:
    #     print(d.shape)
    # train_x, train_y, test_x, test_y = dataset
    # for label, features in zip(train_y, train_x):
    #     print(label, features)

    dataset = load_moons_dataset()
    for d in dataset:
        print(d.shape)
    train_x, train_y, test_x, test_y = dataset

    import matplotlib.pyplot as plt
    for x, y in zip(train_x, train_y):
        if y == 0:
            plt.scatter(*x, color="red")
        else:
            plt.scatter(*x, color="blue")

    plt.show()