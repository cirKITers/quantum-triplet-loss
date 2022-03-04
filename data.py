import numpy as np
import random
from collections import Counter
from sklearn.datasets import load_breast_cancer
from datasets.breast_cancer.bc_features import *
from maskit.datasets import load_data

np.random.seed(1337)
BC_PATH = "./datasets/breast_cancer/breast-cancer.data"


def load_breast_cancer_lju(train_size=150, test_size=100):
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
                            (0, np.pi)
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
                wires=wires)
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


def bc_apn_generator():
    pass


if __name__ == "__main__":
    dataset = load_breast_cancer_skl()
    for d in dataset:
        print(d.shape)
    train_x, train_y, test_x, test_y = dataset
    print(train_x[0])

    dataset = load_breast_cancer_lju()
    for d in dataset:
        print(d.shape)
    train_x, train_y, test_x, test_y = dataset
    print(train_x[0])




