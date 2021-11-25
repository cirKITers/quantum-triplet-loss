from maskit.datasets import load_data
from pennylane import numpy as np

SEED = 2
CLASSES = (4, 6, 2)

data = load_data("mnist", shuffle=SEED, train_size=len(CLASSES)*1000, classes=CLASSES)

occurences = [0] * len(CLASSES)

for label in data.train_target:
    occurences[np.argmax(label)] += 1 

print(occurences)