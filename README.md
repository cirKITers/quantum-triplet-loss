[![DOI](https://zenodo.org/badge/430692221.svg)](https://zenodo.org/badge/latestdoi/430692221)

# quantum-triplet-loss

## How To Run

Create a `hyperparameters.json` in the main directory with the desired values. A good start would be:

```
{
    "dataset": "mnist_ae",
    "qubits": 12,
    "data_qubits": 8,
    "output_qubits": 2,
    "layers": 10,
    "train_size": 2500,
    "test_size": 500,
    "classes": [3, 6],
    "steps": 2000,
    "test_every": 50,
    "grads_every": 1,
    "start_stepsize": 0.005,
    "update_sz_every": 35000,
    "sz_factor": 0.1,
    "alpha": 1.0,
    "shots": null,
    "seed": 1234
}
```

To reproduce the results from the paper, use `"mnist_ae"` or `"moons"` as the `"dataset"` parameter. 

To start a Triplet Loss Training run `main.py`. To start a Cross Entropy Training run `classical_training.py`.

## Fixed Parameters

- Circuit, measurement and embedding
- 12 qubits
  - 2 outputs for cross-entropy loss
  - 2 to 8 outputs for triplet loss
- LR 0.005, Alpha 1.0 - both constant
- Training steps
  - 2000 for triplet loss
  - 6000 for cross-entropy loss
- Triplet choice completely random

## Tests

### Tests with positive effect
- ZZ measurement on qubits 0,1 and 2,3

### Tests without any effect
- StronglyEntanglingLayers
- Change of qubit- and layer count
- Changes of learning rate (dynamic and static) – 0.01 leads to best result
- Changes of alpha
  - Bigger alpha values lead to bigger variances of measurements
  - Changing the alpha value makes previous steps obsolete
- Measuring 3 qubits
- One separate circuit per measurement
- Data-Reuploading

### Tests with negative effect
- Online mining of triplets (everything is mapped to one value)
- Usual training, followed by online mining
- Change of loss
