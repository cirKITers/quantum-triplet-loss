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
To start a Triplet Loss Training run `main.py`. To start a Cross Entropy Training run `classical_training.py`.

## Parameters explanation

| Parameter       | Explanation                                                                   |
|-----------------|-------------------------------------------------------------------------------|
| dataset         | Use `mnist_ae` or `moons` to reproduce the results from the paper.            |
| qubits          | Number of qubits of the PQC.                                                  |
| data_qubits     | On how many qubits the data is encoded. Might be fixed by the dataset choice. |
| output_qubits   | Number of measured qubits. For CE it will always be the number of classes.    |
| layers          | Number of layers of the PQC.                                                  |
| train_size      | Number of training samples. Might be limited by the dataset choice.           |
| test_size       | Number of test samples. Might be limited by the dataset choice.               |
| classes         | Classes (digits) used for a training with MNIST dataset.                      |
| steps           | Number of training steps.                                                     |
| test_every      | After how many steps the test dataset is evaluated.                           |
| grads_every     | After how many steps the gradients are printed.                               |
| start_stepsize  | Initial stepsize of the optimizer.                                            |
| update_sz_every | After how many steps the stepsize is updated (LR decay).                      |
| sz_factor       | Which factor is used to update the stepsize (not used for the paper).         |
| alpha           | Alpha value of the Triplet Loss.                                              |
| shots           | Number of shots. `null` for analytic mode.                                    |
| seed            | Random seed.                                                                  |

Note that only a few classes/features combinations are provided for the MNIST dataset. 

## Using the `mnist_ae` dataset

The `mnist_ae` dataset was created by training an autoencoder on the popular MNIST dataset. Every created dataset that we used for the experiments of our paper are provided in the `quantum-triplet-loss/datasets/minst_ae` directory. 

Lets look at the file `Train_[3,4,6,9]_features_8.npz` as an example. This dataset was created from the classes/digits 3, 4, 6 and 9, and every datapoint has 8 features. To use this dataset, you have to set `"data_qubits": 8` and `"classes": [3, 4, 6, 9]" in your `hyperparameters.json` file. The respective train and test set will be loaded automatically. 


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
