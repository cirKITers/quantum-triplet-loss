[![DOI](https://zenodo.org/badge/430692221.svg)](https://zenodo.org/badge/latestdoi/430692221)

# quantum-triplet-loss

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
