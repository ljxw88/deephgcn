# DeepHGCN
This repository contains a PyTorch implementation of "DeepHGCN: Toward Deeper Hyperbolic Graph Convolutional Networks".

Paper: https://ieeexplore.ieee.org/abstract/document/10632071

### Directories
- `deephgcn`: code implementation for the proposed DeepHGCN model.
- `geooptplus`: the augmented geoopt package for implementing key features of HNN++.
- `layers`: including the implementation of proposed Hyperbolic Linear Layer, as well as other compared feature transformation methods in DeepHGCN paper.
- `notes`: miscellaneous visualizations and basic evaluations.

### Requirement
- geoopt ```(pip install geoopt)```
- networkx
- scikit-learn

### Reference Code

- geoopt: https://github.com/geoopt/geoopt
- HNN++: https://github.com/mil-tokyo/hyperbolic_nn_plusplus
- HGCN: https://github.com/HazyResearch/hgcn
- GCNII: https://github.com/chennnM/GCNII
- Fréchet Mean: https://github.com/CUAI/Differentiable-Frechet-Mean

### Citation
```
@article{liu2024deephgcn,
  title={DeepHGCN: Toward Deeper Hyperbolic Graph Convolutional Networks},
  author={Liu, Jiaxu and Yi, Xinping and Huang, Xiaowei},
  journal={IEEE Transactions on Artificial Intelligence},
  year={2024},
  publisher={IEEE}
}
```
