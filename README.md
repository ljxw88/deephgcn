# DeepHGCN
This repository contains a PyTorch implementation of "DeepHGCN: Toward Deeper Hyperbolic Graph Convolutional Networks".

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
@ARTICLE{10632071,
  author={Liu, Jiaxu and Yi, Xinping and Huang, Xiaowei},
  journal={IEEE Transactions on Artificial Intelligence}, 
  title={DeepHGCN: Toward Deeper Hyperbolic Graph Convolutional Networks}, 
  year={2024},
  volume={},
  number={},
  pages={1-14},
  keywords={Graph neural networks;Riemannian manifold;hyperbolic operations;deep model architecture},
  doi={10.1109/TAI.2024.3440223}}
```