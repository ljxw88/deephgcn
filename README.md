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
@misc{liu2024deephgcndeeperhyperbolicgraph,
      title={DeepHGCN: Toward Deeper Hyperbolic Graph Convolutional Networks}, 
      author={Jiaxu Liu and Xinping Yi and Xiaowei Huang},
      year={2024},
      eprint={2310.02027},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2310.02027}, 
}
```