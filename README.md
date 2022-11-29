# Escaping Feature Twist: A Variational Graph Auto-Encoder for Node Clustering (IJCAI 2022)

## Abstract

Most recent graph clustering methods rely on pretraining graph auto-encoders using self-supervision techniques and fine tuning based on pseudo-supervision. However, the transition from self-supervision (pretext task) to pseudo-supervision (main task) has never been studied. To fill this gap, we focus on the geometric aspect by analysing the intrinsic dimension of the embedded manifolds. We found that the curved and low-dimensional latent structures undergo coarse geometric transformations during the transition state. We call this problem Feature Twist. To avoid this deterioration, we propose a principled approach that can gradually smooth the strong local curves while preserving the global curved structures. Our experimental evaluations have shown notable improvement over multiple state-of-the-art approaches.

## Usage

We provide the code of our model FT-VGAE for two datasets (Cora and Citeseer). For each dataset and each model, we provide the pretraining weights (the code of the first phase and the pretraining weights can be obtained using vanilla VGAE from this repos: https://github.com/zfjsail/gae-pytorch/tree/master/gae). The data is also provided with the code. To run the code of FT-VGAE on Cora, you should clone this repo and use the following command: 
```
python3 ./FT-VGAE/main_cora.py
```

## Citation

@inproceedings{ijcai2022p465,
  title     = {Escaping Feature Twist: A Variational Graph Auto-Encoder for Node Clustering },
  author    = {Mrabah, Nairouz and Bouguessa, Mohamed and Ksantini, Riadh},
  booktitle = {Proceedings of the Thirty-First International Joint Conference on
               Artificial Intelligence, {IJCAI-22}},
  publisher = {International Joint Conferences on Artificial Intelligence Organization},
  editor    = {Lud De Raedt},
  pages     = {3351--3357},
  year      = {2022},
  month     = {7},
  note      = {Main Track},
  doi       = {10.24963/ijcai.2022/465},
  url       = {https://doi.org/10.24963/ijcai.2022/465},
}
