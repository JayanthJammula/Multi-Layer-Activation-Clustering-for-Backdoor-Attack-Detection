# Multi-Layer Activation Clustering for Backdoor Attack Detection

## Introduction

Deep neural networks trained on untrusted data can inherit malicious backdoor behaviors that trigger targeted misclassification whenever an attacker-controlled pattern appears. Our objective is to surface those hidden behaviors without relying on pristine reference data by studying how poisoned and benign samples activate the internal layers of a model.

This repository implements a multi-layer activation clustering pipeline that fuses Spectral Signature analysis with the Activation Clustering defense. By extracting representations from several layers, we capture both high-level class semantics and lower-level trigger traces, making it harder for a backdoor signal to remain concealed in any single feature space. Suspicious clusters are highlighted, visualized, and can be used to filter or relabel compromised samples before retraining.

The scripts mirror the workflow from our paper: craft and visualize attacks, train potentially compromised models, harvest per-layer activations, and run the Spectral Signature (SS) and Activation Clustering (AC) defenses. Together they form a reproducible toolkit for evaluating and hardening models against data poisoning attacks.

### References

Spectral Signature: B. Tran, J. Li, and A. Madry, "Spectral signatures in backdoor attacks," in Proc. NIPS, 2018.

Activation Clustering: B. Chen, W. Carvalho, N. Baracaldo, H. Ludwig, B. Edwards, T. Lee, I. Molloy, and B. Srivastava, "Detecting Backdoor Attacks on Deep Neural Networks by Activation Clustering," arXiv:1811.03728, Nov 2018.

### Preparation

Pytorch 1.6.0

CUDA V10.1.243

### Usage

Create an attack (1SC attack with pattern B in our paper) by

```
python attack_crafting.py
```

The backdoor pattern can be visualized by

```
python true_pert_plot.py
```

Train a DNN on the possibly poisoned training set by

```
python train_models_contam.py
```

SS and AC both require extracting internal layer features, which can be done by

```
python get_features.py
```

SS defense:

```
python SS_defense.py
```

AC defense:

```
python AC_defense.py
```
