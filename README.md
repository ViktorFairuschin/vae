# Variational Autoencoder

Tensorflow implementation of dense variational autoencoder for MNIST.

### Theory

#### Variational inference

In probabilistic modelling, one is often interested in inferring the posterior distribution p(z|x) 
of some latent variables z given the observations x

<img src="https://latex.codecogs.com/svg.image?p(z|x)=\frac{p(z,x)}{p(x)}" />

### Requirements

`tensorflow 2.0.0` or higher, `numpy`, `matplotlib`

### Results

### References

1. Kingma, D. P., & Welling, M. (2013). Auto-encoding variational bayes. ([arXiv:1312.6114](https://arxiv.org/pdf/1312.6114.pdf))
2. Burgess, C. P., Higgins, I., Pal, A., Matthey, L., Watters, N., Desjardins, G., & Lerchner, A. (2018). Understanding disentangling in $\beta$-VAE. ([arXiv:1804.03599](https://arxiv.org/pdf/1804.03599.pdf))