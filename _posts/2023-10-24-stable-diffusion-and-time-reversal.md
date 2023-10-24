---
layout: post
title: Stable Diffusion and time reversal
---

I recently spent some time reading about the algorithms behind [Stable Diffusion](https://stability.ai/blog/stable-diffusion-public-release). They heavily rely on a 40 years old result[^1] on diffusion processes. In essence, this result states that there exist an explicit path from an initial probability distribution $$p$$ to a random noise (a normal distribution), and that this path can be reversed.

One application is sampling : when drawing samples from $$p$$ is difficult, one can draw from a random noise and use the reverse diffusion to get a sample from $$p$$. This is precisely how Stable Diffusion and other similar algorithms work, where $$p$$ is the distribution of pixels intensities in a $$N \times N$$ image.

In this note I am studying time reversal diffusion using an Ornstein-Uhlenbeck diffusion with constant noise $$\sigma$$. I derive the backward diffusion using a naive Monte-Carlo method.

## Forward diffusion

The Ornstein-Uhlenbeck process is defined by the following stochastic differential equation (SDE) : 

$$ dX_t = - \theta (X_t - \mu) dt + \sigma dW_t $$

We can show using Itô's formula with $$f(x, t) = x e^{\theta t}$$ that 

$$ X_t = \mu + (X_0 - \mu) e^{-\theta t} + \sigma \int_0^t e^{-\theta (t - s)} dW_s $$

This means that the marginals of $$X$$ conditional to $$X_0$$ follow a normal distribution

$$ X_t  | X_0 \sim \mathcal{N}\left(\mu + (X_0 - \mu) e^{-\theta t}, \sigma^2 \frac{1-e^{-2\theta t}}{2\theta}\right) $$

We note that the mean tends to $$\mu$$, and the variance to $$\sigma^2/(2\theta)$$. We can reparameterize the SDE such that the distribution of $$X$$ converges to a standard normal distribution :

$$ dX_t = - \frac{1}{2} \sigma^2 X_t dt + \sigma dW_t \tag{1} $$

where :

$$ X_t | X_0 \sim \mathcal{N}\left(X_0 \sqrt{\alpha_t}, 1 - \alpha_t \right), \ \alpha_t = e^{- \sigma^2 t} $$

At this point we can note a few things :

- **Choice of $$\sigma$$**: $$\sigma$$ and $$t$$ have a similar impact on the distribution of $$X$$. Any strictly positive value for $$\sigma$$ leads to a diffusion from $$p$$ to $$\mathcal{N}(0, 1)$$. My intuition is that their must be some optimality in the choice of $$\sigma$$, maybe to keep the reverse process stable
- The distribution of $$X$$ given $$X_0$$ is normal, but the unconditional distribution is not necessary normal. In the case of a normal $$X_0$$ distribution however, each step of the Euler scheme will generate a new normal distribution as we sum independant Gaussian variables

## PyTorch implementation

We consider a 1-dimensional distribution $$p$$, and use the previous diffusion to generate a random noise. In order to target an end variance of $$1 - \varepsilon$$, we let $$T = 1$$ and $$\sigma$$ such that $$\alpha_T = \varepsilon$$.

```python
import torch
import math

NB_PATHS = 50
NB_TIMESTEPS = 100
T = 1.0
EPS = 0.001

dt = T / NB_TIMESTEPS
sigma = math.sqrt(-math.log(EPS) / T)  # alpha_t = eps
X0 = torch.normal(3, 1, size=(NB_PATHS,))
X = torch.zeros((NB_TIMESTEPS, NB_PATHS))
Z = torch.normal(0, 1, size=(NB_TIMESTEPS - 1, NB_PATHS))

def diffusion_step(X: torch.Tensor, dt: float, sigma: float, Z: torch.Tensor) -> torch.Tensor:
  return X - 0.5 * sigma * X * dt + sigma * Z * math.sqrt(dt)

X[0] = X0
for i in range(NB_TIMESTEPS - 1):
  X[i+1] = diffusion_step(X[i], dt, sigma, Z[i])
```

Here is a forward diffusion from $$\mathcal{N}(3, 1)$$ to $$\mathcal{N}(0, 1)$$ :

![Diffusion from N(3, 1) to N(0, 1)](/assets/images/diffusion/diffusion_n31_n01.png){: .center}

We can also do this with a bimodal distribution, for instance the concatenation of two normal distributions :

```python
X0 = torch.concat([
  torch.normal(+3, 1, size=(NB_PATHS // 2,)),
  torch.normal(-3, 1, size=(NB_PATHS // 2,)),
])
```

![Diffusion from [N(3, 1), N(-3, 1)] to N(0, 1)](/assets/images/diffusion/diffusion_bimodal.png){: .center}

According to Anderson[^1], the reverse time SDE reads

$$ dX_t = -\sigma^2 \left[ \frac{1}{2} X_t - s_t(X_t) \right] dt +  \sigma(t) d\hat{W}_t $$

where 

$$ s_t(x) = \frac{d\log p_t(x)}{dx}$$

is the score function and time is flowing from $$T$$ to 0 and $$dt$$ is assumed to be a negative time step. This equation is a McKean-Vlasov SDE as it depends on the local distribution $$p_t$$. In practice, this quantity is intractable and approximated using different techniques. 

One of this technique is score matching, see [^2] and [^3] for reference. The score function $$s_t$$ is approximated using parametrical models and a clever integration by part trick (see Hyvärinen[^4]). This method is well suited for high dimensional problem as the scoring function can be learned ahead of time using neural networks. The U-Net[^5] architecture seems to be used extensively for this purpose in computer vision (see [^6] for an overview).

A naive way would be to express the distribution $$p_t$$ as a function of the initial distribution $$p_0$$ and use a Monte-Carlo method :

$$ p_t(x) = \int p_{y\sqrt{\alpha_t}, 1-\alpha_t}(x) p_0(y) dy $$

where $$ p_{\mu, \sigma} \sim \mathcal{N}(\mu, \sigma)$$. Then we can approximate $$p_t$$ using estimator $$\widetilde{p}_t$$:

$$ \widetilde{p}_t(x) = \frac{1}{N} \sum_{i=1}^N  p_{X_0^i\sqrt{\alpha_t}, 1-\alpha_t}(x) $$

The score function then reads

$$ s(x) = \frac{\sum_{i=1}^N  - \frac{x - X_0^i \sqrt{\alpha_t}}{1-\alpha_t} p_{X_0^i\sqrt{\alpha_t}, 1-\alpha_t}(x)}{\sum_{i=1}^N p_{X_0^i\sqrt{\alpha_t}, 1-\alpha_t}(x)}  $$

This method can be heavy as it needs to operate on the entire dataset at each step. Since we are dealing with one-dimensional data, we should be fine.

TBC...


[^1]: ANDERSON, Brian DO. Reverse-time diffusion equation models. Stochastic Processes and their Applications, 1982, vol. 12, no 3, p. 313-326.

[^2]: SONG, Yang, SOHL-DICKSTEIN, Jascha, KINGMA, Diederik P., et al. Score-based generative modeling through stochastic differential equations. arXiv preprint [arXiv:2011.13456](https://arxiv.org/abs/2011.13456), 2020.

[^3]: WEBER, Romann M. The Score-Difference Flow for Implicit Generative Modeling. arXiv preprint [arXiv:2304.12906](https://arxiv.org/abs/2304.12906), 2023.

[^4]: HYVÄRINEN, Aapo et DAYAN, Peter. Estimation of non-normalized statistical models by score matching. Journal of Machine Learning Research, 2005, vol. 6, no 4.

[^5]: RONNEBERGER, Olaf, FISCHER, Philipp, et BROX, Thomas. U-net: Convolutional networks for biomedical image segmentation. In : Medical Image Computing and Computer-Assisted Intervention–MICCAI 2015: 18th International Conference, Munich, Germany, October 5-9, 2015, Proceedings, Part III 18. Springer International Publishing, 2015. p. 234-241.

[^6]: KARRAS, Tero, AITTALA, Miika, AILA, Timo, et al. Elucidating the design space of diffusion-based generative models. Advances in Neural Information Processing Systems, 2022, vol. 35, p. 26565-26577.