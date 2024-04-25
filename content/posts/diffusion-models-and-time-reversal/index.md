---
author: "vanna"
title: "Diffusion models and time reversal"
date: "2023-10-24"
tags: ["diffusion"]
ShowToc: false
ShowBreadCrumbs: false
math: katex
---

I recently spent some time reading about the algorithms behind [Stable Diffusion](https://stability.ai/blog/stable-diffusion-public-release) and similar image generation models. They have been linked with an interesting 40-years-old result on diffusion processes[^1]. In short, this result states that there exists an explicit path from an initial probability distribution $p_0$ to a random noise (a normal distribution), and that this path can be reversed.

One application of this concept is sampling : we can draw a sample from a random noise and use the backward diffusion to obtain a sample from $p_0$. In the context of computer vision, this distribution would be multivariate with dimension $3 \times W \times H$, where $W$ and $H$ are respectively the width and height of a RGB image in pixels. The initial release of Stable Diffusion was using $H=W=512$.

In this document, I'll delve into the mechanics of reverse-time diffusions in dimension 1 (one pixel, one color channel) and derive equations to build a better understanding and intuition.

## Forward diffusion

The transition from an image to noise is accomplished by repetitively introducing random perturbations to our initial distribution. This can be done in a continuous time setting using an [Ornstein-Uhlenbeck stochastic process](https://en.wikipedia.org/wiki/Ornstein%E2%80%93Uhlenbeck_process). The Ornstein-Uhlenbeck process is defined by the following stochastic differential equation (SDE) :

$$ dX_t = - \theta (X_t - \mu) dt + \sigma dW_t $$

where $W$ is a Brownian motion defined on some probability space $(\Omega, \mathcal{F}, P)$. The first term is forcing the mean to converge to $\mu$, while the second term is adding noise.

We can show using [Itô's formula](https://en.wikipedia.org/wiki/It%C3%B4%27s_lemma) with $f(t, x) = x e^{\theta t}$ that 

$$ X_t = \mu + (X_0 - \mu) e^{-\theta t} + \sigma \int_0^t e^{-\theta (t - s)} dW_s $$

This means that the marginals of $X$ conditional to $X_0$ follow a normal distribution

$$ X_t  | X_0 \sim \mathcal{N}\left(\mu + (X_0 - \mu) e^{-\theta t}, \sigma^2 \frac{1-e^{-2\theta t}}{2\theta}\right) $$

We note that the mean tends to $\mu$, and the variance to $\sigma^2/(2\theta)$. We can then reparameterize the SDE such that the distribution of $X$ converges to a standard normal distribution (with unit variance) :

$$ dX_t = - \frac{\sigma^2}{2} X_t dt + \sigma dW_t \tag{1} $$

and :

$$ X_t | X_0 \sim \mathcal{N}\left(X_0 \sqrt{\alpha_t}, 1 - \alpha_t \right), \ \alpha_t = e^{- \sigma^2 t} $$

At this point we can notice a few things :

- **Impact of the noise level $\sigma$** : the noise (or volatility) parameter $\sigma$ has a similar impact on the diffusion as the time $t$. Increasing $\sigma$ will lead to noise being added faster
- **Distribution of $X$** : the distribution of $X$ given $X_0$ is normal, but the unconditional distribution is not in general. In the case where $X_0$ is itself normal, each step of the diffusion will generate a new normal distribution as we sum independant Gaussian variables together

## Backward diffusion

We follow the notations of Haussmann and Pardoux[^7] and define $Y_t = X_{1-t}$ for $0 \leq t \leq 1$. It can be shown[^1][^7] that the reverse process $Y$ follows the SDE

$$ dY_t = \left[ \frac{\sigma^2}{2} Y_t + \sigma^2 s_{1-t}(Y_t) \right] dt +  \sigma d\overline{W}_t $$

where 

$$ s_t(x) = (\log p_{t})'(x)$$

is the score function associated with distribution $p_{t}$ (derivative of its logarithm) and $\overline{W}$ another Brownian motion. In practice, the score is intractable and needs to be approximated.

### Score matching

One method is score matching[^2][^3]. The score function $s_t$ is approximated using a parametrical model $s_\theta$ :

$$ \min_\theta J(\theta) = \min_\theta \mathbb{E}\left[ \frac{1}{2} \left(s_\theta(X_t) - s_t(X_t)\right)^2  \right] $$

The $1/2$ constant is only a convention. We can simplify this expression by using an integration by part trick[^4]. First we develop $J$ :

$$ \int_D \frac{1}{2}(s_\theta(x)^2 - 2 s_\theta(x) s_t(x))p_t(x)dx + \frac{1}{2}\mathbb{E}\left[ s_t(X_t)^2 \right] $$

Since the last term does not depend on $\theta$, minimizing the rest will suffice. Thus our new cost function $\tilde{J}$ reads, after integration by part :

$$ \tilde{J}(\theta) = \mathbb{E}\left[ \frac{1}{2} s_\theta(X_t)^2 +  s_\theta'(X_t) \right] $$

This removes the dependency to an explicit formulation of the distribution $p_t$. Instead we can rely on Monte-Carlo estimation for instance to compute the cost function.

**Note 1** : $s_\theta = 0$ is the best constant model, but is beaten by linear models with negative slope. Indeed, the second term pushes the derivative of the score to be negative. This is consistent with the Gaussian case.

**Note 2** : In dimension $N > 1$, the formulation becomes[^4] :

$$ s_t(\textbf{x}) = \nabla \log p_t (\textbf{x}) $$

$$ \tilde{J}(\theta) = \mathbb{E}\left[ \frac{1}{2} \lVert s_\theta(X_t) \rVert_2^2 + \textrm{tr}(\nabla s_\theta(X_t)) \right] $$

 Neural networks are used in practice, where $\theta$ represents the weights of the network. The U-Net[^5] architecture for instance is used to estimate the score function in computer vision[^6].

### Gaussian mixtures

Another method would be to actually approximate the distribution $p_t$ with a [Gaussian mixture](https://en.wikipedia.org/wiki/Mixture_model#Gaussian_mixture_model). In this case, the score is explicit, which is a nice property.  The drawback of this method is probably that the space of Gaussian mixtures is restrictive in some sense.

## Centered Gaussian case

When the initial distribution is Gaussian, the process $X$ stays Gaussian, and we get a closed form for the distribution $p$. We can then derive explicitely the reverse process.

**Note** : going from a normal distribution to a standard normal distribution can be done in one step : $X_1 = (X_0 - \mu) / \sigma$. The following derivation is for learning purpose only!

When $X_0 \sim \mathcal{N}(0, \sigma_0^2)$ we get

$$ p_t \sim \mathcal{N}\left(0, \sigma_0^2 \alpha_t + 1 - \alpha_t\right) $$

In this case the score function is

$$ s_t(x) = -\frac{x}{\sigma_0^2 \alpha_t + 1 - \alpha_t} $$

and the reverse SDE reads

$$ d\bar{X}_t = -\frac{\sigma^2}{2} \beta_t \bar{X}_t dt + \sigma d\bar{W}_t $$

where

$$ \beta_t = \frac{1 + (1 -\sigma_0^2) \alpha_{1-t}}{1 - (1 - \sigma_0^2 )\alpha_{1-t}} $$

Some interesting cases :
- $\sigma_0 = 1$ yields the forward SDE, which is expected since the initial and target distributions are the same in this case
- $\sigma_0^2 < 2$ means the mean reversion speed is always positive, while when $\sigma_0^2 \geq 2$, it can change sign during diffusion
- $\sigma_0 = 0$ leads to a SDE that starts as the forward SDE and tends to a singularity as $\alpha_{1-t}$ approaches 1. Indeed, the SDE cannot converge to a constant, as an independent noise is added at each step

The backward diffusion is an Ornstein-Uhlenbeck process with time dependent parameters. Using Itô's formula again with $f(t, x) = x e^{\frac{\sigma_0^2}{2}\int_0^t \beta_s ds}$, we get :

$$ \bar{X}_t = \bar{X}_0 e^{-\frac{\sigma^2}{2} \int_0^t \beta_s ds} + \sigma \int_0^t e^{-\frac{\sigma^2}{2}\int_s^t \beta_u du} d\bar{W}_s $$

By rewriting $\beta_t$ as

$$ \beta_t = 1 - \frac{2 (1 -\sigma_0^2) \alpha_{1-t}}{1 - (1 - \sigma_0^2 )\alpha_{1-t}} $$

one can recognize the derivative of a logarithm and find that

$$ e^{-\frac{\sigma^2}{2}\int_0^t \beta_s ds } = e^{-\frac{\sigma^2}{2}t} \frac{1- (1-\sigma_0^2)\alpha_{1-t}}{1-(1-\sigma_0^2)\alpha_1} $$

which tends to $0$ as $t \rightarrow 1$. This means that the variance contribution of the firm term is zero. The variance of the second term is more involved but leads to :

$$ \frac{(1-\gamma \alpha_{1-t})^2}{\gamma \alpha_{a-t}} \left[ \frac{1}{1 -\gamma \alpha_{1-t}} - \frac{1}{1-\gamma\alpha_1} \right] $$

where $\gamma = 1 - \sigma_0^2$, which tends to

$$ (1-\gamma)\frac{1 - \alpha_1}{1 - \gamma \alpha_1}$$

as $t \rightarrow 1$. Finally, we notice that this expression tends to $\sigma_0^2$ if $\sigma$ is sufficiently high. Similarly, we could have used $T > 1$ for a fixed $\sigma$, and let $T$ tend to infinity. As previously mentioned, noise level and time play similar roles.

Something interesting to note is that for a given level of convergence of the variance in the forward SDE, that is $\alpha_1 = \epsilon$, the variance of the reverse process will either converge faster or slower relative to $\sigma_0^2$ depending on the sign of $\gamma$. When the initial distribution has more variance than the random noise, the backward SDE will converge slower and vice versa.

## PyTorch implementation

We consider a 1-dimensional distribution $p$, and use the previous diffusion to generate a random noise. In order to target an end variance of $1 - \varepsilon$, we let $T = 1$ and $\sigma$ such that $\alpha_T = \varepsilon$. We use the [Euler scheme](https://en.wikipedia.org/wiki/Euler%E2%80%93Maruyama_method) to discretize the diffusion.

```python
import math

import torch


NB_PATHS = 50
NB_TIMESTEPS = 100
T = 1.0
EPS = 0.001


def forward_diffusion_step(
    X: torch.Tensor,
    Z: torch.Tensor,
    sigma: float,
    dt: float,
) -> torch.Tensor:
    return X - 0.5 * sigma * sigma * X * dt + sigma * Z * math.sqrt(dt)


dt = T / NB_TIMESTEPS
sigma = math.sqrt(-math.log(EPS) / T)  # alpha_T = eps
X0 = torch.normal(0, 1, size=(NB_PATHS,))
X = torch.zeros((NB_TIMESTEPS, NB_PATHS))
Z = torch.normal(0, 1, size=(NB_TIMESTEPS - 1, NB_PATHS))
X[0] = X0
for i in range(NB_TIMESTEPS - 1):
  X[i+1] = forward_diffusion_step(X[i], Z[i], sigma, dt)
```

Here is a forward diffusion from $\mathcal{N}(10, 0.1)$ to $\mathcal{N}(0, 1)$ :

{{< figure align=center src="/posts/diffusion-models-and-time-reversal/forward_diffusion.png" >}}

We write backward diffusion with score function in the normal case $\mathcal{N}(\mu_0, \sigma_0^2)$

```python
def backward_diffusion_step(
    X: torch.Tensor,
    Z: torch.Tensor,
    sigma: float,
    mu0: float,
    sigma0: float,
    dt: float,
    t: float,
) -> torch.Tensor:
    sigma2 = sigma * sigma
    alpha = math.exp(-sigma2 * (1 - t))
    score = -(X - mu0 * math.sqrt(alpha)) / (sigma0 * sigma0 * alpha + 1 - alpha)
    return (
        X
        + 0.5 * sigma2 * X * dt
        + score * sigma2 * dt
        + sigma * Z * math.sqrt(dt)
    )
```

{{< figure align=center src="/posts/diffusion-models-and-time-reversal/backward_diffusion.png" >}}

Finally, we can re-write the score used in the backward SDE when the initial distribution is a Gaussian mixture. Indeed, Gaussian mixtures are stable by addition with a normal variable, and we can proceed similarly to the previous case.

```python
def backward_diffusion_step(
    X: torch.Tensor,
    Z: torch.Tensor,
    sigma: float,
    params0: List[Tuple[float, float]],
    t: float,
    dt: float,
) -> torch.Tensor:
    sigma2 = sigma * sigma
    alpha = math.exp(-sigma2 * (1 - t))
    n = len(params0)
    score, divisor = 0, 0
    for (mu0, sigma0) in params0:
        mut = mu0 * math.sqrt(alpha)
        sigmat2 = sigma0 * sigma0 * alpha + 1 - alpha
        denominator = math.sqrt(2 * math.pi * sigmat2)
        pit = np.exp(-(X - mut) * (X - mut) / (2 * sigmat2)) / denominator
        divisor += 1 / n * pit
        score -= 1 / n * pit * (X - mut) / sigmat2
    score /= divisor
    return (
        X
        + 0.5 * sigma2 * X * dt
        + score * sigma2 * dt
        + sigma * Z * math.sqrt(dt)
    )
```

{{< figure align=center src="/posts/diffusion-models-and-time-reversal/gmm.png" >}}

That's all for today! ✨

[^1]: ANDERSON, Brian DO. Reverse-time diffusion equation models. Stochastic Processes and their Applications, 1982, vol. 12, no 3, p. 313-326.

[^7]: HAUSSMANN, Ulrich G. et PARDOUX, Etienne. Time reversal of diffusions. The Annals of Probability, 1986, p. 1188-1205.

[^2]: SONG, Yang, SOHL-DICKSTEIN, Jascha, KINGMA, Diederik P., et al. Score-based generative modeling through stochastic differential equations. arXiv preprint [arXiv:2011.13456](https://arxiv.org/abs/2011.13456), 2020.

[^3]: WEBER, Romann M. The Score-Difference Flow for Implicit Generative Modeling. arXiv preprint [arXiv:2304.12906](https://arxiv.org/abs/2304.12906), 2023.

[^4]: HYVÄRINEN, Aapo et DAYAN, Peter. Estimation of non-normalized statistical models by score matching. Journal of Machine Learning Research, 2005, vol. 6, no 4.

[^5]: RONNEBERGER, Olaf, FISCHER, Philipp, et BROX, Thomas. U-net: Convolutional networks for biomedical image segmentation. In : Medical Image Computing and Computer-Assisted Intervention–MICCAI 2015: 18th International Conference, Munich, Germany, October 5-9, 2015, Proceedings, Part III 18. Springer International Publishing, 2015. p. 234-241.

[^6]: KARRAS, Tero, AITTALA, Miika, AILA, Timo, et al. Elucidating the design space of diffusion-based generative models. Advances in Neural Information Processing Systems, 2022, vol. 35, p. 26565-26577.