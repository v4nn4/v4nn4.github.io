---
author: "v4nn4"
title: "Transformers Dashboard 🤖📈"
date: "2024-05-04"
tags: []
ShowToc: false
ShowBreadCrumbs: false
math: mathjax
---

➡️ https://transformers-dashboard.vercel.app

{{< figure align=center src="/posts/transformers-dashboard/dashboard.png" >}}


Since the publication of the now famous 2017 paper [Attention is All You Need](https://arxiv.org/abs/1706.03762)[^1], many large language models based on the transformer architecture have emerged. Fortunately, some studies [^2] [^3] have compiled extensive data on many published models, including the dimensions of their transformers.

Much like my experience learning about CNNs and their growth in complexity, I wanted to analyze LLM transformers. Which models are the largest? What is the optimal size for the feed-forward layer? Is it better to add more embeddings or more attention heads? Can we easily derive the total number of parameters from the network dimensions?

## Transformer model parameters

I will use the notations from the original Attention is All You Need [^1] paper.

- $N$ : the number of layers
- $h$ : the number of attention heads
- $d_{\textrm{model}}$ : the size of the embeddings
- $d_{\textrm{ff}}$ : the size of the hidden FFN layer
- $V$ : the vocabulary size, that is the number of tokens used

{{< figure align=center width=400 src="/posts/transformers-dashboard/transformer.png" >}}

In order to count model parameters, we need break the model down into building blocks:

- **Multi-head attention block** : trainable parameters are contained in weight matrices $W_i^Q, W_i^K, W_i^V$, for $1 \leq i \leq h$, as well as $W^O$ and their associated biaises. We then multiply the added number of parameters by $h$, the number of heads. Using the relationship $d_k=d_v=d_{\textrm{model}} / h$ [^1] we get

$$
\begin{aligned}
P_{\textrm{MHA}} &= h (2d_{\textrm{model}}d_k + 2d_k + d_{\textrm{model}}d_v + d_v) + hd_vd_{\textrm{model}} + d_{\textrm{model}} \\\\
                 &= 4 (d_{\textrm{model}}^2 +  d_{\textrm{model}})
\end{aligned}
$$

- **Feed-forward block** : in both the encoder and the decoder, the output of size $d_{\textrm{model}}$ is passed throught a feed-forward block [^1] : $f(x) = ReLU(xW_1 + b_1)W_2 + b_2$. This leads to the following number of parameters

$$ P_{\textrm{FFN}} = 2 (d_{\textrm{ff}} d_{\textrm{model}} + d_{\textrm{model}})$$

- **Layer normalization block** : gain and bias with dimension $d_{\textrm{model}}$

$$ P_{\textrm{LN}} = 2 d_{\textrm{model}} $$

- **Encoder** : the encoder has one MHA and one FFN. Each one has a norm layer.
$$ P_{\textrm{encoder}} = N (P_{\textrm{MHA}} +  P_{\textrm{FFN}} + 2P_{\textrm{LN}} )$$

- **Decoder** : the decoder has two MHA and one FFN. Each one has a norm layer.
$$ P_{\textrm{decoder}} = N (2P_{\textrm{MHA}} +  P_{\textrm{FFN}} + 3 P_{\textrm{LN}})$$

- **Linear block** : the linear block outputs as many logits as the vocabulary size, hence the dimension of its matrix and bias is

$$ P_{\textrm{linear}} = d_{\textrm{model}} V + V $$

Finally the total number of parameters is
$$ P = P_{\textrm{encoder}} + P_{\textrm{decoder}} + P_{\textrm{linear}} $$


## Gathering data

Although the aforementioned studies [^2] [^3] are invaluable and packed with useful information, they've become quickly outdated given the pace of model releases these days. I decided to collect my own data from original research papers, announcement posts, as well as some [Hugging Face](https://huggingface.co/) configuration files. I focused on models published by large research teams and/or that had significant impact.

Here are my findings:

- GPT [^4] used a causal decoder-only transformer, which many models have adopted. This means the encoder block is not present in most models
- GPT used $d_{\textrm{ff}}/d_{\textrm{model}} = 4 $
- According to [^2], sometimes biases are omitted in the model
- Sometimes, some parameters are omitted in the paper and implied from previous version of the model
- Closed-source models rarely disclose detailed architecture information
- Hugging Face configuration files generally display one version (size) from a family of models, potentially leading to misleading interpretations

## Publishing a dashboard

Once the data started to look interesting, I put together a small [Next.js](https://nextjs.org/) app using [shadcn/ui](https://ui.shadcn.com/) data tables. A dashboard is available at https://transformers-dashboard.vercel.app.


[^1]: Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., ... & Polosukhin, I. (2017). Attention is all you need. Advances in neural information processing systems, 30.

[^2]: Naveed, H., Khan, A. U., Qiu, S., Saqib, M., Anwar, S., Usman, M., ... & Mian, A. (2023). A comprehensive overview of large language models. arXiv preprint arXiv:2307.06435.

[^3]: Zhao, W. X., Zhou, K., Li, J., Tang, T., Wang, X., Hou, Y., ... & Wen, J. R. (2023). A survey of large language models. arXiv preprint arXiv:2303.18223.

[^4]: Radford, A., & Narasimhan, K. (2018). Improving Language Understanding by Generative Pre-Training.