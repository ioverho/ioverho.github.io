---
title: Manifold-Constrained Hyper-Connections
date: 2026-02-14
tags:
  - machine-learning
math: true
draft: false
---

{{< toc >}}

This post comes from a read-group style presentation I gave to my [NLU lab-mates](https://www.shutova.org/home/people). You can find the embedded slides below, or by [following this link](https://www.ivoverhoeven.nl/mHC). The rest of this post is a more detailed write up of notes.

<iframe src="https://www.ivoverhoeven.nl/mHC" width="100%" style="aspect-ratio: 16 / 9;"></iframe>

<!-- <hr style="margin-top: 1em; margin-bottom: 1em;"> -->

## Introduction

Zhu et al.'s {{< cite "zhuHyperConnections2025" >}} Hyper-Connections (HC) and the recent follow-up work by Xie et al. {{< cite "xieMHCManifoldConstrainedHyperConnections2026" >}} on Manifold-Constrained Hyper-Connections (mHC) represent some of the most exciting architectural design work in machine-learning that we've seen in a long time. It touches upon a lot of fundamental ideas in deep learning, and proposes a very elegant framework for massively increasing the topologocial complexity of any contained neural network with minimal additional parameters or computational overhead.

In the following sections I'll start by discussing residual connections, and why they led to deep learning taking off, before starting with Hyper-Connections and finishing with Manifold-Constrained Hyper-Connections.

I've tried to align the notation between the papers and sections. This can make it more difficult to go from article to article. Here's a conversion table:
 
| Name                          |       This Article        |        HC        |             mHC             |
| :---------------------------- | :-----------------------: | :--------------: | :-------------------------: |
| Input                         |            $x$            |        -         |              -              |
| Hidden state                  |            $z$            |       $h$        |             $x$             |
| Hyper Hidden Matrix           |       $\mathbf{Z}$        |   $\mathbf{H}$   |        $\mathbf{X}$         |
| Operation/Function            |            $f$            |  $\mathcal{T}$   |        $\mathcal{F}$        |
| Residual connections          | $\boldsymbol{\alpha}_{r}$ | $\mathbf{A}_{r}$ | $\mathcal{H}^{\text{res}}$  |
| $z^{l-1}$-$f^{l}$ connections | $\boldsymbol{\alpha}_{m}$ | $\mathbf{A}_{m}$ | $\mathcal{H}^{\text{pre}}$  |
| $f^{l}$-$z^{l}$ connections   |   $\boldsymbol{\beta}$    |   $\mathbf{B}$   | $\mathcal{H}^{\text{post}}$ |

## Residual Connections

A residual connection, also known as a skip connection, adds to a layer's output the identity function transformed input:
$$z^{l}=f^{l}(z^{l-1})+z^{l-1}$$
Graphically, we can depict this as follows:

{{< figure-dynamic
    dark-src="./figures/residual_connections_dark.svg"
    light-src="./figures/residual_connections_light.svg"
    alt="A residual connection as a network diagram."
>}}

When we apply a residual connection in succession, we get:
$$
\begin{align*}
	z^{(1)}&=f^{1}(x)+x \\
	z^{(2)}&=f^{2}(f^{1}(x)+x)+f^{1}(x)+x \\
	z^{(3)}&=f^{3}(f^{2}(f^{1}(x)+x)+f^{1}(x)+x)+f^{2}(f^{1}(x)+x)+f^{1}(x)+x \\
	\vdots
\end{align*}$$

Simply put, the representation at layer $l$ is always constructed from [the sum of the previous $l$ layer representations](https://en.wikipedia.org/wiki/Residual_neural_network#Forward_propagation):
$$
\begin{align*}
	z^{l}&=x+\sum_{i=1}^{l-1}f^{i}(z^{i-1}) \\
	&x=z^{(0)}
\end{align*}
$$
Graphically, we can represent this recurrence relation as:

{{< figure-dynamic
	dark-src="./figures/residual_connections_unrolled_dark.svg"
	light-src="./figures/residual_connections_unrolled_light.svg"
	alt="A residual connection as an unrolled network diagram."
	target="residual_connections_unrolled"
	attr="Adapted from veitResidualNetworksBehave2016"
>}}

Applied in succession, this gives:
$$
\begin{align*}
	z^{(1)}&=f_{1}(x)+x \\
	z^{(2)}&=f_{2}(f_{1}(x)+x)+f_{1}(x)+x \\
	z^{(3)}&=f_{3}(f_{2}(f_{1}(x)+x)+f_{1}(x)+x)+f_{2}(f_{1}(x)+x)+f_{1}(x)+x \\
	\vdots
\end{align*}$$

The advantages of residual connections are numerous.

### Stable Gradients

Residual connections make optimizing deep neural nets much, much easier. Each layer contributes equally to the representation of the final layer, and as such, each receives a strong gradient signal.

For a neural net without residual connections, thanks to the chain rule, the gradient looks like a long product of gradients:

$$
\frac{\partial }{\partial x}\mathcal{L}(z^{l})= \frac{\partial}{\partial z^{l}}\mathcal{L}(z^{l})\frac{\partial}{\partial z^{l-1}}f^{l}(z^{l-1})\frac{\partial}{\partial z^{l-2}}f^{l-1}(z^{l-2})\ldots
$$

If the norm of these gradients are too small or too large, the backprop signal passed to earlier layers vanish or explode, respectively. With residual connections, instead of a long product chain, we instead get a sum over smaller products:

$$
\begin{align*}
	\frac{\partial }{\partial x}\mathcal{L}(z^{l}) &= \frac{\partial}{\partial z^{l}}\mathcal{L}(z^{l})\frac{\partial}{\partial x}z^{l} & \text{(chain rule)}\\
	&= \frac{\partial}{\partial z^{l}}\mathcal{L}(z^{l})\left(\frac{\partial}{\partial x} x +\frac{\partial}{\partial x}\sum_{i=1}^{L-1}f_{i}(z^{(i-1)})\right) & \text{(linearity)}\\
	&= \frac{\partial}{\partial z^{l}}\mathcal{L}(z^{l}) + \frac{\partial}{\partial z^{l}}\mathcal{L}(z^{l})\sum_{i=1}^{L-1}\frac{\partial}{\partial x}f_{i}(z^{(i-1)}) & \text{(linearity)}\\
	&= \frac{\partial}{\partial z^{l}}\mathcal{L}(z^{l}) + \frac{\partial}{\partial z^{l}}\mathcal{L}(z^{l})\left(\frac{\partial}{\partial x}f_{1}(x) + \frac{\partial}{\partial x}f_{2}(z_{1})+\ldots\right)
\end{align*}
$$

This mitigates the effect of vanishing/exploding gradients, because from the representations at each layer's output, there is always a direct path to the lowest layer inputs. To convinve yourself of this, take another look at the figure above.

<!-- Thus, residual connections mitigate the effect of [vanishing gradients](https://en.wikipedia.org/wiki/Vanishing_gradient_problem) in (arbitrarily) deep neural networks. This intuition led to the [ResNet](https://en.wikipedia.org/wiki/Residual_neural_network) architecture, which remains one of the most highly cited deep learning papers ever, and heralded in the new deep learning age {{< cite "heDeepResidualLearning2016" >}}. -->

### Ensembles of Networks & Generalization

Another suspected benefit of residual connections is that it forces complete neural net to act as an ensemble of sub-layers, rather than just a composition. This means later, more specialised representations still have access to the representations of earlier, more general layers, and that earlier layers are equally responsible for the final output as the later layers.

In some sense, this enforces a generalization effect on the network. This was famously visualized by Li et al. (2018) {{< cite "liVisualizingLossLandscape2018" >}}:

{{< figure-dynamic
    dark-src="./figures/resnet_loss_surfaces.png"
    light-src="./figures/resnet_loss_surfaces.png"
    alt="Loss surface of ResNet with and without residual connections."
>}}

A [ResNet](https://en.wikipedia.org/wiki/Residual_neural_network) {{< cite "heDeepResidualLearning2016" >}} with residual connections has a much smoother loss landscape than an equivalent model without those connections. This leads to easier (i.e., more convex) optimization, and usually to broader minima. This latter property is often associated with improved generalization. [DenseNet](https://pytorch.org/hub/pytorch_vision_densenet/) {{< cite "huangDenselyConnectedConvolutional2017a" >}}, a succesor to ResNet with residual connections from each layer to **all** succeeding layer, shows an even smoother, more convex loss landscape.

### Theoretical Guarantees

According to [D2L](https://d2l.ai/chapter_convolutional-modern/resnet.html#function-classes) {{< cite "zhangDiveDeepLearning2024" >}}, residual connections also give some theoretical benefits. A more powerful network *with* residual connections is guaranteed to perform at least as well as its less powerful counterparts. This is because the residual networks ensure the less powerful network is part of the more powerful one. Without residual connections, there is no such guarantee.

## Residual Connections in Transformers

In short, residual connections are a pretty good idea when training deep neural networks. It is no surprise, then, that [Transformers](https://en.wikipedia.org/wiki/Transformer_(deep_learning)) {{< cite "vaswaniAttentionAllYou2017" >}} make extensive use of residual connections. These come in two flavours: pre- and post-norm. With pre-norm the $\mathtt{LayerNorm}$ operation comes before the sublayer operations ($\mathtt{MHA}$ followed by $\mathtt{FFNN}$), whereas in post-norm architectures the $\mathtt{LayerNorm}$ comes after those sublayer operations and the addition with the residual stream.

Symbolically, we would denote this as:
$$\begin{align}
    &f^l(\mathtt{LayerNorm}(z^{l-1}))+z^{l-1} &\text{(pre-norm)} \\
    &\mathtt{LayerNorm}(f^l(z^{l-1})+z^{l-1}) &\text{(post-norm)}
\end{align}$$

Ang graphically we can depict this as:

{{< figure-dynamic
    dark-src="figures/pre_and_post_norm_dark.svg"
    light-src="figures/pre_and_post_norm_light.svg"
    alt="Pre- and post-norm residual blocks in a Transformer."
>}}

Note that in post-norm model, the residual stream is not a true residual stream, as the sub-layer output and input are scaled by the inverse standard deviation of their sum ($\sigma(z^{l-1}+z^{l})^{-1}$) after each sublayer operation. Given that each $z^{l-1}$ is $\mathtt{LayerNorm}$ normalized, we may assume that $\sigma(z^{l-1}+z^{l})>1$, and that this scaling decreases the influence of $z^{l-1}$ to the representations at subsequent layers. This can be clearly seen in Zhu et al.'s {{< cite "zhuHyperConnections2025" >}} Figure 7:
 
{{< figure-dynamic
    dark-src="figures/baseline_connection_pattern_dark.svg"
    light-src="figures/baseline_connection_pattern_light.svg"
    alt="Contribution of earlier layers to later layers in a Transformer with pre- or post-norm residual blocks."
>}}

In a pre-norm model, the contribution of earlier layers to later layers exponentially decays, resulting in each representation being consructed primarily from the closest ancestor layers. In pre-norm, which does use a 'true' residual connection, this does not happen: each layer contributes an equal amount to the representation of each later layer.

Either architecture flavour comes with its own benefits and downsides. With pre-norm, the model ensures a stable and consistent residual stream, resulting in strong gradients even in very deep models. This comes at the cost of model representation collapse. Since earlier layers essentially have an $L-l$ times greater impact on the final representations, later layers have to accomodate for this and tend to produce increasingly smaller changes to the input. As a result, the token representations between different layers start to become too similar. Post-norm models suffer from the opposite problem. The impact of earlier layers is dimished (due to the $\mathtt{LayerNorm}$) scaling, reducing the occurence of representation collapse, but at the cost of incurring vanishing gradients.

There are various other empirical trade-offs between the two model archetypes. Importantly, the macroscopic architecture of the model is a hyper-parameter determined beforehand, and remains fixed for all tasks to which we apply the pre-trained model.

## Hyper-Connections

What if we could let the model optimize the structure of residual connections? This is the guiding idea behind Hyper-Connections {{< cite "zhuHyperConnections2025" >}}.

A Hyper-Connections based-architecture augments a standard Transformer[^architecture_agnostic] with multiple residual streams which are constructed from weighted combinations of the previous layer's output and the sublayer output, allowing the model to optimize its own residual paths between layers.

[^architecture_agnostic]: Strictly speaking, Hyper-Connections seems to be architecture agnostic, althought the paper only discusses Transformers.

Specifically, an $\mathtt{HC}$ block consists of a sublayer operation (e.g., $\mathtt{MHA}$) and a residual network with $N$ separate residual streams.

To initialize the different residual streams, in the first layer ($\mathtt{HC}^1$), we simply copy the initial hidden state, $z^0$, $N$ times:

$$\mathbf{Z}^0=\begin{bmatrix}z^0 & z^0 & \ldots & z^0\end{bmatrix}\in \mathbb{R}^{N\times d}$$

The authors dub $N$ the expansion rate, and the various $\mathbf{Z}^{l}$ the Hyper Hidden Matrix (HHM). Initially, the HHM ($\mathbf{Z^0}$) just contains $N$ copies of the same input, but as the number of layers increases, this changes.

Each residual stream is allowed to transfer weighted information from itself to the input of the sublayer operation, and to each of the next layers residual streams. The sublayer operation additionally transfers its weighted output to each residual stream in the next layer. Thus, a residual network follows the graph instantiated by the following block adjacency matrix:

$$
\begin{align*}
	\mathcal{E}_{\mathtt{HC}}=\begin{pmatrix}
		\boldsymbol{0_{1\times 1}} & \boldsymbol{\beta} \\
		\boldsymbol{\alpha}_{m} & \boldsymbol{\alpha}_{r} \\
	\end{pmatrix}
\end{align*}
$$

where the first row corresponds to connections from the output of the sublayer operation ($f$) to the next layer residual streams ($\boldsymbol{\beta}\in\mathbb{R}^{1\times N}$), and later rows contain connections from the residual streams to $f$ ($\boldsymbol{\alpha}_{m}\in\mathbb{R}^{N\times 1}$), and from this layer's residual streams to the next layer's residual streams ($\boldsymbol{\alpha}_{r}\in\mathbb{R}^{N\times N}$).

This makes $\mathcal{E}_{\mathtt{HC}}$ the adjacency matrix for a graph with $N+1$ nodes. Graphically, for $N=2$, it should look something like this:

{{< figure-dynamic
    dark-src="figures/hc_n2_base_plain_dark.svg"
    light-src="figures/hc_n2_base_plain_light.svg"
    alt="The Hyper-Connections residual network graph for $N=2$."
>}}

We can equivalently represent all of this as a series of matrix products:
$$
\begin{align*}
	\mathbf{Z}^{l} &= \boldsymbol{\beta}f_{l}(\boldsymbol{\alpha_{m}}^{\intercal}\mathbf{Z}^{l-1})+\boldsymbol{\alpha}_{r}\mathbf{Z}^{l-1} \\
	\mathbf{Z}^0&=\begin{bmatrix}z^0 & z^0 & \ldots & z^0\end{bmatrix}
\end{align*}
$$

Note the similarity to the original formulation of a residual connection!

### Dynamic Hyper-Connections

So far, $\mathcal{E}_{\mathtt{HC}}$ is depedent on the training task. While the structure of the residual network will change as we change pre-training tasks, it remain input independent. Howevever, we can easily make the $\boldsymbol{\alpha}_{m}, \boldsymbol{\alpha}_{r}, \boldsymbol{\beta}$ matrices a function of $\mathbf{Z}$. This makes the macro-scopic structure of the architecture input dependent: the particular flow of information will change for each input to the model.

Zhu et al. {{< cite "zhuHyperConnections2025" >}} achieve this by adding the following weights:
$$
\begin{align*}
\bar{\mathbf{Z}}^{l} & =\mathtt{norm}(\mathbf{Z}^{l}) & \in\mathbb{R}^{N\times d} \\
\boldsymbol{\beta}^{l}(\bar{\mathbf{Z}}^{l}) & =\mathbf{s}_{\beta}^{l}\odot\mathtt{tanh}(\bar{\mathbf{Z}}^{l})+\mathbf{b}_{\beta}^{l} & \in\mathbb{R}^{N\times 1} \\
\boldsymbol{\alpha}^{l}_{m}(\bar{\mathbf{Z}}^{l}) & = \mathbf{s}_{\alpha_{m}}^{l}\odot\mathtt{tanh}(\bar{\mathbf{Z}}^{l})+\mathbf{b}_{\alpha_{m}}^{l} & \in\mathbb{R}^{N\times 1} \\
\boldsymbol{\alpha}^{l}_{r}(\bar{\mathbf{Z}}^{l})&=\mathbf{S}_{\alpha_{r}}^{l}\odot\mathtt{tanh}(\bar{\mathbf{Z}}^{l})+\mathbf{B}_{\alpha_{r}}^{l} & \in\mathbb{R}^{N\times N}
\end{align*}
$$
Note that despite the $\mathtt{tanh}\in(-1, 1)$, these matrices are unbounded due to the scaling and bias factors. THis has major implications for the stability of Hyper-Connections in deep networks.

### Results
#### The Good

The benefit of a Hyper-Connections augmented neural network seem clear. With minimal additional parameters ($N^2 + 2N$ for each layer), and an $N$ times larger memory footprint of the intermediate reprsentations[^memory_footprint], we get a substantially more topologically complex architecture that should make the model more expressive while also reducing optimization issues like vanishing or exploding gradients.

[^memory_footprint]: Note that this only affects the intermediate representations. The input to the sublayer operations (which dominate memory, especially with attention) are summed over the residual stream dimension.

But does theory translate to practise? At least according to Zhu et al. {{< cite "zhuHyperConnections2025" >}}, yes. When looking at the connectivity patterns of the Hyper-Connections network we see rich, non-uniform connections between layers, with many long-range dependencies.

{{< figure-dynamic
    dark-src="figures/hc_connectivity_pattern_dark.svg"
    light-src="figures/hc_connectivity_pattern_light.svg"
    alt="Learned connectivity patterns for different HC residual streams."
	attr="Adapted from Appendix F Figure 13 in zhuHyperConnections2025"
>}}

More importantly, when we compare the different residual streams against each other (see above figure), we see differentiation. Each stream is retaining different types of information. For example, the left most stream seems to gather information from the nearest layers, whereas the middle stream seems to prioritize the MLP layers.

Converting this to actual performance, when pre-training an OLMo-1B model augmented with an HC network, we see lower training loss for $N>2$ and fewer loss spikes. 

{{< figure-dynamic
    dark-src="figures/hc_training_loss_dark.svg"
    light-src="figures/hc_training_loss_light.svg"
    alt="The loss during training an HC augmented OLMo-1B model."
	attr="Adapted from Figure 5 in zhuHyperConnections2025"
>}}

In their appendix they show that this benefit extends to a plethora of downstream tasks, and in other domains as well.

#### The Bad

What should immediately jump out at you though, is that an HC augmented OLMo with $N=1$ actually performns *worse* than a standard residual stream OLMo. This seems counterintuitive at first. With $N=1$, the HC network is just a standard residual connection with some additional learned scaling and shifting parameters. Why would this minimal addition cause such a degradation in performance?

Zhu et al. {{< cite "zhuHyperConnections2025" >}} only expend a little ink discussing this surprising result (bottom of page 23). They suggest the fault lies with a missing conection from layer 17 to subsequent layers, indicative of vanishing gradients. This is a somewhat unsatisfying conclusion, especially considering that residual connections were supposed to mitigate this behaviour.

Could there be something more fundamental going on?

## Manifold-Constrained Hyper-Connections

Xie et al. {{< cite "xieMHCManifoldConstrainedHyperConnections2026" >}}, argue that the scaling parameters in combination with composition are the chief cause of the problem. The guiding principle behind a residual connection is that we want to leave an unaltered stream of information from the first to the last layers, to allow for a stable backpropagation path. By adding unconstrained scalars into the residual, we start to mess with this residual path.

Recall the Hyper-Connections residual network as:

$$
\mathbf{Z}^{l} = \boldsymbol{\beta}f_{l}(\boldsymbol{\alpha}_{m}^{\intercal}\mathbf{Z}^{l-1})+\boldsymbol{\alpha}_{r}\mathbf{Z}^{l-1}
$$

When we unroll this recurrence relationship, like we did with the standard residual connection, we get the following monstrosity (adapted from Xie et al. {{< cite "xieMHCManifoldConstrainedHyperConnections2026" >}} Formula 4):

$$
\begin{align*}
    \mathbf{Z}^{1} &= \boldsymbol{\beta}^{1}f^{1}((\boldsymbol{\alpha}_{m}^{1})^{\intercal}\mathbf{Z}^{0})+\boldsymbol{\alpha}^{1}_{r}\mathbf{Z}^{0} \\
    \mathbf{Z}^{2} &= \boldsymbol{\beta}^{2}f^{2}((\boldsymbol{\alpha}_{m}^{2})^{\intercal}(\boldsymbol{\beta}^{1}f^{1}((\boldsymbol{\alpha}_{m}^{1})^{\intercal}\mathbf{Z}^{0})+\boldsymbol{\alpha}^{1}_{r}\mathbf{Z}^{0})) \\
    &\quad+\boldsymbol{\alpha}^{2}_{r}(\boldsymbol{\beta}^{1}f^{1}((\boldsymbol{\alpha}_{m}^{1})^{\intercal}\mathbf{Z}^{0})+\boldsymbol{\alpha}^{1}_{r}\mathbf{Z}^{0}) \\
    &\vdots \\
    \mathbf{Z}^{l}&=\left(\prod_{i=1}^{L-1}\boldsymbol{\alpha}^{L-i}_{r}\right)\mathbf{Z}^{0}+\sum_{i=1}^{L-1}\left(\prod_{j=1}^{L-1-i}\boldsymbol{\alpha}^{L-j}_{r}\right)\boldsymbol{\beta}^{i}f^{i}\left(\left(\boldsymbol{\alpha}^{i}_{m}\right)^{\intercal}\mathbf{Z}^{i-1}\right)
\end{align*}
$$

Instead of a nice summation, we get a series of long product chains, which tend to explode or vanish, depending on the norm of $\boldsymbol{\alpha}_{r}$. This re-introduces the exact problem we tried to mitigate with residual connections in the first place. If we were to set $\boldsymbol{\alpha}_{r}=\mathbf{I}$, we recover the residual connetion, and given that $\mathbf{I}\mathbf{I}=\mathbf{I}$ the nasty product term collapses into a much friendlier summation again:

$$\mathbf{Z}^{l}=\mathbf{I}\mathbf{Z}^{0}+\sum_{i=1}^{L-1}\mathbf{I}\boldsymbol{\beta}^{i}f^{i}\left(\left(\boldsymbol{\alpha}^{i}_{m}\right)^{\intercal}\mathbf{Z}^{i-1}\right)$$

Obviously, we don't actually want to set $\boldsymbol{\alpha}_{r}=\mathbf{I}$ as it severly restricts the expressivity of our model. Instead, we can set a constraint on $\boldsymbol{\alpha}_{r}$ such that the norm of its product is equal to that of an identity matrix:

$$\|\prod_{l=1}^{L} \boldsymbol{\alpha}^{l}_{r}\|_{2}=\|\prod_{l=1}^{L} \mathbf{I}\|_{2}=1$$

This preserves the expressivity of $\boldsymbol{\alpha}_{r}$ while ensuring that the residual stream signal does not vanish or explode. But what kind of matrix would $\boldsymbol{\alpha}_{r}$ need to be to abide by this constraint?

### Doubly-Stochastic Matrix

A square matrix $\mathbf{A}\in\mathbb{R}^{N\times N}$ is [doubly-stochastic](https://en.wikipedia.org/wiki/Doubly_stochastic_matrix) if all of its elements are non-negative, and its rows **and** columns sum to 1:

$$A_{i,j}>0\wedge\sum_{i=1}^{N}\mathbf{A}_{i,j}=\sum_{j=1}^{N}\mathbf{A}_{i,j}=1\quad\forall i,j\in \{1, \ldots, N\}$$

For our purposes, doubly stochastic matrices have two very important properties:

1. **Norm**: the $1$- and $\infty$-norm $\|\mathbf{A}\|_{1}=\|\mathbf{A}\|_{\infty}=1$. More importantly, the spectral/Frobenius norm is also 1: $\|\mathbf{A}\|_2=1$. The former is trivial to prove, but the latter is much less obvious, and the mHC paper glosses over the details. I reccomend the following sources: {{< cite "jiangSpectralNormDoubly2024" >}} and {{< cite "nylenNumericalRangeDoubly1991" >}}
2. **Closure under Multiplication**: the product of two doubly-stochastic matrices is [another doubly-stochastic matrix](https://en.wikipedia.org/wiki/Doubly_stochastic_matrix#Properties). This [proof is relatively easy](https://math.stackexchange.com/a/2221823), and follows from the fact that the product of a doubly stochastic matrix preserves the 1- and $\infty$-norms.

Together, this gives us exactly the type of matrix we were looking for. If $\boldsymbol{\alpha}_{r}^{l}$ is doubly stochastic for all $l\in\{1,\ldots,L\}$, then their product has constant norm, $\|\prod_{l=1}^{L} \boldsymbol{\alpha}^{l}_{r}\|_{2}=1$.

The space of all possible doubly-stochastic matrices of size $N$ is called the Birkhoff polytope, which is an $N-1$ manifold in $N$ dimensional space. Thus, enforcing double stochasticity on  $\boldsymbol{\alpha}_{r}$ is tantamount to constraining it to the Birkhoff polytope manifold (hence, **Manifold-Constrained**).

> [!abstract]+ The Birkhoff Polytope
> The Birkhoff polytope is an $N-1$ manifold containing all $N\times N$ doubly-stochastic matrices (a [polytope](https://en.wikipedia.org/wiki/Polytope) being a geometric object with flat faces). The nodes/vertices of the polytope (i.e., the corners) represent the doubly-stochastic matrices where exactly 1 non-zero value is present in each row and column. These are the [permutation matrices](https://en.wikipedia.org/wiki/Permutation_matrix). The n-gon faces (always either triangles or rectangles) of the polytope represent convex mixtures of these permutation matrices.
> 
> The [Birkhoff-von Neumann Theorem](https://en.wikipedia.org/wiki/Doubly_stochastic_matrix#Birkhoff%E2%80%93von_Neumann_theorem) states that on the inside of the of the Birkhoff lie all possible $N\times N$ doubly-stochastic matrices.
>
> [Linear Algebra for Programmers](https://www.linearalgebraforprogrammers.com/series/permutation_sinkhorn/0_permutation_cycles) provides a very nice tutorial on permutation matrices and the Birkhoff-von Neumann theorem.

### Sinkhorn's Theorem

[Sinkhorn's theorem](https://en.wikipedia.org/wiki/Sinkhorn%27s_theorem) states that for every $N\times N$ matrix with positive elements, there exist two diagonal matrices ($\operatorname{diag}_{N}(d_{1})$ and $\operatorname{diag}_{N}(d_{2})$) with strictly positive elements such that $\operatorname{diag}_{N}(d_{1})\mathbf{A}\operatorname{diag}_{N}(d_{1})$ is a doubly stochastic matrix.

Thus, if we want to convert our $\boldsymbol{\alpha}_{r}$ to a doubly stochastic matrix, we simply need to find suitable $d_{1}$ and $d_{2}$ values. Luckily, this is not so difficult, as [Sinkhorn-Knopp](https://en.wikipedia.org/wiki/Sinkhorn%27s_theorem#Sinkhorn%E2%80%93Knopp_algorithm) provide a very simple algorithm. They simply divide the matrix by the row sums, then by the column sums, and iterate until the matrix is doubly stochastic.

```python
def sinkhorn_knopp(
	A: Float[np.ndarray, "N N"],
) -> Float[np.ndarray, "N N"]:
	while True:
		row_sum = np.sum(A, axis=1, keepdims=True)
		
		A /= row_sum
		
		col_sum = np.sum(A, axis=0, keepdims=True)
		
		A /= col_sum
		
		# Check for convergence
		...
	
	return A
```

This is also known as [Iterative Proportional Fitting](https://en.wikipedia.org/wiki/Iterative_proportional_fitting), and is guaranteed to converge.

Thus, to convert an arbitrary matrix to a doubly stochastic one (i.e., constraining it to the Birkhoff polytope), the Xie et al. {{< cite "xieMHCManifoldConstrainedHyperConnections2026" >}} suggest an exponentiation followed by $20$ iterations of the Sinkhorn-Knopp algorithm:

$$
\begin{align*}
	^{+}\boldsymbol{\alpha}_{r}^{l}(\bar{\mathbf{Z}}^l)&=\exp(\boldsymbol{\alpha}_{r}^{l}(\bar{\mathbf{Z}}^l)) \\
	^{\text{DS}}\boldsymbol{\alpha}_{r}^{l}(\bar{\mathbf{Z}}^l)&=\mathtt{SinkhornKnopp}(^{+}\boldsymbol{\alpha}_{r}^{l}(\bar{\mathbf{Z}}^l), 20) 
\end{align*}
$$

This is an inherently sequential algorithm, and does not play well with the GPU. Xie et al. {{< cite "xieMHCManifoldConstrainedHyperConnections2026" >}} managed to fit all of this in a single kernel however, keeping the latency minimal (see Section 4.3).

All in all, going from HC to mHC is as simple as[^probability_simplex]:

[^probability_simplex]: the sigmoid operations, $\sigma$, used for $\boldsymbol{\alpha}_{m}$ and $\boldsymbol{\beta}$ are also types of manifold constraints. Specifically, they constrain the output to the [probability simplex](https://en.wikipedia.org/wiki/Simplex), which is an $N-1$ manifold where the sum of each vector sums to 1

$$
\begin{align*}
    \boldsymbol{\beta}^{l}_{\text{mHC}}(\bar{\mathbf{Z}}^{l})&=\sigma(\boldsymbol{\beta}^{l}_{\text{HC}}(\bar{\mathbf{Z}}^{l})) \\
    \boldsymbol{\alpha}^{l}_{m, \text{mHC}}(\bar{\mathbf{Z}}^{l}) &= 2\sigma(\boldsymbol{\alpha}^{l}_{m, \text{HC}}(\bar{\mathbf{Z}}^{l})) \\
    \boldsymbol{\alpha}^{l}_{r,\text{mHC}}(\bar{\mathbf{Z}}^{l}) &= \mathtt{sinkhorn\_knopp}(\exp\{\boldsymbol{\alpha}^{l}_{r,\text{HC}}(\bar{\mathbf{Z}}^{l})\}) \\
\end{align*}
$$

## Results

The benefits of the manifold-constraints are immediately obvious. In the following figure, Xie et al. {{< cite "xieMHCManifoldConstrainedHyperConnections2026" >}} compare the magnitude of the maximum element in the cumulative (product) forward and backward signals. On the left facet are the standard HC connections (unconstrained), and on the right are the mHC connections. Where HC is prone to exploding representations and especially gradients, the mHC representations remain more or less constant (note the logarithmic axis on the left).

{{< figure-dynamic
    dark-src="figures/mhc_forward_backward_gain_dark.svg"
    light-src="figures/mhc_forward_backward_gain_light.svg"
    alt="The maximum absolute value of the forward and backward signal in an HC or mHC model."
	attr="Adapted from Figures 3 & 7 in xieMHCManifoldConstrainedHyperConnections2026"
	width=1000
>}}

We can see this even more clearly when inspecting the $\boldsymbol{\alpha}_{r}^{l}$ and $\prod_{l=1}^{L}\boldsymbol{\alpha}_{r}^{l}$ matrices directly.

{{< figure-dynamic
    dark-src="figures/mhc_heatmap_dark.svg"
    light-src="figures/mhc_heatmap_light.svg"
    alt="Various residual connection matrices and their products."
	attr="Adapted from Figure 8 in xieMHCManifoldConstrainedHyperConnections2026"
	width=1000
>}}

In the figure above, the top row has these matrices for HC, and the bottom row corresponds to mHC. The first three columns provide $\boldsymbol{\alpha}_{r}^{l}$ at different layers (1, 30, 60) and the right three columns provide several products of $\boldsymbol{\alpha}_{r}^{l}$, (1-30, 30-61, 1-61).

The HC residual connection matrices are unbounded and tend to be dominated by a single residual stream. These problems are exacerbated in the HC model, with large absolute values, dominated by one or two residual streams.

With mHC residual connections, we do not see this. The individual $\boldsymbol{\alpha}_{r}^{l}$ remain bounded are tend to look more like identity matrices. Furthermore, their product remains similarly bounded, and sees a more uniform distribution of residual connections. This latter finding suggests that the model is mixing the different residual streams more effectively.

Of course, at the end of the day, what we care about most is the actual performance of an mHC augmented LLM. 

{{< figure-dynamic
    dark-src="figures/mhc_loss_grad_dark.svg"
    light-src="figures/mhc_loss_grad_light.svg"
    alt="The loss during training an HC augmented OLMo-1B model."
	attr="Adapted from Figure 8 in xieMHCManifoldConstrainedHyperConnections2026"
	width=1000
>}}

The answer here is another resounding yes. The loss is smaller throughout training and especially at later stages where the HC model tends towards the baseline. This occurs right around the time that the gradients of the HC model diverges, while the mHC gradients tend to follow much more stable pattern indicative of convergence. Additionally, we also see far fewer gradient spikes, implying that the training process is made more robust.

## Conclusion

I had a ton of fun reading and presenting these papers. They use conceptually elegant techniques to develop a more powerful architecture with minimal additional computation overhead. I also feel like it's a natural extension for Transformers, where the sequential connection are already data dependent due to MHA blocks. Now, the connections between layers are (to a certain extent) also data dependent. So in effect, both the micro- and macro-architecture or an HC augmented Transformer changes with each input.

It wouldn't surprise me if we'd see a lot of Hyper-Connections start to appear going forward. I wonder if we could take the idea of treating the macro-architecture as a network to further extremes. What if we made the same extension to HC that Densenets made to ResNets? What if instead of a stack of sequential layers, we'd have a flat graph, where each 'layer' is able to send input and output to each other 'layer'?

Another application I'm excited about is the use of mHC as a PEFT technique. Take a pre-trained LLM, and simply fine-tune the Hyper-Connections residual network around it. This would be *extremely* parameter efficient, and easily combined with other PEFT techniques.

## References

{{< references >}}

## Changelog

```
[2026-01-19] First draft
[2026-02-12] Presentation added
[2026-02-14] Finished the write up
[2026-02-15] Added dark mode figures
```
