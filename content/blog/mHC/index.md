---
title: Manifold-Constrained Hyper-Connections
date: 2026-01-19
tags:
  - machine-learning
math: true
draft: true
---

{{< toc >}}

> [!note]+ Notation
> In this article, I've tried to align the notation between the different sections. This can make it more difficult to go from article to article. Here's a conversion table:
> 
>| Name                      |       This Article        |        HC        |             mHC             |
| :------------------------ | :-----------------------: | :--------------: | :-------------------------: |
| Hidden state              |            $z$            |       $h$        |             $x$             |
| Hyper Hidden Matrix       |       $\mathbf{Z}$        |   $\mathbf{H}$   |        $\mathbf{X}$         |
| Operation/Function        |            $f$            |  $\mathcal{T}$   |        $\mathcal{F}$        |
| Residual connections      | $\boldsymbol{\alpha}_{r}$ | $\mathbf{A}_{r}$ | $\mathcal{H}^{\text{res}}$  |
| $z^{l}$-$f$ connections   | $\boldsymbol{\alpha}_{m}$ | $\mathbf{A}_{m}$ | $\mathcal{H}^{\text{pre}}$  |
| $f$-$z^{l+1}$ connections |   $\boldsymbol{\beta}$    |   $\mathbf{B}$   | $\mathcal{H}^{\text{post}}$ |

## Residual Connections

A residual connection, also known as a skip connection, adds to a layer's output the identity function transformed input:
$$z^{l}=f(z^{l-1})+z^{l-1}$$

Applied in succession, this gives:
$$
\begin{align*}
	z^{(1)}&=f_{1}(x)+x \\
	z^{(2)}&=f_{2}(f_{1}(x)+x)+f_{1}(x)+x \\
	z^{(3)}&=f_{3}(f_{2}(f_{1}(x)+x)+f_{1}(x)+x)+f_{2}(f_{1}(x)+x)+f_{1}(x)+x \\
	\vdots
\end{align*}$$

In other words, the representation at layer $l$ is constructed from [the sum of the previous $l$ layer representations and the application of $f^{l}$ to $z^{l-1}$](https://en.wikipedia.org/wiki/Residual_neural_network#Forward_propagation):
$$
\begin{align*}
	z^{l}&=x+\sum_{i=1}^{l-1}f_{i}(z^{(i-1)}) \\
	&x=z^{(0)}
\end{align*}
$$

### Advantages of Residual Connections
#### Stable Gradients

When we apply the gradient function to the above recurrence relation, we see that each preceding layer contributes receives a strong gradient signal directly from the final layer representation:
$$
\begin{align*}
	\frac{\partial }{\partial x}\mathcal{L}(z^{l}) &= \frac{\partial}{\partial z^{l}}\mathcal{L}(z^{l})\frac{\partial}{\partial x}z^{l} & \text{(chain rule)}\\
	&= \frac{\partial}{\partial z^{l}}\mathcal{L}(z^{l})\left(\frac{\partial}{\partial x} x +\frac{\partial}{\partial x}\sum_{i=1}^{L-1}f_{i}(z^{(i-1)})\right) & \text{(linearity)}\\
	&= \frac{\partial}{\partial z^{l}}\mathcal{L}(z^{l}) + \frac{\partial}{\partial z^{l}}\mathcal{L}(z^{l})\sum_{i=1}^{L-1}\frac{\partial}{\partial x}f_{i}(z^{(i-1)}) & \text{(linearity)}\\
	&= \frac{\partial}{\partial z^{l}}\mathcal{L}(z^{l}) + \frac{\partial}{\partial z^{l}}\mathcal{L}(z^{l})\left(\frac{\partial}{\partial x}f_{1}(x) + \frac{\partial}{\partial x}f_{2}(z_{1})+\ldots\right)
\end{align*}
$$

Thus, residual connections mitigate the effect of [vanishing gradients](https://en.wikipedia.org/wiki/Vanishing_gradient_problem) in (arbitrarily) deep neural networks. This intuition led to the [ResNet](https://en.wikipedia.org/wiki/Residual_neural_network) architecture, which remains one of the most highly cited deep learning papers ever, and heralded in the new deep learning age {{< cite "heDeepResidualLearning2016" >}}.

#### Ensembles of Networks & Generalization

Another suspected benefit of residual connections is that it forces all layers in the networks to act as an ensemble of networks, rather than just a composition. This means later, more specialised layer representations still contain the representation from earlier, more general layers.

In some sense, this enforces a generalization effect on the network. This was famously visualized by Li et al. (2018) {{< cite "liVisualizingLossLandscape2018" >}}:

{{< figure-dynamic
    dark-src="./figures/resnet_loss_surfaces.png"
    light-src="./figures/resnet_loss_surfaces.png"
    alt="Loss surface of ResNet with and without residual connections."
>}}

A ResNet model with residual connections have a much smoother loss landscape than an equivalent model without those connections. This leads to easier (i.e., more convex) optimization, and usually to broader minima. This latter property is often associated with improved generalization. [DenseNet](https://pytorch.org/hub/pytorch_vision_densenet/) {{< cite "huangDenselyConnectedConvolutional2017a" >}}, a succesor to ResNet with residual connections from each layer to **all** succeeding layer, shows an even smoother, more convex loss landscape.

## Pre-norm or Post-norm: Residual Connections in Transformers

Residual connections in [Transformers](https://en.wikipedia.org/wiki/Transformer_(deep_learning)) {{< cite "vaswaniAttentionAllYou2017" >}} come in two flavours: pre- and post-norm. With pre-norm the $\mathtt{LayerNorm}$ operation comes before the sublayer operations ($\mathtt{MHA}$ followed by $\mathtt{FFNN}$), whereas in post-norm architectures the $\mathtt{LayerNorm}$ comes after those sublayer operations and the addition with the residual stream.

Symbolically, we would denote this as:
$$\begin{align}
    &f(\mathtt{LayerNorm}(z^{l-1}))+z^{l-1} &\text{(pre-norm)} \\
    &\mathtt{LayerNorm}(f(z^{l-1})+z^{l-1}) &\text{(post-norm)}
\end{align}$$

Ang graphically we can depict this as:

{{< figure-dynamic
    dark-src="figures/pre_and_post_norm.svg"
    light-src="figures/pre_and_post_norm.svg"
    alt="Pre- and post-norm residual blocks in a Transformer."
>}}

Note that in pre-norm, the residual stream is never sent through a $\mathtt{LayerNorm}$ layer, whereas with post-norm, the residual stream is re-scaled twice within each attention block due to the $\mathtt{LayerNorm}$ inverse standard deviation scaling. 

Both architecture flavours have their benefits and downsides. With pre-norm, the model ensures a stable and consistent residual stream, resulting in strong gradients even in very deep models. This comes at the cost of model representation collapse. Since earlier layers essentially have an $L-l$ times greater impact on the final representations, later layers have to accomodate for this and tend to produce increasingly smaller changes to the input. As a result, the token representations between different layers start to become too similar. Post-norm models suffer from the opposite problem. The impact of earlier layers is dimished (due to the $\mathtt{LayerNorm}$) scaling, reducing the occurence of representation collapse, but at the cost of incurring vanishing gradients.

This can be seen clearly in Figure 7 of the HC paper, which shows the contribution of earlier layers to the representation of later layers:

{{< figure-dynamic
    dark-src="figures/pre_and_post_norm_contributions.png"
    light-src="figures/pre_and_post_norm_contributions.png"
    alt="Contribution of earlier layers to later layers in a Transformer with pre- or post-norm residual blocks."
>}}

There are various other empirical trade-offs, as well as a large number of compromises between the two model archetypes. Regardless, the macroscopic architecture of the model is determined beforehand as a hyperparameter, and is thus not optimized during training.

## Hyper-Connections

What if we could let the model optimize the structure of residual connections? This is the guiding idea behind Hyper-Connections.

The Hyper-Connections architecture takes a standard Transformer, but allows the model to optimize its own residual paths between layers. Specifically, an $\mathtt{HC}$ block consists of a sublayer operation (e.g., $\mathtt{MHA}$) and a residual network with $N$ separate residual streams around that sublayer operation. The residual network is the graph instantiated by the following block adjacency matrix:

$$
\begin{align*}
	\mathcal{E}_{\mathtt{HC}}=\begin{pmatrix}
		\boldsymbol{0_{1\times 1}} & \boldsymbol{\beta} \\
		\boldsymbol{\alpha}_{m} & \boldsymbol{\alpha}_{r} \\
	\end{pmatrix}
\end{align*}
$$

where the first row corresponds to connections from the output of the sublayer's operation, to the $\mathtt{HC}$ block output, and the remaining rows contain: ($\boldsymbol{\alpha}_{m}$) connections from the inputs to $f$, and ($\boldsymbol{\alpha}_{r}$) from the inputs to $N$ separate residual streams.

To initialize the different residual streams, in the first layer ($\mathtt{HC}^1$), we simply copy the initial hidden state, $z^0$, $N$ times:

$$\mathbf{Z}^0=\begin{bmatrix}z^0 & z^0 & \ldots & z^0\end{bmatrix}\in \mathbb{R}^{N\times d}$$

The authors dub $N$ the expansion rate, and the various $\mathbf{Z}^{l}$ the Hyper Hidden Matrix (HHM). Initially, the HHM ($\mathbf{Z^0}$) just contains $N$ copies of the same input, but as the number of layers increases, this changes.

Thus, $\boldsymbol{\beta}$ is a (transposed) $N\times 1$ vector, $\boldsymbol{\alpha}_{m}$ is another $N\times 1$ vector, but $\boldsymbol{\alpha}_{r}$ is an $N\times N$ matrix. This makes $\mathcal{E}_{\mathtt{HC}}$ the adjacency matrix for a graph with $N+1$ nodes. Graphically, for $N=2$, we can represent the whole unrolled structure as this:

{{< figure-dynamic
    dark-src="figures/hc_n2_base_plain.svg"
    light-src="figures/hc_n2_base_plain.svg"
    alt="The Hyper-Connections residual network graph for $N=2$."
>}}

Each node/row of the HHM can contribute both to the input of the layer's operation, $f^{l}$, as well as each of the next HHM nodes/rows. The strength of the association between the different nodes is modulated by the weights in $\mathcal{E}_{\mathtt{HC}}$.

We can equivalently represent all of this as a series of matrix products:
$$
\begin{align*}
	\mathbf{Z}^{l} &= \boldsymbol{\beta}f_{l}(\boldsymbol{\alpha_{m}}^{\intercal}\mathbf{Z}^{l-1})+\boldsymbol{\alpha}_{r}\mathbf{Z}^{l-1} \\
	\mathbf{Z}^0&=\begin{bmatrix}z^0 & z^0 & \ldots & z^0\end{bmatrix}
\end{align*}
$$

Note the similarity to the original formulation of a residual connection!

### Dynamic Hyper-Connections

Rather than leaving $\mathcal{E}_{\mathtt{HC}}$ static across layers and/or across training, we can also learn the optimal architecture for each input. Specifically, we can make the matrices $\boldsymbol{\beta}^{l}$, $\alpha_{r}^l$, $\alpha_{m}^l$ a function of the input:
$$
\begin{align*}
\bar{\mathbf{Z}}^{l}&=\mathtt{norm}(\mathbf{Z}^{l}) & \in\mathbb{R}^{N\times d} \\
\boldsymbol{\beta}^{l}(\bar{\mathbf{Z}}^{l})&=\mathbf{s}_{\beta}^{l}\odot\mathtt{tanh}(\bar{\mathbf{Z}}^{l})+\mathbf{b}_{\beta}^{l} & \in\mathbb{R}^{N\times 1} \\
\boldsymbol{\alpha}^{l}_{m}(\bar{\mathbf{Z}}^{l}) &= \mathbf{s}_{\alpha_{m}}^{l}\odot\mathtt{tanh}(\bar{\mathbf{Z}}^{l})+\mathbf{b}_{\alpha_{m}}^{l} & \in\mathbb{R}^{N\times 1} \\
\boldsymbol{\alpha}^{l}_{r}(\bar{\mathbf{Z}}^{l})&=\mathbf{S}_{\alpha_{r}}^{l}\odot\mathtt{tanh}(\bar{\mathbf{Z}}^{l})+\mathbf{B}_{\alpha_{r}}^{l} &\in\mathbb{R}^{N\times N}
\end{align*}
$$
Note that despite the $\mathtt{tanh}$, these matrices are unbounded due to the scaling and bias factors.

### The Good & The Bad

With minimal additional parameters, assuming $N\ll d$ and $N\ll T$, the Transformer architecture with Hyper-Connections becomes **much** more topologically complex. Since we're summing over the weighted inputs, the sublayer operations ($\mathtt{MHA}$, $\mathtt{FFNN}$, $\mathtt{LayerNorm}$) all act the same, and dominate the computational complexity of the model.

Despite this, the intermediate states are more complex, each layer can receive different combinations of input, and the specific form of input into a layer is a function of the properties of the input.

==TODO: add the downsides of unconstrained Hyper-Connections ==

## Manifold-Constrained Hyper-Connections

### Doubly-Stochastic Matrix

A matrix $\mathbf{A}\in\mathbb{R}^{N\times N}$ is [doubly-stochastic](https://en.wikipedia.org/wiki/Doubly_stochastic_matrix) if all of its rows **and** all of its columns sum to 1:
$$\sum_{i=1}^{N}\mathbf{A}_{i,j}=\sum_{j=1}^{N}\mathbf{A}_{i,j}=1$$

Let $\boldsymbol{1}$ be an $N\times 1$ column vector of ones. Then, we may equivalently state:
$$
\begin{align*}
	\mathbf{A}\boldsymbol{1}=\boldsymbol{1}~\wedge \boldsymbol{1}^{\intercal}\mathbf{A}=\boldsymbol{1}^{\intercal}
\end{align*}
$$
A doubly-stochastic matrix is per-definition square and has as 1- and $\infty$-norm $\|\mathbf{A}\|_{1}=\|\mathbf{A}\|_{\infty}=1$. Additionally, the spectral/Frobenius norm is also 1: $\|\mathbf{A}\|_2=1$. This property is much less obvious, and the mHC paper glosses over the details. I reccomend the following sources for finding a proof: {{< cite "jiangSpectralNormDoubly2024" >}} and {{< cite "nylenNumericalRangeDoubly1991" >}}

The space of all possible doubly-stochastic matrices of size $N$ is called the Birkhoff polytope.

> [!abstract]+ The Birkhoff Polytope
> The Birkhoff polytope is an $N-1$ manifold containing all $N\times N$ doubly-stochastic matrices (a [polytope](https://en.wikipedia.org/wiki/Polytope) being a geometric object with flat faces). The nodes/vertices of the polytope (i.e., the corners) represent the doubly-stochastic matrices where exactly 1 non-zero value is present in each row and column. These are the [permutation matrices](https://en.wikipedia.org/wiki/Permutation_matrix). The n-gon faces (always either triangles or rectangles) of the polytope represent convex mixtures of these permutation matrices.
> 
> The [Birkhoff-von Neumann Theorem](https://en.wikipedia.org/wiki/Doubly_stochastic_matrix#Birkhoff%E2%80%93von_Neumann_theorem) states that on the inside of the of the Birkhoff lie all possible $N\times N$ doubly-stochastic matrices.
>
> [Linear Algebra for Programmers](https://www.linearalgebraforprogrammers.com/series/permutation_sinkhorn/0_permutation_cycles) provides a very nice tutorial on permutation matrices and the Birkhoff-von Neumann theorem.

#### Properties
##### Norms of Matrix Products
The spectral norm of a product of matrices is [sub-multiplicative](https://mathworld.wolfram.com/MatrixNorm.html):
$$\|\mathbf{A}\mathbf{B}\|_2\leq\|\mathbf{A}\|_2\|\mathbf{B}\|_2$$
Then, if $A$ is a doubly stochastic matrix, we may assume that the spectral norm of $\mathbf{A}\mathbf{B}$ is bounded by the spectral norm of $\mathbf{B}$. In other words, the operation $\mathbf{A}\mathbf{B}$ is at the very least non-expansive. Within the context of Transformers, this suggests no more exploding gradients, and a mitigation of representation collapse.

> [!danger] 
> The mHC paper suggests that the mapping $\mathbf{A}\mathbf{B}$ is norm preserving (and thus also not contractive). I haven't been able to find a proof of this property for the Frobenius norm $\|\mathbf{A}\mathbf{B}\|_2$.

##### Closure under Matrix Multiplication

The matrix product of two doubly-stochastic matrices is itself a doubly-stochastic matrix. This [proof is relatively easy](https://math.stackexchange.com/a/2221823), and follows from the fact that the product of a doubly stochastic matrix preserves the 1- and $\infty$-norms.

Within the context of DNNs, this means that a composition of doubly stochastic matrices (i.e., resulting in $\prod_{l=1}^{L}\mathbf{A}^l$) will still yield a doubly stochastic matrix, preserving the properties of the operation.

### Constraining $\boldsymbol{\alpha}_{r}$ to the Birkhoff Polytope

[Sinkhorn's theorem](https://en.wikipedia.org/wiki/Sinkhorn%27s_theorem) states that for every $N\times N$ matrix with positive elements ($\mathbf{A}_{i,j}\geq0 \forall i,j \in (1, N)$), there exist two diagonal matrices ($\operatorname{diag}_{N}(d_{1})$ and $\operatorname{diag}_{N}(d_{2})$) with strictly positive elements such that $\operatorname{diag}_{N}(d_{1})\mathbf{A}\operatorname{diag}_{N}(d_{1})$ is a doubly stochastic matrix.

All well and good, but how do we find those diagonal matrices? Luckily, Sinkhorn-Knopp provide a very simple algorithm. Specifically, they simply divide the matrix by the row sums, then by the column sums, and continue until the matrix is doubly stochastic.

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

This is also known as [Iterative Proportional Fitting](https://en.wikipedia.org/wiki/Iterative_proportional_fitting), and is guaranteed to converge to a doubly-stochastic matrix.

Thus, to convert an arbitrary matrix to a doubly stochastic one (i.e., constraining it to the Birkhoff polytope), the mHA authors suggest an exponentiation followed by $20$ iterations of the Sinkhorn-Knopp algorithm:
$$
\begin{align*}
	^{+}\boldsymbol{\alpha}_{r}^{l}(\bar{\mathbf{Z}}^l)&=\exp(\boldsymbol{\alpha}_{r}^{l}(\bar{\mathbf{Z}}^l)) \\
	^{\text{DS}}\boldsymbol{\alpha}_{r}^{l}(\bar{\mathbf{Z}}^l)&=\mathtt{SinkhornKnopp}(^{+}\boldsymbol{\alpha}_{r}^{l}(\bar{\mathbf{Z}}^l), 20) 
\end{align*}
$$

## Results

## References

{{< references >}}
