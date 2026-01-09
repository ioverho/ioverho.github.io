---
title: The Beta Distribution
date: 2019-05-15
tags:
  - math
  - statistics
math: true
---

{{< toc >}}

This article is a rewrite of an undergrad assignment that has proven popular. The original can still be found [here](../../unlisted/theory%20of%20statistics%20and%20data%20analysis.pdf).

## Introduction

The Beta distribution, $\operatorname{Beta}(x;\alpha,\beta)$, parameterized by the positive real parameters $\alpha$ and $\beta$, is a univariate continuous probability density function defined on the interval $[0,1]$. Due to its existence between upper and lower bounds, it easily lends itself to applications where the variable is limited to intervals themselves (e.g., rates, probabilities).

Applications of the Beta distribution are many and varied. Its frequent use can be partially motivated by its innate flexibility, with special cases including the Uniform, Power, Gamma, F, $\chi^2$ and exponential distributions; for the [generalized Beta distribution](https://en.wikipedia.org/wiki/Generalized_beta_distribution), this list of special cases becomes much longer {{< cite "mcdonaldGeneralizationBetaDistribution1995" >}}. For large and equal values of $\alpha$ and $\beta$ the Beta function approximates a normal distribution {{< cite "krishnamoorthyHandbookStatisticalDistributions2016" >}}. The CDF of the Beta distribution is further tied to the binomial, negative binomial and Student's t distributions {{< cite "walckHandbookStatisticalDistributions1996" >}}. In fact, the distribution is so popular, that it has its own simple to compute approximation; [the Kumaraswamy distribution](https://en.wikipedia.org/wiki/Kumaraswamy_distribution).

One application of note is its role within Bayesian statistics. Herein the Beta distribution is applied as a conjugate prior for the probabilities of Bernoulli trials {{< cite "krishnamoorthyHandbookStatisticalDistributions2016" >}}. Herein, the parameters are defined such that $\alpha$ represents the number of successes and $\beta$ the number of failures in the $N$ preceding trials ($\alpha+\beta=N$).

This article provides an in-depth overview of the Beta distribution. First, I define the Beta distribution from first principles by discussing the Gamma and Beta functions. This is followed by a section discussing important properties and moments, like its normalization, mode, mean, variance and moment generating function. I discuss two parameter estimation techniques theoretically, before empirically verifying their unbiasedness, consistency and relative efficiency.

## Definition

### The Gamma Function

Consider the discrete function of factorials such that it maps the points 1, 2, 3, 4, 5 to their factorial 1, 2, 6, 24, 120: $f(n)=n!$. What is the continuous function that interpolates this discrete function?

This problem was first considered and solved by Leonhard Euler, however the modern notation used is attributed to Adrien Marie Legendre. The solution is the Gamma function:
$$\Gamma(z)=(z-1)!=\int^\infty_0x^{z-1}e^{-x}dx$$

{{< figure-dynamic
    dark-src="./figures/factorial_interporlation_dark.svg"
    light-src="./figures/factorial_interporlation_light.svg"
    alt="Interpolation of the factorials using the Gamma function."
>}}

The most important property of the Gamma function is its definition through a recurrence relation. By applying integration by parts to the Gamma function, the value of $\Gamma(z+1)$ can be expressed in terms of $\Gamma(z)$.

$$
\begin{gather}
    \Gamma(z)=\int^\infty_0x^{z-1} e^{-x}dx \\
    \begin{bmatrix}
        {\color{red} f(x)=e^{-x}}&\implies& {\color{orange} F(x)=-e^{-x}}\\
        {\color{green} G(z)= x^{z-1}}&\implies&{\color{teal} g(z)=(z-1)x^{z-2}}
    \end{bmatrix} \\
    \Gamma(z)=\underbrace{{\color{orange}-}{\color{green} x^{z-1}}{\color{orange}e^{-x}}\bigr\rvert^\infty_0}_{(1)}+\underbrace{{\color{teal}(z-1)}\int^\infty_0{\color{teal}x^{z-2}}{\color{red} e^{-x}}dx}_{(2)}
\end{gather}
$$

Note that the first term (1) tends to 0 for both the upper and lower integration bounds. The integral in the second term is simply the Gamma function evaluated at $(z-1)$, the recurrence relation is found.
$$
    \Gamma(z)=(z-1)\Gamma(z-1)\implies \Gamma(z+1)=z\Gamma(z)
$$

### The Beta Function

The Beta function is very closely related to the Gamma function, in both origin and definition. In fact, Euler first derived the Beta function before coming to the Gamma function. In Legendre notation {{< cite "davisLeonhardEulersIntegral1959" >}}.
$$
    \operatorname{B}(\alpha,\beta)=\int^1_0x^{\alpha-1}(1-x)^{\beta-1}dx
$$

[We can rewrite the Beta function as a quotient of Gamma functions evaluated at $\alpha$ and $\beta$](https://mathworld.wolfram.com/BetaFunction.html).
$$
    \operatorname{B}(\alpha,\beta)=\frac{\Gamma(\alpha)\Gamma(\beta)}{\Gamma(\alpha+\beta)}
$$

> [!Note]- Proof (Beta-Gamma function relation)
> Consider the definition of the Gamma function as above. The product of two Gamma functions is then,
> $$
\begin{align*}
    \Gamma(\alpha)\Gamma(\beta)&=\int^\infty_0x_1^{\alpha-1}e^{-x_1}dx_1\int^\infty_0x_2^{\beta-1}e^{-x_2}dx_2 \\
    &=\int^\infty_0\int^\infty_0e^{-(x_1+x_2)}x_1^{\alpha-1}x_2^{\beta-1}dx_1dx_2
\end{align*}
$$
> Rather than evaluate this integral, we apply a transformation. Consider the variables:
> $$
\begin{align*}
    y_1&=x_1+x_2 \\
    y_2&=\frac{x_1}{x_1+x_2}
\end{align*}
$$
> Rewrite the variable $x_1=y_1y_2$ and $x_2=y_1(1-y_2)$. This immediately allows one to simplify the sum of $x_1$ and $x_2$ as, $x_1+x_2=y_1y_2+y_1(1-y_2)=y_1$. The Jacobian of this transformation follows as,
> $$
\begin{align*}
    J&=\begin{bmatrix}
    \frac{\partial x_1}{\partial y_1}&&\frac{\partial x_1}{\partial y_2}\\
    \frac{\partial x_2}{\partial y_1}&&\frac{\partial x_2}{\partial y_2}
    \end{bmatrix}
    =
    \begin{bmatrix}
    y_2&&y_1\\
    1-y_2&&-x_1
    \end{bmatrix}
\end{align*}
$$
> The absolute determinant of the Jacobian follows as $|-y_2\cdot y_1-y_1(1-y_2)|=|-y_1|=y_1$. This allows rewriting from $x_1$ and $x_2$ into $y_1$ and $y_2$ as per the change of variables theorem {{< cite "hoggIntroductionMathematicalStatistics2019" >}}. This theorem states that,
> $$
\begin{align*}
    &\int\int_A f(x_1,x_2)dx_1dx_2=\\
    &\int\int_Bf(w_1(y_1,y_2),w_2(y_1,y_2))|J|dy_1dy_2
\end{align*}
$$
> Returning to the product of two Gamma functions, it follows that (in terms of $x$ and $y$) this yields,
> $$
\begin{align*}
    \Gamma(\alpha)\Gamma(\beta)&=
    \int_{Y_2}\int_{Y_1} e^{-y_1}(y_1y_2)^{\alpha-1}(y_1(1-y_2))^{\beta-1}y_1dy_1dy_2\\
    &=\int_{Y_2}y_2^{\alpha-1}(1-y_2)^{\alpha-1}dy_2\int_{Y_1} e^{-y_1}y_1^{\alpha-1}y_1^{\beta-1}y_1dy_1
\end{align*}
$$
> Note that the functions $\Gamma(\alpha)$ and $\Gamma(\beta)$ are both positive. This requires $Y_2$ to be bounded between $[0,1]$, while $Y_1$ can take any value between $[0,\infty]$. The integration ranges of the above integral then follow as,
> $$
\begin{align}
    \Gamma(\alpha)\Gamma(\beta)&=\int_0^1y_2^{\alpha-1}(1-y_2)^{\beta-1}dy_2\int_0^\infty e^{-y_1}y_1^{\alpha-1}y_1^{\beta-1}y_1dy_1\\
    &=\underbrace{\int_0^1y_2^{\alpha-1}(1-y_2)^{\beta-1}dy_2}_{\operatorname{B}(\alpha, \beta)}\underbrace{\int_0^\infty e^{-y_1}y_1^{\alpha+\beta}dy_1}_{\Gamma(\alpha+\beta)}
\end{align}
$$
> Here the left integral is equivalent to $B(\alpha,\beta)$, while the right integral is equivalent to $\Gamma(\alpha+\beta)$. The product of two Gamma functions therefore follows as,
> $$
\Gamma(\alpha)\Gamma(\beta)=B(\alpha,\beta)\Gamma(\alpha+\beta)\implies B(\alpha,\beta)=\frac{\Gamma(\alpha)\Gamma(\beta)}{\Gamma(\alpha+\beta)}
$$

Where the Gamma function interpolates the factorial function, the Beta function is [proportional to the inverse binomial coefficients at integer valued $\alpha$ and $\beta$](https://en.wikipedia.org/wiki/Beta_function#Properties).

### The Beta Distribution

The Beta distribution can be defined as:
$$
\begin{align*}
    \operatorname{Beta}(x;\alpha,\beta)&=\frac{x}{x+y}\\
    &x\sim \operatorname{Gamma}(\alpha,\theta)\\
    &y\sim\operatorname{Gamma}(\beta,\theta)
\end{align*}
$$

The Gamma distribution is defined as,
$$
    \operatorname{Gamma}(x;\alpha,\theta)=\frac{1}{\Gamma(\alpha)\theta^\alpha}x^{\alpha-1}e^{-x/\theta}
$$
where $\alpha$ is commonly called the shape parameter, and $\beta$ the scale parameter. The Gamma distribution is a heavily left-skewed distribution defined over the positive reals.

> [!NOTE]- Proof (Beta-Gamma distribution relation)
> The derivation of the Beta distribution is very similar to the Beta-Gamma relation presented above. However, instead of using Gamma functions, we use Gamma distributions.
>
> Consider two *independent* Gamma distributed random variables, $x_1$ and $x_2$, whose product distribution is given as,
> $$
\begin{align*}
    f(x_1,x_2)&=\operatorname{Gamma}(x;\alpha,1)\operatorname{Gamma}(x;\beta,1) \\
    &=\frac{1}{\Gamma(\alpha)\Gamma(\beta)}x_1^{\alpha-1}x_2^{\beta-1}e^{-x_1-x_2}
\end{align*}
$$
> We use the following transformation:
> $$
\begin{align*}
    y_1&=x_1+x_2 \\
    y_2&=\frac{x_1}{x_1+x_2}
\end{align*}$$
> Rewrite the variables as $x_1=y_1y_2$ and $x_2=y_1(1-y_2)$. The Jacobian follows as,
$$
\begin{align*}
    J&=\begin{bmatrix}
    \frac{\partial y_1}{\partial x_1}&&\frac{\partial y_2}{\partial x_1}\\
    \frac{\partial y_1}{\partial x_1}&&\frac{\partial y_2}{\partial x_2}
    \end{bmatrix} \\
    &=
    \begin{bmatrix}
    y_2&&y_1\\
    1-y_2&&-y_1
    \end{bmatrix}
\end{align*}
$$
> The absolute determinant of this Jacobian follows as $|-y_2y_1-y_1(1-y_2)|=y_1$.
> Once again invoking the transformation of variable theorem {{< cite "hoggIntroductionMathematicalStatistics2019" >}}, the joint PDF of $y_1$ and $y_2$ follows as,
$$
\begin{align*}
    f(y_1,y_2)&=\frac{1}{\Gamma(\alpha)\Gamma(\beta)}(y_1y_2)^{\alpha-1}(y_1(1-y_2))^{\beta-1}e^{-y_1y_2-y_1(1-y_2)}|J|\\
    &=\frac{1}{\Gamma(\alpha)\Gamma(\beta)}y_1y_1^{\alpha-1}y_1^{\beta-1}y_2^{\alpha-1}(1-y_2)^{\beta-1}e^{-y_{1}}
\end{align*}
$$
> By integrating out the variables, the marginal PDFs can be found (it can be proven that $y_1$ and $y_2$ are independent of each other). The integration ranges follow by the same argument applied in the earlier proof, such that $y_1\in[0,\infty]$ and that $y_2\in[0,1]$.
> The PDF of $f(x_1)$ follows as,
> $$
\begin{align*}
    f(y_1)&= \frac{1}{\Gamma(\alpha)\Gamma(\beta)}y_1^{\alpha+\beta-1}e^{-y_{1}}\underbrace{\int_0^1 y_2^{\alpha-1}(1-y_2)^{\beta-1}dy_2}_{=1} \\
\end{align*}
$$
> Which is just a Gamma distribution: $\operatorname{Gamma}(\alpha+\beta,1)$.
>
> The PDF of $f(x_2)$ follows as,
> $$
\begin{align*}
    f(y_2)&=\frac{1}{\Gamma(\alpha)\Gamma(\beta)}y_2^{\alpha-1}(1-y_2)^{\beta-1}\underbrace{\int_0^\infty y_1^{\alpha+\beta-1}e^{-y_{1}}dy_1}_{=1} \\
    &=\frac{1}{\Gamma(\alpha+\beta)}y_2^{\alpha-1}(1-y_2)^{\beta-1}
\end{align*}
$$
> This is the function referred to by the Beta distribution: specifically, $\operatorname{Beta}(x;\alpha,\beta)$.
>
> Note that, very much like the earlier presented proof, one can relate the product of two Gamma distributions to the product of a Beta and a Gamma function,
> $$
\begin{gather*}
    \operatorname{Gamma}(\alpha)\operatorname{Gamma}(\beta)=\operatorname{Beta}(\alpha,\beta)\operatorname{Gamma}(\alpha+\beta) \\
    \implies \operatorname{Beta}(\alpha,\beta)=\frac{\operatorname{Gamma}(\alpha)\operatorname{Gamma}(\beta)}{\operatorname{Gamma}(\alpha+\beta)}
\end{gather*}
$$

Without Gamma distributions, the PDF for the Beta distribution is:
$$
\begin{align*}
    \operatorname{Beta}(x;\alpha,\beta)&=\frac{\Gamma(\alpha+\beta)}{\Gamma(\alpha)\Gamma(\beta)}x^{\alpha-1}(1-x)^{\beta-1} \\
    &=\frac{1}{\operatorname{B}(\alpha,\beta)}x^{\alpha-1}(1-x)^{\beta-1}
\end{align*}
$$

Integrating from 0 to some $x_i<1$ gives the CDF of the Beta distribution as,
$$
\begin{align}
    \operatorname{BETA}(x;\alpha,\beta)&=\frac{1}{\text{B}(\alpha,\beta)}\int^x_0x^{\alpha-1}(1-x)^{\beta-1}\\
    &=\frac{\text{B}(x;\alpha,\beta)}{\text{B}(\alpha,\beta)}
\end{align}
$$

Which is also known as the [(regularized) incomplete beta function](https://en.wikipedia.org/wiki/Beta_function#Incomplete_beta_function)[^incomplete_beta].

[^incomplete_beta]: the integrand is from $0$ to $x$, rather than $1$. Since $x<1$, it is integration over the incomplete domain of the Beta function.

{{< figure-dynamic
    dark-src="./figures/beta_distribution_shapes_dark.svg"
    light-src="./figures/beta_distribution_shapes_light.svg"
    alt="Beta distributions with different parameters."
>}}

The above figure provides various instances of the Beta PDF function. For $\alpha=\beta\geq1$ the distribution is unimodal and symmetric about $x = 0.5$. For $\alpha=\beta\leq1$ the distribution inverts, becoming U-shaped, but remains symmetric about $x = 0.5$. Whenever $\alpha>\beta>1$ or $1<α<β$ the distribution is skewed positively or negatively, respectively.

Generally, as the values for $\alpha$ and $\beta$ increase, so does it precision about its mode. A special case is $\alpha=\beta=1$: this is the uniform distribution for $0\leq x\leq1$. As a rule of thumb, for equal values for $\alpha$ and $\beta$ the cumulative probabilities can be estimated by the normal distribution {{< cite "krishnamoorthyHandbookStatisticalDistributions2016" >}}. Another special case is for $\alpha=\beta=0.5$: this is the arcsine distribution, and happens to be the [Jeffreys prior for a Bernoulli trial](https://en.wikipedia.org/wiki/Jeffreys_prior#Bernoulli_trial).

## Statistical Properties

### Normalisation

From the definition of the cumulative Beta density function, $\operatorname{BETA}$, the proof of normalisation becomes trivial.
$$
\begin{align}
    &\int_{0}^{1}\operatorname{BETA}(x;\alpha,\beta)dx\\
    &=\frac{1}{\operatorname{B}(\alpha,\beta)} \int^1_0 \operatorname{B}(x;\alpha,\beta)\\
    &=\frac{1}{\operatorname{B}(\alpha,\beta)}\cdot \operatorname{B}(\alpha,\beta)\\
    &=1
\end{align}
$$
In other words, computing the probability of some $x\sim \text{Beta}(\alpha,\beta)$ is simply computing the Beta function for the integration range of 0 to some $x<1$ divided by the Beta function computed over the whole possible domain.

### Mode

The mode of a distribution is the most probable value, or the value that maximizes the density. We can find this value by taking the first derivative of the Beta density function and setting it to $0$. This involves using the derivatives product rule:

$$
    \frac{d}{dx} f(x)g(x) = (\frac{d}{dx} f(x))g(x) - f(x)(\frac{d}{dx}g(x))
$$

After some rewriting, we get:

$$
\begin{align*}
    \frac{d}{d x}\operatorname{Beta}(x;\alpha,\beta)&=\frac{d}{d x}\frac{1}{\operatorname{B}(\alpha,\beta)}x^{\alpha-1}(1-x)^{\beta-1} \\
    &=\frac{1}{\operatorname{B(\alpha,\beta)}}\left[(\alpha-1)x^{\alpha-2}(1-x)^{\beta-1}-x^{\alpha-1}(\beta-1)(1-x)^{\beta-2}\right] \\
    &=\underbrace{\frac{1}{\operatorname{B(\alpha,\beta)}}x^{\alpha-2}(1-x)^{\beta-2}}_{(1)}\underbrace{\left[(\alpha-1)(1-x)-(\beta-1)x\right]}_{(2)}
\end{align*}
$$

We can set this expression to $0$. For this function to be $0$, one of terms must be $0$. Excluding the trivial solutions for $x=0$ or $x=1$, term $(1)$ cannot evaluate to $0$. Note that the Beta function is positive for $\alpha>0$ and $\beta>0$. Thus, we only care about when term $(2)$ goes to 0.

$$
\begin{align*}
    \frac{d}{d x}\operatorname{Beta}(x;\alpha,\beta)&=0 \\
    &\implies (\alpha-1)(1-x)-(\beta-1)x = 0 \\
    &\implies (\alpha-1)(1-x)=(\beta-1)x \\
    &\implies (\alpha-1)-(\alpha-1)x=(\beta-1)x \\
    &\implies (\alpha-1)=(\alpha+\beta-2)x \\
    &\implies x=\frac{\alpha-1}{\alpha+\beta-2}
\end{align*}
$$

The mode of the Beta distribution, is thus at $\frac{\alpha-1}{\alpha+\beta-2}$. Note that it is only defined for $\alpha>1$ and $\beta>1$ (see above figure). Otherwise the mode lies on one or both of the distribution limits. If $\alpha<1$ and $\beta<1$, this expression yields the anti-mode: the lowest density region in the PDF.

### Mean

Let the mean of x, $\mu_x$ be defined by the first expectation value. Taking X to follow a Beta distribution, the mean of X can be calculated as follows.
$$
\begin{align*}
    \mathbb{E}[x]&=\int_X x\cdot p(x) dx \\
    &= \int^1_0 x\cdot \frac{1}{B(\alpha,\beta)}x^{\alpha-1}(1-x)^{\beta-1}dx\\
    &=\frac{1}{B(\alpha,\beta)} \int^1_0 x^{\alpha}(1-x)^{\beta-1}dx
\end{align*}
$$

Note that the above integral closely approximates the definition of the Beta function with $\alpha+1$ as its first parameter. This allows for rewriting the mean as
$$
\mathbb{E}[x]=\frac{1}{B(\alpha,\beta)} B(\alpha+1,\beta)
$$
Recall the relation of the Gamma function to the Beta function. Rewriting the mean in terms of the Gamma function rather than the Beta function provides a result that is more easily simplified.
$$
\begin{align*}
    \mathbb{E}[x]&=\frac{B(\alpha+1,\beta)}{B(\alpha,\beta)}\\
    &=\frac{\Gamma(\alpha+\beta+1)/(\Gamma(\alpha+1)\Gamma(\beta))}{\Gamma(\alpha+\beta)/(\Gamma(\alpha)\Gamma(\beta))}\\
    &=\frac{\Gamma(\alpha+1)\Gamma(\beta)\Gamma(\alpha+\beta)}{\Gamma(\alpha)\Gamma(\beta)\Gamma(\alpha+\beta+1)}
\end{align*}
$$
Recall that the Gamma function can be defined using a recurrence relationship:
$$
    \Gamma(x+1)=x\Gamma(x)
$$
The mean of $X$ can then easily be found by rewriting in terms of $\Gamma(\alpha)$ and $\Gamma(\beta)$ and simplifying.
$$
\begin{align*}
    \mathbb{E}[x]&=\frac{{\color{red} \Gamma(\alpha+1)}\Gamma(\beta)\Gamma(\alpha+\beta)}{\Gamma(\alpha)\Gamma(\beta){\color{green}\Gamma(\alpha+\beta+1)}}\\
    &=\frac{{\color{red}\alpha\Gamma(\alpha)}\Gamma(\beta)\Gamma(\alpha+\beta)}{\Gamma(\alpha)\Gamma(\beta){\color{green}(\alpha+\beta)\Gamma(\alpha+\beta)}}\\
    &=\frac{\alpha}{\alpha+\beta}
\end{align*}
$$
Given that both $\alpha$ and $\beta$ are strictly positive reals, it follows that for all values of these parameters the mean of x will fall within the $[0,1]$ domain.

### Variance

Let the variance of x, $\operatorname{var}[x]$, be the difference of the expected value of $X^2$ and the squared mean: $\mathbb{E}[x^2]-\mathbb{E}[x]^2$. Due to the repetitive nature of this derivation, some steps explained in the derivation of $\mathbb{E}[x]$ will be applied but not shown or highlighted.
$$
\begin{align*}
    \mathbb{E}[x^2]&=\frac{1}{B(\alpha,\beta)} \int^1_0 x^2x^{\alpha-1}(1-x)^{\beta-1}dx \\
    &=\frac{1}{B(\alpha,\beta)} \int^1_0 x^{\alpha+1}(1-x)^{\beta-1}dx\\
    &=\frac{B(\alpha+2,\beta)}{B(\alpha,\beta)} \\
    &=\frac{(\alpha+1)\Gamma(\alpha+1)\Gamma(\beta)\Gamma(\alpha+\beta)}{\Gamma(\alpha)\Gamma(\beta)(\alpha+\beta+1)\Gamma(\alpha+\beta+1)}\\
    &=\frac{\alpha(\alpha+1)}{(\alpha+\beta)(\alpha+\beta+1)}
\end{align*}
$$
The variance then follows as:
$$
\begin{align*}
    \operatorname{var}[x]&=\mathbb{E}[x^2]-\mathbb{E}[x]^2\\
    &=\frac{\alpha(\alpha+1)}{(\alpha+\beta)(\alpha+\beta+1)}-\frac{\alpha^2}{(\alpha+\beta)^2}\\
    &=\frac{\alpha(\alpha+1)(\alpha+\beta)}{(\alpha+\beta)^2(\alpha+\beta+1)}-\frac{\alpha^2(\alpha+\beta+1)}{(\alpha+\beta)^2(\alpha+\beta+1)}\\
    &=\frac{\alpha^3+\alpha^2\beta+\alpha^2+\alpha\beta-\alpha^3-\alpha^2\beta-\alpha^2}{(\alpha+\beta)^2(\alpha+\beta+1)}
\end{align*}
$$
After simplifying we end up with:
$$
\operatorname{var}[x]=\frac{\alpha\beta}{(\alpha+\beta)^2(\alpha+\beta+1)}
$$

### Moment Generating Function

The moment generating function (MGF) is defined as the expectation value of the Laplace transform of the probability density function. One important use is the calculation of the PDF's moments, through computing the n-th derivative at $t=0$ for the n-th moment.
$$
\begin{align*}
    m_X(t)&=\mathbb{E}[e^{tx}]\\
    &=\int^1_0e^{tx}\cdot f(x;\alpha,\beta)\\
    &=\frac{1}{\text{B}(\alpha,\beta)}\int^1_0e^{tx}\cdot x^{\alpha-1}(1-x)^{\beta-1}
\end{align*}
$$

Next, we apply the Taylor series approximation of $e^{tx}$:
$$
\begin{align*}
    m_X(t)&=\frac{1}{\text{B}(\alpha,\beta)}\int^1_0(\sum^\infty_{k=0}\frac{t^k x^k}{k!})\cdot x^{\alpha-1}(1-x)^{\beta-1}\\
    &=\frac{1}{\text{B}(\alpha,\beta)}\sum^\infty_{k=0}\frac{t^k}{k!}\int^1_0x^{\alpha+k-1}(1-x)^{\beta-1}
\end{align*}
$$

Now apply the definition of the Beta function:
$$
\begin{align*}
    m_X(t)&=\sum^\infty_{k=0}\frac{t^k}{k!}\frac{\text{B}(\alpha+k,\beta)}{\text{B}(\alpha,\beta)}\\
    &=1+\sum^\infty_{k=1}\frac{t^k}{k!}\frac{\text{B}(\alpha+k,\beta)}{\text{B}(\alpha,\beta)}
\end{align*}
$$

The quotient of these Beta functions can be solved when inserting the Gamma functions, using the Beta-Gamma function relation. The final result is the [confluent hypergeometric function of the first kind](https://en.wikipedia.org/wiki/Confluent_hypergeometric_function).
$$
    m_X(t)=1+\sum^\infty_{k=1}\frac{t^k}{k!}\prod^{k-1}_{n=0}\frac{\alpha+n}{\alpha+\beta+n}
$$

## Inference

### Method of Moments Estimation

The method of moments estimators (MoME) derivation requires setting the moments of the PDF equal to the moments derived from a data set of length $N$, $x_1,x_2,\dots,x_N$. We set the first two moments, the mean and variance as derived above, to the sample mean and variance,
$$
\begin{gather*}
    \mathbb{E}[x]=\overline{x},~~\operatorname{var}[x]=s^2_x\implies \\
    \overline{x}=\frac{\alpha}{\alpha+\beta},\\
    s^2_x=\frac{\alpha\beta}{(\alpha+\beta)^2(\alpha+\beta+1)}
\end{gather*}
$$

These relations can then be reformulated to describe the parameters $\alpha$ and $\beta$ solely in terms of the sample mean and variance. For the sample mean, this follows as,
$$
\begin{align*}
    \overline{x}&=\frac{\alpha}{\alpha+\beta}\\
    (\alpha+\beta)\overline{x}&=\alpha\\
    \alpha-\alpha\overline{x}&=\beta\overline{x}\\
    \beta&=\frac{\alpha}{\overline{x}}-\alpha
\end{align*}
$$

Attempting the same for the sample variance will yield an expression independent of $\beta$ for $\alpha$.
$$
\begin{align*}
    s^2&=\frac{\alpha\beta}{(\alpha+\beta)^2(\alpha+\beta+1)}\\
    \alpha\beta&=(\alpha+\beta)^2(\alpha+\beta+1)s^2\\
    \alpha(\frac{\alpha}{\overline{x}}-\alpha)&=(\alpha+\frac{\alpha}{\overline{x}}-\alpha)^2(\alpha+\frac{\alpha}{\overline{x}}-\alpha+1)s^2\\
    \alpha^2\frac{1}{\overline{x}}-\alpha^2&=\frac{\alpha^2}{\overline{x}^2}(\frac{\alpha}{\overline{x}}+1)s^2\\
    \frac{1}{\overline{x}}-1&=\frac{1}{\overline{x}^2}(\frac{\alpha}{\overline{x}}+1)s^2\\
    (\frac{1-\overline{x}}{\overline{x}})\frac{\overline{x}}{s^2}&=\frac{\alpha}{\overline{x}}+1\\
    \hat{\alpha}&=\overline{x}(\frac{\overline{x}(1-\overline{x})}{s^2}-1)
\end{align*}
$$

All that remains is expressing $\beta$ using the result for $\alpha$,
$$
\begin{align*}
    \beta&=\frac{\alpha}{\overline{x}}-\alpha\\
    &=\frac{\alpha(1-\overline{x})}{\overline{x}}\\
    &=\overline{x}(\frac{\overline{x}(1-\overline{x})}{s^2}-1)\frac{(1-\overline{x})}{\overline{x}}\\
    \hat{\beta}&=(1-\overline{x})(\frac{\overline{x}(1-\overline{x})}{s^2}-1)
\end{align*}
$$

This estimator is unbiased. Assuming the sample mean and variance to be unbiased—$\mathbb{E}[\overline{x}]=\mu_x$ and $\mathbb{E}\{\overline{s^2}\}=\operatorname{var}[x]$—unbiaseness can be proven by inputting the derived Beta distribution mean and variance into the MoME estimators for $\alpha$ and $\beta$.

### Maximum Likelihood Estimation

The maximum likelihood estimators (MLE) requires the calculation of the likelihood of the parameters based on an i.i.d. data set from sampled the target distribution. The likelihood function is defined as the product of the probabilities of each value within the data set given the distribution in terms of its unknown parameters. For the Beta distribution the likelihood function follows as,
$$
\begin{align*}
   \mathcal{L}(\alpha,\beta)&=\prod^N_{i=1}\frac{\Gamma(\alpha+\beta)}{\Gamma(\alpha)\Gamma(\beta)}x_i^{\alpha-1}(1-x_i)^{\beta-1}\\
    &=(\frac{\Gamma(\alpha+\beta)}{\Gamma(\alpha)\Gamma(\beta)})^N\prod^N_{i=1}x_i^{\alpha-1}(1-x_i)^{\beta-1}
\end{align*}
$$

Typically, rather than maximising the likelihood function, we use the log-likelihood. This often making subsequent calculations much simpler, and does not affect the maximizers. For the Beta distribution, this is,
$$
\begin{align*}
   \mathcal{l}(\alpha,\beta)&=\log(\frac{\Gamma(\alpha+\beta)}{\Gamma(\alpha)\Gamma(\beta)})^N\prod^N_{i=1}x_i^{\alpha-1}(1-x_i)^{\beta-1} \\
   &=N\log(\Gamma(\alpha+\beta))-N\log(\Gamma(\alpha))-N\log(\Gamma(\beta)) \\
   &~~~+(\alpha-1)\sum^N_{i=1}\log(x_i)+(\beta-1)\sum^N_{i=1}\log(1-x_i)
\end{align*}
$$

To maximise the likelihood, the partial derivatives in terms of $\alpha$ and $\beta$ are set to 0 and solved for $\hat{\alpha}$ and $\hat{\beta}$. The functions that need to be solved are displayed below.
$$
\begin{align*}
    \frac{\partial}{\partial\alpha}l(\alpha,\beta)&=N(\frac{\Gamma^{(1)}(\alpha+\beta)}{\Gamma(\alpha+\beta)})-N(\frac{\Gamma^{(1)}(\alpha)}{\Gamma(\alpha)})+\sum^N_{i=1}\log(x_i) \\
    \frac{\partial}{\partial\beta}l(\alpha,\beta)&=N(\frac{\Gamma^{(1)}(\alpha+\beta)}{\Gamma(\alpha+\beta)})-N(\frac{\Gamma^{(1)}(\beta)}{\Gamma(\beta)})+\sum^N_{i=1}\log(1-x_i)
\end{align*}
$$

Immediately the interdependence of this system become obvious: one can't solve for $\alpha$ without $\beta$, and vice versa. A closed form solution does not exist, however an iterative method can be applied to approximate the values for $\alpha$ and $\beta${{< cite "owenParameterEstimationBeta2008" >}}.

The [digamma function](https://en.wikipedia.org/wiki/Digamma_function) is defined as the derivative of the natural logarithm of the Gamma function. This makes notation much easier.
$$
    \psi(x)=\frac{d}{dx}\log(\Gamma(x))=\frac{\Gamma^{(1)}(x)}{\Gamma(x)}
$$

Its derivatives, $\psi^{(i)}(x)$ are known as the [polygamma functions](https://en.wikipedia.org/wiki/Polygamma_function).

Owen{{< cite "owenParameterEstimationBeta2008" >}} recommends applying the two-dimensional Newton-Raphson method to root finding of the system defined above. Let $\vec{g}$ be the vector with these equations, rewritten in terms of the digamma function, and $\vec{G}$ be the Hessian matrix (the matrix containing all second-order derivatives)[^mean_instead_of_sum].
$$
\begin{align*}
    \vec{g}&=
    \begin{bmatrix}
    \psi(\alpha)-\psi(\alpha+\beta)-\frac{1}{N}\sum^n_{i=1}\log(x_i)\\
    \psi(\beta)-\psi(\alpha+\beta)-\frac{1}{N}\sum^n_{i=1}\log(1-x_i)
    \end{bmatrix} \\
    \vec{G}&=\begin{bmatrix}
    \frac{\partial}{\partial \alpha}\frac{\partial}{\partial\alpha}l(\alpha,\beta)&&\frac{\partial}{\partial \alpha}\frac{\partial}{\partial\beta}l(\alpha,\beta)\\
    \frac{\partial}{\partial \beta}\frac{\partial}{\partial\alpha}l(\alpha,\beta)&&\frac{\partial}{\partial \beta}\frac{\partial}{\partial\beta}l(\alpha,\beta)
    \end{bmatrix}\\
    &=\begin{bmatrix}
    \psi^{(1)}(\alpha)-\psi^{(1)}(\alpha+\beta)&&-\psi^{(1)}(\alpha+\beta)\\
    -\psi^{(1)}(\alpha+\beta)&&\psi^{(1)}(\beta)-\psi^{(1)}(\alpha+\beta)
    \end{bmatrix}
\end{align*}
$$

[^mean_instead_of_sum]: Note that this implicitly divides the log-likelihood derivates by $N$. This does not affect the outcome, since this would cancel out anyways, but dividing beforehand does make the algorithm more numerically stable.

The parameters can then be approximated by iteratively subtracting the product of the inverse Hessian matrix ($\vec{G}^{-1}$) and the system of differential equations ($\vec{g}$) from the current estimate:
$$
    \{\hat{\alpha},\hat{\beta}\}_{i+1}=\{\hat{\alpha},\hat{\beta}\}_{i}-\vec{G}^{-1}\cdot\vec{g}
$$
As $i\rightarrow\infty$, the estimate converges to $\hat{\alpha}_{\text{MLE}}$ and $\hat{\beta}_{\text{MLE}}$. Empirically, 10 iterations proved sufficient for most cases, especially when good starting conditions were chosen.

### Efficiency

The Cramer-Rao Lower Bound (CRLB) states that the variance of an estimator is always above or at the inverse of the Fisher information of the parameters. Under the assumption that $\theta$ is an unbiased estimator of the true parameter, it follows as
$$
    \operatorname{var}[\hat{\theta}]\geq\frac{1}{I_X(\theta)}
$$

Here $I_X(\theta)$ is the Fisher information function in terms of $\theta$, defined as,
$$
    I_X(\theta)=-\mathbb{E}[\frac{\partial^2}{\partial^2\theta}\log l(x;\theta)]
$$

The log-likelihood functions were derived in the preceding sections. Their derivatives, in terms of $\alpha$ and $\beta$, are independent in terms of $x_i$. These are given by the main-diagonal elements of the above computed Hessian matrix.
$$
\begin{align}
    I(\alpha)&=N(\psi^{(1)}(\alpha)-\psi^{(1)}(\alpha+\beta)) \\
    I(\beta)&=N(\psi^{(1)}(\beta)-\psi^{(1)}(\alpha+\beta))
\end{align}
$$

## Simulation

To simulate the properties of the MoME and MLE estimators, a $\operatorname{Beta}(\alpha=2,\beta=6)$ distribution was sampled with various $N$, repeated $100$ times for each $N$. We are interested in three metrics:
1. **bias**: $\mathbb{E}[\theta-\hat{\theta}]$
2. **variance**: $\operatorname{var}[\hat{\theta}]$
3. **efficiency**: $\operatorname{var}[\hat{\theta}]/\operatorname{CRLB}$

{{< figure-dynamic
    dark-src="./figures/simulation_results_dark.svg"
    light-src="./figures/simulation_results_light.svg"
    alt="Simulation results comparing the MoME and MLE estimators."
>}}

The results are depicted in the graph above. Both the MoME and MLE estimators are consistent, as their bias tends to $0$ as $N\rightarrow \infty$. Furthermore, estimator variance clearly decreases with sample size. While neither estimator is efficient (their variance for a given $N$ never approaches the CRLB), the MLE estimator does tend to be a little more efficient. That said, the quality of the MLE solution is dependent on starting conditions. As such, a good strategy combining both is to use the MoME estimate as the initial estimate which refined by several MLE iterations.

## References

{{< references >}}

## Changelog

```
[2025-01-09] Fixed a lot of errors and typos
```
