---
title: WIP Comparing Models using Ratios
date: 2024-10-01
tags:
  - statistics
  - machine-learning
math: true
draft: false
---

{{< toc >}}

Many machine learning and NLP experiments require comparing various ratios which measure...

## Frequentist Treatment

Let's assume we have two models whose performance we are going to measure through some ratio (e.g., the accuracy score). A ratio is defined as:
$$r=\dfrac{\#\text{hits}}{N}$$
where $N$ is the size of the sample, and $\#\text{hits}$ is the number of times we note a hit or success. The ratio value will be bounded in the interval $[0,1]$, but the possible values it can take will be determined by the value of $N$.

When *comparing* ratios, we care less about the values of the individual ratios, and more about some *difference* in ratios. If system 2 has an accuracy of $0.8$, while the baseline system 1 has a score of $0.7$, we know a lot more about the system that if we had just been looking at each value in isolation. In short, we want to define some function that quantifies how large or small the difference between the two ratios are:
$$\Delta(r_1, r_2)=???$$

Many such ratio difference functions exist, and I list some common ones below. Most of these come from the biomedical field, hence the somewhat morbid naming conventions.

| Metric                                                                           | Formula                                                           |
| -------------------------------------------------------------------------------- | ----------------------------------------------------------------- |
| [Risk Difference](https://en.wikipedia.org/wiki/Risk_difference)                 | $\text{RD}(r_1,r_2)=r_2-r_1$                                      |
| [Number Needed to Treat](https://en.wikipedia.org/wiki/Number_needed_to_harm)    | $\text{NNH}(r_1, r_2)=\dfrac{1}{r_2-r_1}$                         |
| [Relative Risk](https://en.wikipedia.org/wiki/Relative_risk)                     | $\text{RR}(r_{1}, r_{2})=\dfrac{r_2}{r_1}$                        |
| [Relative Risk Reduction](https://en.wikipedia.org/wiki/Relative_risk_reduction) | $\text{RRD}(r_1, r_2)=\dfrac{r_1-r_2}{r_1}=1-\text{RR}(r_1, r_2)$ |
| [Relative Risk Increase](https://en.wikipedia.org/wiki/Relative_risk_reduction)  | $\text{RRI}(r_1, r_2)=\dfrac{r_2-r_1}{r_1}=\text{RR}(r_1, r_2)-1$ |

While these are easy to understand, and frequently get 'reinvented' by machine learning papers, it turns out that such metrics can lead to mistakes or paradoxes.

### The Problem with Ratio Differences

A first, rather obvious weakness of these approaches is that these tend to explode near critical values. While most of the range is sensibly defined, a few extreme outliers can quickly ruin summary statistics.

{{< figure-dynamic
    dark-src="./figures/naive_risk_differences_dark.webp"
    light-src="./figures/naive_risk_differences_light.webp"
    alt="3D plots of the range and domains of the above presented ratio differencing methods."
    caption="3D plots of the range and domains of the above presented ratio differencing methods."
>}}

More importantly, however, the interpretation of a (relative) change in ratios depends on the *direction* of the ratios. Whether I use the ratio $r$ or its complement, $\bar{r}=1-r$ can have a drastic effect on the interpretation of the ratio difference. This is probably best showcased in the [potato paradox](https://en.wikipedia.org/wiki/Potato_paradox), where a simple change in perspective leads to very different results.

To ground the discussion in more familiar machine learning terms, consider a model that goes from an accuracy rate of $0.9$ to $0.99$ after some intervention. In absolute terms, this is a small increase of only $0.09$, or
$$\frac{0.99-0.9}{0.9}=+10\%$$
My gut instinct, however, is that this represents a massive improvement. In fact, the amount of errors (the complement of the accuracy rate) has been reduced by $90\%$!
$$\frac{(1-0.99)-(1-0.9)}{(1-0.9)}=\frac{0.01-0.1}{0.1}=-90\%$$
There seems to be a disconnect between these two perspectives, despite discussing the exact same data.

Ideally, our differencing function $\Delta(r_1, r_2)$ is symmetric w.r.t. our interpretation of our ratios. [Stating this mathematically, what we want is something like](https://math.stackexchange.com/a/231731),
$$\Delta(p_1, p_2)=-\Delta(1-p_1,1-p_2)$$
Achieving this with raw ratios is likely not possible, but there is a closely related quantity that does have this property (and many more): the log odds ratio.

### Odds & Odds Ratios

[Odds](https://en.wikipedia.org/wiki/Odds) are defined as the ratio of a probability to its complement,
$$\text{Odds}(x)=\frac{p}{1-p}$$

While this might seem cumbersome initially, their interpretation is actually quite intuitive. One area where odds are used frequently is in gambling, where the payout of a bet is proportional to the odds of that bet succeeding. For example, if a horse has a 40% chance of winning a race, the odds are,
$$p=\frac{2}{5}\mapsto \text{Odds}(p)=\frac{2}{3}$$
If you win, you would make €300 on a €200 stake.

When comparing ratios, however, the odds and odds ratio has another appealing property. Namely, probability complements give odds reciprocals.

Consider the same problem as before, where a model increases its accuracy score from $0.9$ to $0.99$. Expressed as odds, an accuracy of $0.9$ and its complement are,
$$
\begin{align*}
	\text{Odds}(0.9)&=\frac{0.9}{0.1}=9 \\
    \text{Odds}(1-0.9)&=\frac{0.1}{0.9}=\frac{1}{9}
\end{align*}
$$
Shifting our perspective now gives a clear relationship between the two situations.

The canonical method for comparing two odds is the odds ratio. It is defined as,
$$\text{OR}(r_1, r_2)=\frac{\text{Odds}(r_2)}{\text{Odds}(r_1)}=\frac{p_1(1-p_2)}{p_2(1-p_1)}$$

Notice how the same property holds. If we let $r_2=0.99$ and $r_1=0.90$ be the system ratios, the odds ratios we get are:
$$\text{OR}(r_1, r_2)=\frac{99}{9}=11,\quad \text{OR}(\bar{r_1}, \bar{r_2})=\frac{1/99}{1/9}=\frac{9}{99}=\frac{1}{11}$$

Finally, to retrieve the desired relationship of $\Delta(p_1, p_2)=-\Delta(1-p_1,1-p_2)$, we just throw some logarithms at it,
$$\log \text{OR}(0.9, 0.99)=\log11, \quad\log\text{OR}(0.1, 0.01)=\log\frac{1}{11}=-\log11$$
*Et voilà*, the only thing hat changes when we shift perspective from a ratio to its complement, is that we add a minus sign in front. Purely based on this property, the **log odds ratio** is the ratio differencing function we want.

### Log Odds Ratios

This comes at a price though. Any hope for a nicely intuitive metric went out the window as soon as we went to an odds ratio, and adding a logarithm only makes it worse. Odds are already a ratio of ratios, so the log odds ratio is a logarithm of a ratio of ratios of ratios???

The log odds ratio is a good choice if you care about correctness, but indeed less so if you care about interpretability. Newspaper headlines sometimes state things like “dog owners are twice as likely to get into car accidents”. These are usually statements about odds ratios, and can be used to 'lie' with otherwise sound statistics. For example, for a probability of 1 in a million (1e-6) to 2 in a million (2e-6), the odds ratio is $2$ (pretty large). While this is factually correct, it is also practically meaningless.

So why use the (log) odds ratio? Simply because it has [plenty of other properties that make statistical inference very easy](https://stats.stackexchange.com/a/452844).

1. The log odds ratio has access to the entire real line, and can take any value in $[-\infty, \infty]$
2. The variance of the log odds distribution does **not** depend on the mean[^1]
3. The log odds ratio scale is symmetric around 0[^2]
4. The log odds ratio is **asymptotically** normally distributed

[^1]: This is not the case for most of the other discussed ratio differencing functions. For why this is important, check out the page on [Binomial proportion confidence intervals](https://en.wikipedia.org/wiki/Binomial_proportion_confidence_interval). In short, not having this property makes using distributions of ratio very difficult.

[^2]: This makes computing means or other metrics of central tendency sensible. For example, the mean of the odds ratios $\frac{1}{2}$ and $\frac{2}{1}$ is $\frac{5}{4}$, but the mean of their log-odds ratios is $0$, which is more correct.

While each of these properties is useful, the last supersedes them all. The distribution of all the other ratio difference functions are unknown but probably intractable. In comparison, the log odds ratio is nicely defined with an abundance of summary statistics available.

The word “asymptotic” is doing a lot of lifting in that sentence. For ratios estimated using few samples (the $N$ in our first formula) the approximation falls apart. The following figure shows a histogram of 10,000 samples of draws from a binomial distribution with ratios $r_1=0.7$ and $r_2=0.8$ of various sizes $N$.

{{< figure-dynamic
    dark-src="./figures/odds_and_log_odds_ratio_distributions_dark.webp"
    light-src="./figures/odds_and_log_odds_ratio_distributions_light.webp"
    alt="Histograms drawn from samples of the odds (top) and log odds ratio (bottom) at various $n$. For the log odds ratio, the asymptotic normal distribution approximation is also shown using an outline."
    caption="Histograms drawn from samples of the odds (top) and log odds ratio (bottom) at various $n$. For the log odds ratio, the asymptotic normal distribution approximation is also shown using an outline."
>}}

As the size of the samples increases, the approximation becomes better and better, but for smaller $N$, the histograms are clearly discrete. What size of $N$ is enough for the approximation to hold up? From stats 101, the number 30 is a good rule of thumb.

Regardless, inference with the log odds ratio is easy. All of a sudden, we take a family of difficult quantities, and transport it to the comfortable realm of normally distributed statistics. All of a sudden, means and medians become sensible measures of central tendency; we can compute variances and derive confidence intervals; we can add or subtract under closure; etc. These properties allow us to reason about log odds ratios and ask natural question about their quantity. One important one might whether the value is statistically significant or not. We can achieve this through the $z$-test.

The *asymptotic* standard error for the log odds ratio is defined as,
$$\text{SE}(\log \text{OR}(r_1, r_2))=\sqrt{\frac{1}{r_1N_1}+\frac{1}{N_1-r_1N_1}+\frac{1}{N_2-r_2N_2}+\frac{1}{N_2-r_2N_2}}$$
Here $rN$ is a proxy for the number of wins, $\#\text{hits}$, and $N-rN$ for the number of 'losses'.

From this, we can compute a confidence interval as
$$\log \text{OR}(p_1, p_2)\pm z_{1-\alpha/2}\text{SE}(\log \text{OR}(p_1, p_2))$$
where $z_{1-\alpha/2}$ is the critical value for some confidence vvalue $\alpha$ (e.g., $1.96$ for the $95$% confidence interval). We can then exponentiate these values to retrieve a confidence interval for our odds ratios[^3].

[^3]: Though this has some caveats. It's better to stay in log odds ratio space

In turn, this enables statistical hypothesis testing. To compute a $p$-value, w.r.t. some hypothesized value $H$, we can use the [$z$-test](https://en.wikipedia.org/wiki/Z-test). We first compute the following effect size,
$$z=\frac{\log \text{OR}(p_1, p_2)-H}{\text{SE}(\log \text{OR}(p_1, p_2))}$$
We know that this quantity is [asymptotically standard normally distributed](https://stats.stackexchange.com/a/467631), $z\sim\mathcal{N}(0,1)$. To compute a $p$-value, we plug it into the normal CDF. Effectively, this answers how likely it is to see this effect size under the null hypothesis.

It is easy enough to do manually, but it is also implemented in the [`statsmodels` Python library](https://www.statsmodels.org/dev/generated/statsmodels.stats.weightstats.ztest.html):

```python
>>> statsmodels.stats.weightstats.ztest(
    x1=[0, 1, 1, 1, 1, 0, 1, 1, 1, 0], # samples_1 for r_1=0.7
    x2=[1, 1, 0, 1, 1, 1, 1, 1, 0, 1], # samples_1 for r_1=0.8
    alternative="two-sided",
)

np.float64(-0.49319696191607226), np.float64(0.6218734243307407)
```

The first number is the test statistic, the second the $p$-value. At the moment, this is very large, likely due to the fact that our sample sizes $N$ are very small. To really be able to leverage frequentist approaches, you ideally have large sample sizes. The smaller the ratios are, the larger the sample sizes need to for statistical significance. The sensitivity of a statistical test to differentiate between real and fictional results is called [power](https://en.wikipedia.org/wiki/Power_(statistics)). For small sample sizes, the $z$-test is relatively underpowered, in which case alternatives like the $t$-test can be considered.

### Logistic Regression

A more comprehensive, and potentially easier, method for achieving this result, is by running a very simple logistic regression.  We can estimate:
$$\log(\frac{p}{1-p})\sim \beta_0+\beta_1 \text{wasIntervened(x)}$$
where $x$ are our rates, and $\text{wasIntervened(x)}$ is dummy variable (one-hot encoded) determining whether the rate comes from our base or intervened model.

Parameter $\beta_1$ measures the impact that the intervention has had on the dependent variable. If using standard statistical software, the $z$-test is usually performed automatically.

[This excellent blog post](https://www.countbayesie.com/blog/2021/9/30/the-logit-normal-a-ubitiqutious-but-strange-distribution)[^4] discussed implementations, comparisons to the Bayesian Beta-Binomial model and computing magnitude differences using the spooky [Logit-Normal distribution](https://en.wikipedia.org/wiki/Logit-normal_distribution).

[^4]: Another reason to read this blog post is the philosophical musing at the end: "... many statisticians ... turn to statistics as a tool to hide from the realities of a rapidly changing world, clinging to thin strands of imagined certainty, and hiding doubt in complexity." *chef's kiss*

Using `statsmodels`, we can estimate our logistic regression model on the same data as above, as,

```python
>>> import numpy as np
>>> import statsmodels
>>> import statsmodels.api

>>> dependent = np.concatenate([samples_1, samples_2])

>>> independent = np.stack(
        [
            # A column of 1s for our constant
            np.concatenate(
                [np.ones(shape=samples_1.shape), np.ones(shape=samples_2.shape)]
            ),
            # A column of 0 or 1 for our dummy variable
            np.concatenate(
                [np.zeros(shape=samples_1.shape), np.ones(shape=samples_2.shape)]
            ),
        ],
        axis=1,
    )

>>> log_reg = statsmodels.api.Logit(
        endog=dependent,
        exog=independent,
    ).fit()

>>> print(log_reg.summary())

                           Logit Regression Results
==============================================================================
Dep. Variable:                      y   No. Observations:                   20
Model:                          Logit   Df Residuals:                       18
Method:                           MLE   Df Model:                            1
                                     ...
==============================================================================
                 coef    std err          z      P>|z|      [0.025      0.975]
------------------------------------------------------------------------------
const          0.8473      0.690      1.228      0.220      -0.505       2.200
x1             0.5390      1.049      0.514      0.608      -1.518       2.596
==============================================================================
```

From this, we see that the second model is $\exp(0.5390)=1.74\quad[0.22,13.41]$ times more likely to make a correct prediction, but that this is not significant at the 95% confidence level ($p=0.608$).

Notice that the `const` variable corresponds to our rate of $\sigma(0.8473)=0.7$ and that the logit of the sum of the coefficients gives us the rate of the second model $\sigma(0.8473+0.5390)=0.8$. Here $\sigma$ denotes the [standard logistic function](https://en.wikipedia.org/wiki/Logistic_function), the inverse of the logistic map we used to convert rates to odds.

The added benefit is that now we can start to analyse *why* the second model does or does not perform better by controlling for different variables.

In this simple example, however, we have a **very** small sample sizes. This is likely the main reason behind non-significance. More importantly, however, is that the small samples might violate the assumptions of the $z$-test. As seen in the figure above, for $N=10$, the distributions do not particularly normally distributed. If we want to get around these limitations, we'll have to go to a Bayesian framework.

## Bayesian Treatment

Bayesian approaches are more complicated, but if done correctly, it could help in cases of small sample sizes and boost interpretability. There is also a 'coolness' factor that frequentist approaches just can't touch.

Assuming that the models we will discuss align with the ratio's actual underlying distributions, we can utilize posterior distributions that are exact models. This means concerns about 'asymptotic' behaviour and small sample sizes go right out the window. We can also estimate these models directly on the ratios, so we don't have to transform to odds or odds ratios.

### Beta-Binomial Model for Independent Samples Comparison

Recall the definition of our ratio or probability as,
$$r=\frac{\#\text{hits}}{N}$$
Essentially, we're taking repeated independent draws from a [Bernoulli distribution](https://en.wikipedia.org/wiki/Bernoulli_distribution). A Bayesian model for this scenario is the [Beta-Binomial](https://en.wikipedia.org/wiki/Beta_distribution#Bayesian_inference) distribution. Here the Beta distribution serves as the conjugate prior for the true ratio $\hat{r}$, which we estimate from the Binomial likelihood distribution.

Using some hyperparameters $\alpha, \beta$, the conjugate posterior ratio distribution is,
$$\text{Beta}(\alpha+rN, \beta+(N-rN))=\text{Beta}(\alpha+\#\text{hits}, \beta+(N-\#\text{hits}))$$
To derive a distribution for $\Delta(r_1, r_2)=r_2-r_2$, we'd need to utilize the Beta difference distribution. While this [distribution is known in closed form](https://stats.stackexchange.com/a/535748), it's probably easier to just sample from two different distributions and take their difference. The result of this process is depicted in the figure below, with two fixed rates ($0.7$ and $0.8$) but with varying sample sizes. Note how the width of the distribution shrinks with the sample size, just like the frequentist case.

{{< figure-dynamic
    dark-src="./figures/two_betas_and_diff_dist_dark.webp"
    light-src="./figures/two_betas_and_diff_dist_light.webp"
    alt="Histograms of samples from the two Beta-Binomial models (top), along with the MCMC computed difference distribution (bottom)."
    caption="Histograms of samples from the two Beta-Binomial models (top), along with the MCMC computed difference distribution (bottom)."
>}}

To estimate the probability that two probabilities are different, we can just count how often, relatively, a sample from one distribution is larger than from the other distribution:
$$\begin{align}
p(r_2>r_1)&\approx \frac{1}{K}\sum_{k=1}^{K}\mathbb{1}(\tilde{r}_{2,k}>\tilde{r}_{1,k}) \\
\tilde{r}_{1,k}&\sim\text{Beta}(\alpha+r_1N_1, \beta+(N_1-r_1N_1)), \\
\tilde{r}_{2,k}&\sim\text{Beta}(\alpha+r_2N_2, \beta+(N_2-r_2N_2)),
\end{align}$$
If you want some actual summary statistics to report, you could always take the mean, median or modes and with an HDI (the Bayesian equivalent of the confidence interval) of the separate Beta distributions. These are all available analytically. Computing the same for the difference distribution is likely only possible using samples, as discussed above.

If you want to know more about Bayesian hypothesis testing, have a look at [Kruske (2013). *Bayesian Estimation Supersedes the t Test*](https://jkkweb.sitehost.iu.edu/articles/Kruschke2013JEPG.pdf) or the tutorials in the [`bayestestr`](https://easystats.github.io/bayestestR/) package. Otherwise, Count Bayesie has [another good post about a very similar problem](https://www.countbayesie.com/blog/2015/4/25/bayesian-ab-testing).

Some good quantities to estimate include the,
1. **Probability of Direction**: the amount of mass in the difference distribution greater than 0.0 (as computed above)
2. **Median Difference**: the median of the difference distribution (here $0.0866$)
3. **HDI of Difference Distribution**: to compute a [credible interval](https://stats.stackexchange.com/questions/2272/whats-the-difference-between-a-confidence-interval-and-a-credible-interval), just take the locations of the $(1-\alpha)/2\%$ and $1-(1-\alpha)/2\%$ quantiles (here $[-0.2606, 0.4289]$ for $\alpha=0.95$)

Where the first quantity tells us something about statistical significance, the latter two tell us something about practical significance. In this case, the average difference is quite small, and likely just an artefact of the sampling. For more information, we really need to increase the sample sizes.

### Dirichlet-Multinomial Model for Paired Samples Comparison

One situation where small sample sizes might still yield high statistical significance is if we're running a paired samples experiments. In paired experiments we have two different measurements for each instance in the sample, for example, by running different models on the same data point.

[Goutte & Gaussier (2005)](https://link.springer.com/chapter/10.1007/978-3-540-31865-1_25) have an interesting approach to this problem. For two competing systems on the same dataset, all predictions can fall into 1 of 3 outcomes:
1. System 1 is correct, while system 2 is incorrect
2. System 1 is incorrect, while system 2 is correct
3. Both system 1 and system 2 agree on their decision

Before, we were dealing with a binary encoding scheme; each prediction was either correct or not. Now we're dealing with 3 different outcomes, meaning the Beta-Binomial model is no longer applicable. Luckily, it's multidimensional cousin, the [Dirchlet-Multinomial](https://en.wikipedia.org/wiki/Dirichlet-multinomial_distribution) model is perfect for this. If this distribution is new to you, I'd recommend playing around with [this visualization](https://observablehq.com/@herbps10/dirichlet-distribution).

Using a vector of prior parameters $\vec{\alpha}$, the conjugate posterior distribution is,

$$\text{Dirichlet}(\alpha_{1>2}+\#1>2, \alpha_{1<2}+\#1<2, \alpha_{1=2}+\#1=2)$$

where $1>2$ is meant to signify the instances where system 1 beat system 2, $1<2$ instances where system 2 beats system 1, and $1=2$ instances wherethe models agree.

Unlike the Beta-Binomial model, the probability distribution describing $p(r_2>r_1)$ is not analytically tractable. This doesn't matter though, we can just sample from the fitted posterior distribution, and estimate it as,
$$\begin{align}
p(r_2>r_1)&\approx \frac{1}{K}\sum_{k=1}^{K}\mathbb{1}(\tilde{r}_{2,k}>\tilde{r}_{1,k}) \\
\tilde{r}_{1,k},\tilde{r}_{2,k}&\sim \text{Dirichlet}(\ldots)
\end{align}$$
where $k$ are the number of samples we draw from the posterior distribution.

Other methods for describing the difference include computing the mean, mode or the mean log odds ratio (there it is again!) of the Dirichlet distribution, all of which *are* available in closed form.

Relative to the independent Beta-Binomial model discussed above, this test should have more statistical power. This means that for paired setups, you need far fewer samples to detect the same effect.

## Comparing Many Ratios

So far we've focused on comparing $2$ ratios against each other. Individually, those ratios might comprise many observations, the number of systems producing these observations is limited. In many cases, however, we have access to many different ratios, each which communicates something distinct about our model's behaviour.

Statistically speaking, having acess to many smaller ratios can benefit the power of the analysis on the aggregated ratios.

### Frequentist Meta-Analysis

In the frequentist framework, there exists an entire (very, very useful) branch of statistics dedicated to generating such aggregated analyses. It's called 'meta-analysis', and has become the norm when working on quantitative systematic reviews[^5].

[^5]: For an example from my undergrad, [see here](/unlisted/sleep%20meta%20analysis.pdf). We combined 32 separate studies and found (to no one's surprise) that sleep correlates pretty well with academic performance

While it's too complex to delve into fully here, the simplest fixed-effect model is relatively easy to compute. Assuming the underlying metrics are all normally distributed (hint: use the log odds), we can compute the aggregated effect as the inverse variance weighted distribution.

$$\mathcal{N}(\mu^{\text{(agg.)}}, \sigma^{\text{(agg.)}})=\mathcal{N}\left(\sum_{j=1}^{J}\frac{w_{j}\mu_{j}}{\sum j^\prime w_{j^\prime}}, \frac{1}{\sum_{j=1}^{J}w_{j}}\right)$$

where the per study weight, $w_{j}$, corresponds to the study's precision: $\frac{1}{\text{SE}_{j}^2}$. In other words, each 'study' (each independent ratio), is weighted by the expected precision of that study. The aggregate distribution is then formed from weighted average of all studies. Almost always, the variance of the aggregate distribution is much smaller than that of the individual studies. For a deeper dive (with worked examples using the (log) odds ratio as study metric), I can recommend [Borenstein et al. (2021) Introduction to Meta-Analysis](https://www.wiley.com/en-us/Introduction+to+Meta-Analysis%2C+2nd+Edition-p-9781119558354), or the more easily accessible [Harrer et al. (2021). Doing Meta-Analysis with R: A Hands-On Guide.](https://bookdown.org/MathiasHarrer/Doing_Meta_Analysis_in_R/). Implementing this in Python is not too difficult, especially given that the `statsmodels` package has [functions galore for meta-analyses](https://www.statsmodels.org/stable/examples/notebooks/generated/metaanalysis1.html).

### Bayesian Meta-Analysis

In the Bayesian framework, as usual, things get a little more complicated. How this is usually done is by fitting a hierarchical Bayesian model to the log odds. In such models, it is assumed each individual study has some error, but all share some underlying parameters. The model recommended by [Harrer et al. (2021)](https://bookdown.org/MathiasHarrer/Doing_Meta_Analysis_in_R/bayesian-ma.html) is as follows,
$$\begin{align}
    \mu &\sim\mathcal{N}(0, 1) \\
    \tau &\sim\text{HalfCauchy}(0,0.5) \\
    \theta_{j}&\sim\mathcal{N}(\mu, \tau^2) \\
    \hat{\theta}_{j}&\sim\mathcal{N}(\theta, \sigma_{k}^2)
\end{align}
$$
where $\mu$ is the aggregated effect size, $\hat{\theta}_{j}$ the observed effect size, $\theta_{j}$ the unobserved effect size, $\sigma_{k}$ the standard error for each study, and $\tau$ a parameter controlling the inter-study heterogeneity. The aggregate distribution is thus parameterised by $\mu, \tau$.

This is by no means the only such model. In general, partial pooling or multi-level Bayesian models are a dime a dozen, with great tutorials in many places. THE Bayesian bible, [Gelman et al. (2014) Bayesian Data Analysis (3rd ed.)](http://www.stat.columbia.edu/~gelman/book/) discusses Bayesian meta-analysis as a special case in chapter 5.6. Fitting hierarchical models to binomial data (usual baseball hits or basketball free-throws) is also a foundational example for many [probabilisitic programming languages](https://cran.r-project.org/web/packages/rstanarm/vignettes/pooling.html).

Regardless, going the Bayesian route here is certainly doable, but requires some more effort, and probably some familiarity with probabilistic programming languages like [PyMC](https://www.pymc.io/welcome.html) or [numpyro](https://num.pyro.ai/en/latest/getting_started.html).

### The Area Under the Improvement Curve

I can't find any citations on this, but it seems like a decent idea. Typically, when comparing probabilities or ratios, we're working in a paired experiment. We have some value $p_1$ which denotes our baseline, and some value $p_2$ which represents performance under a different situation.

If you plot these in a grid, with the value of the original probabilities $p_1$ along the x-axis and the new (hopefully improved) probabilities $p_2$ along the y-axis, you should get something like the following diagram.

{{< figure-dynamic
    dark-src="./figures/area_under_improvement_curve_dark.webp"
    light-src="./figures/area_under_improvement_curve_light.webp"
    alt="The left figure plots the various ratios on the unit square. The solid line along the diagonal denotes the border between improvement and dissapointment. The dashed line provides the estimated expected improvement value for various value of ratio 1. The right figure converts the raw samples into two regions. In green, the area under the improvement expected curve, and in red the area under the dissapoint diagonal."
    caption="The left figure plots the various ratios on the unit square. The solid line along the diagonal denotes the border between improvement and dissapointment. The dashed line provides the estimated expected improvement value for various value of ratio 1. The right figure converts the raw samples into two regions. In green, the area under the improvement expected curve, and in red the area under the dissapoint diagonal."
>}}

In the right corner, under the diagonal, lie all the cases where the probabilities did **not** improve. Given that this right triangle runs from 0 to 1, the area of this 'zone of disappointment' is $0.5$. Everything above the diagonal includes the cases where the probabilities did improve. The further the point is away from the diagonal, the better the improvement.

As a summary statistic, we could figure out what the area is under function that describes the expected improvement, $\mathbb{E}[p_2|p_1]$. This function we can estimate through fitting a complex polynomial using linear regression. Using standard `scipy` functions, we can approximately integrate the regression function, 
$$\begin{aligned}
	&\text{AoI}(p_1, p_2)=\int_{p_1=0}^{p_1=1} f(p_1) d p_1 \\
	&\text{where}\quad f(p_1)=\mathbb{E}[p_2|p_1]\approx \sum_{i=0}\beta_{i}x^{i}
\end{aligned}$$
there more this area of improvement deviates from the area of the zone of disappointment (0.5), the greater the improvement. Finally, to ensure no improvement is 0, perfect improvement is 1, and perfect deterioration is -1, you could simply rescale the values as,
$$\text{ShiftedAoI}(p_1, p_2)=\frac{\text{AoI}(p_1, p_2)-0.5}{1-0.5}=2\text{AoI}(p_1, p_2)-1$$
The number is still not clearly interpretable, but at least it's paired with a nice visual explanation, and it's easy to create an understanding of system performance.
