---
title: DRAFT Sampling Maximally Diverse Subsets
date: 2021-08-26
tags:
  - machine-learning
math: true
draft: false
---

{{< toc >}}

Applied machine-learning often requires reasoning about points in high-dimensional spaces, where distances encode some form of information. A few times now I've needed to summarize such semantically rich spaces using a very small subset of samples. In such cases, ideally you want samples that come from diverse, non-overlapping regions in the space, maximizing the information content per sample.

Succinctly, this yields an optimization problem of the form,
$$
\begin{align}
    \underset{\mathcal{X}}{\operatorname{arg max}}\quad&\min\quad\{d(x, x^\prime)|x, x^{\prime}\in\mathcal{X}, x\not= x^\prime\}\\ &\text{s.t.}\quad|\mathcal{X}|=k
\end{align}
$$
or put otherwise, we want to find a subset $\mathcal{X}$ of size $k$, where the nearest neighbours are as distant as possible. Visually, this looks like:

![A visual depiction of maximally diverse subsampling of a uniformly distributed square.](figures/some_points.svg)

As it happens, such problems crop up all over the place. Unfortunately, similar problems (e.g., the [maximally diverse grouping problem](https://grafo.etsii.urjc.es/optsicom/mdgp.html), [bin packing](https://en.wikipedia.org/wiki/Bin_packing_problem)) are understood to have NP-hard exact solutions. Given that our dataset is going to comprise thousands or millions of datapoints, brute search is probably out of the question.

Then again, exact solutions are overrated. Is it possible to find a good enough approximation quickly?

## Space-Filling Designs

One area where such problems arise, is that of [experimental design](https://en.wikipedia.org/wiki/Design_of_experiments). For example, when running a hyperparameter optimisation search, you'd like to avoid running costly experiments with similar hyperparameters. Instead, initially you want to explore the space of possible values as much as possible before honing in on an optimum.

Designs that maximize the diversity in responses like this, are often called 'Space-filling Designs'. As described, in computer science experiments, you initially want to explore as broad a selection of hyperparemeters as possible. The Wootton, Sergent, Phan-Ta-Luu (WSP)[^wsp-citation] algorithm prescribes a relatively simple method for deriving a space-filling design from a candidate set of points:
```
1. Generate a set of N points
2. Compute the pairwise distance matrix between all N points
3. Choose a seed point and a minimal distance `min_dist`
4. Remove all points whose distance to the seed point is smaller than `min_dist`
5. For the next point, choose the point closests to the seed point whose distance is greater than `min_dist`
6. Iterate steps 4-5 until no more points can be chosen
```

[^wsp-citation]: the clostest citation I could find is: Santiago, Claeys-Bruno & Sergent (2012). [Construction of space-filling designs using WSP algorithm for high dimensional spaces](https://www.sciencedirect.com/science/article/pii/S0169743911001195). Chemometrics and Intelligent Laboratory Systems, 113, 26-31.

We can easily implement something like this is Python. Assuming we have access to the pairwise distance matrix, and have a decent initial guess for `min_dist`, the code should look something like this[^jaxtyping],

[^jaxtyping]: `jtyping`, the import I use to annotate arrays is the amazing [`jaxtyping`](https://docs.kidger.site/jaxtyping/). The general syntax is `dtype[type, "dimensions"]`. If you like readable numpy/Pytorch/jax, consider checking it out

```python
def wsp_space_filling_design(
    min_dist: float,
    seed_point: int,
    dist_matrix: jtyping.Float[np.ndarray, " num_points num_points"],
) -> jtyping.Int[np.ndarray, " num_chosen_points"]:
    # A point should never be able to choose itself, so set diagonals to nan
    dist_matrix_ = np.copy(dist_matrix)
    np.fill_diagonal(dist_matrix_, np.nan)

    # Add the seed point to the list of chosen points
    chosen_points = [seed_point.squeeze()]

    cur_point = np.copy(seed_point)

    # Start the iterations
    while True:
        # Find all points points within a circle of radius min_dist around the current point
        points_within_circle = (dist_matrix_[cur_point, :] < min_dist).squeeze()

        # Eliminate those points from ever being chosen
        dist_matrix_[points_within_circle, :] = np.nan
        dist_matrix_[:, points_within_circle] = np.nan

        # If no points are able to be chosen, stop
        if np.all(np.isnan(dist_matrix_[cur_point, :])):
            break

        # Find the nearest neighbour that is not within that circle
        # Choose it as the next point
        nearest_outside_point = np.nanargmin(dist_matrix_[cur_point, :])

        chosen_points.append(nearest_outside_point)

        # Make sure the current point can no longer be chosen
        dist_matrix_[cur_point, :] = np.nan
        dist_matrix_[:, cur_point] = np.nan

        cur_point = nearest_outside_point

    chosen_points = np.stack(chosen_points)

    return chosen_points
```

We can nicely visualize this as below. You can identify each seed point as the only point without an incoming arrow. The arrows indicate the point chosen at each iteration, being the closest point at least `min_dist` away. The circles have radius `0.5 * min_dist`, giving nicely 'packed' solutions.

![Some WSP space-filling designs](figures/some_wsp_solutions.svg)

At the moment, the starting position has a significant effect on the overal structure of the different solutions.