---
title: Efficient Vocabulary Generation for Very Large Corpora
date: 2024-04-10
tags:
  - nlp
  - code
draft: false
math: true
---

{{< toc >}}

For an ongoing project I had to perform topic clustering on a large corpus of diverse and very long news articles. [BERTopic](https://maartengr.github.io/BERTopic/index.html) usually works very well for such use-cases, with a variety of memory saving techniques already being implemented. Where I ran into trouble is a failry innocuous intermediate step.

After embedding the documents, reducing the embedding feature dimensions and clustering the corpus, a second set of features are estimated for each cluster. Specifically, BERTopic uses the class-based TF-IDF scores to generate a topic-token matrix. The clustering in document embedding space is assumed to be non-convex, making estimation of a central tendency infeasible[^non-convex-central-tendency]. By extent, computing the distance between topics is diffcult or intractable. [By using the topic-token TF-IDF representations, inter topic distance can be estimated in a more reliable and interpretable manner](https://maartengr.github.io/BERTopic/algorithm/algorithm.html#4-bag-of-words). [Another benefit is that we immediately get access to textual representations of our topics](https://maartengr.github.io/BERTopic/algorithm/algorithm.html#5-topic-representation).

[^non-convex-central-tendency]: for example, imagine our cluster is a [thin ring](https://scikit-learn.org/stable/auto_examples/cluster/plot_cluster_comparison.html#sphx-glr-auto-examples-cluster-plot-cluster-comparison-py) (or its $k$-dimensional equivalent). The mean would lie in the middle, far away from the cluster. The mode is spread evenly across the surface of the ring, and the median is not clearly [defined](https://en.wikipedia.org/wiki/Geometric_median). Choosing a single point to represent the cluster remains difficult

But this assumes access to a vocabulary estimated over the entire corpus. For most cases, this vocabulary needs to include higher order n-grams, while simultaneously removing semantically poor words. Human language being what it is, this means the vocabulary size will quickly exceed working memory.

## The Problem

Using the default `CountVectorizer` (a standard [sklearn](https://scikit-learn.org/stable/) function). It simply iterates over the corpus and tracks how often each n-gram token occurs. Once ran, it only keeps the top $n$ tokens. For my corpus, however, this proved to be a problem, with an intermediate vocabulary that just would not fit into memory.

In general, the issue of open-ended vocabularies has been solved by using sub-word tokenization. After all, language models are more or less required to hold two $d\times |\mathcal{V}|$ layers in memory; one to encode each token $t\in\mathcal{V}$ and one to convert the $d$ dimensional contextual embeddings into $|\mathcal{V}|$-sized softmax. Reasonable sized vocabularies are essentially a pre-requisite. Sub-word tokenization strikes a trade-off between vocabulary size and sequence length, at the cost of tokens that are (indivually) semantically meaningless. This invalidates this approach; I'm interested in semantic topic similarity, not whether or not their counts of sub-word tokens happen to overlap.

Using `HashingVectorizer`, I could retain the full word tokens, and it keeps no state, greatly reducing the memory strain. Specifically, it only retains a token's hash while discarding the actual orthographic form of the tokens. This makes post-hoc inspection of the features also impossible. In the end, I would like to have a representation of each topic available. Not mention, we'd still have to have keep a $|\mathcal{V}|$ dictionary in memory.

To it's merit, BERTopic provides an `OnlineCountVectorizer` meant to solve exactly this problem. Instead of estimating a vocabulary over the entire corpus, it uses small batches sampled from the corpus to learn an intermediate vocabulary, dropping any tokens that are occured too infrequently or exceed the desired vocabulary size. While mini-batching, as usual, alleviates the memory woes, it results in an inexact vocabulary at the end. Intermediate counts are forgotten, and which words make the vocabulary and which don't largely relies on the order of documents. Ususally, the words that occur less frequently are also exactly the words that carry the most semantic information.

I'd like to think we can do better.

## The Solution (?)

Online estimation, or mini-batching, is not a bad idea altogether though. If we iterate through our corpus until we have a vocabulary of size $|\mathcal{V}_{i}|=n_{\text{max_terms}}$, we'd be left with $m$ separate relatively small vocabularies. Each of these we can easily store as a list on disk, meaning the memory footprint is as small as possible. At the end, to construct our final vocabulary we'd just have to search through the $m$ lists and sum their occurences to get their exact count.

... except that this incurs a $\mathcal{O}\left(m\cdot n \cdot n_{\text{max_terms}}\right)$ cost search operation. For each one of $n$ words, we'd have to look through (at worst) $n_{\text{max_terms}}$ words in $m$. For infrequent words (again, exactly the class of words we care most about) the probability of a word being in a list is relatively small, meaning we're bound to hit worst-case performance often.

Luckily, we can exploit two properties:

1. We mostly tend to repeat ourselves
2. Alphabetic ordering exists

Neither is a particularly profound statement, but both will prove crucial. Alphabetic ordering implies a natural early termination condition. While scanning through the $m$ intermediate *sorted* vocabularies, we can stop as soon as the key term exceeds the value of the query term, secure in knowing that the term did not occur in that list.

In the example below, we scan from 'aa' to 'ac'. Upon reaching 'ac', we know that we *would have* matched 'ab' if it had been in the list.

```txt
Query | Keys
      | aa
      | aaa
      | aab
      | ...
   ab | ac
```

Of course, any number of tokens might appear between 'aa' and 'ab'. 'antidisestablishmentarianism' technically has a lower alphabetic order than 'ax'. Language is (in theory) infinitely productive, and including n-grams only exacerbates this behaviour.

This is where the other property comes into play. Effective communication requires using common enough words; a text full of new or unseen words is unreadable. As a result, we'll likely only see a few misses across the vocabulary lists.

It also has a second consequence. While each list contains the vocabulary of a random subset of the entire corpus, each list individually likely introduces few new tokens. As a result, once we know the location of a token in a list, the same token in other lists is likely to be in a relatively similar position.

These facts put together, the average search length likely scales far, *far* under quadratic. Instead, we have a much more tractable $\Theta\left(m\cdot n\cdot n_{\text{search}}\right)$[^average_case], where $n_{\text{search}}$ is the average length of the list to traverse before a match, where I've argued $n_{\text{search}}\ll n$. We either find the term we're looking for, other terminate after a few misses.

[^average_case]: the $\Theta$ is meant to denote a tight upper and lower bound on performance. In other words, I'm fairly confident that average performance scales like this

### Implementing a vocabulary reader

The devil lies in the (implementation) details. Let's assume the vocabulary lists are stored as `.csv` files, with each row containing just the token, and its count within the batch. We can *efficiently* achieve the desired behaviour by writing a class with three methods:[^peek_and_keep]

[^peek_and_keep]: peek and keep being antigrams here is a happy, but unintended coincidence

1. **Peek**: look at the next line of the vocabulary list, return the token on that line, and cache the token count
2. **Keep**: return the peeked token's count and empty the cache
3. **Find**: scan through the remaining rows in the list, and return the count of the token in that list if it exists, or return 0 if the token is definitely not in the list

Efficient is the operative term here. We do not want to load all lists into memory. Instead, we want to just load in single rows, and move ever further down the list. Python allows moving down the file stream using the `tell` (returns the current position in the data stream) and `seek` methods (moves the data stream to the provided position).

The Python backbone for this would look something like:

```python
class VocabReader:

  def __init__(self):
    # Set a pointer
    # This is the line of the list we're currently at
    self.i = 0

  def peek(self) -> str:
    # Look at the next token, and cache it
    # Returns the token
    ...

  def keep(self) -> int:
    # Empty the cache
    # Returns the cached token's count
    ...

  def find(self, token: str) -> int:
    # Iterate through the vocab list, looking for a specific token
    # Returns the token's count if it exists, or 0
    ...

  @property
  def terminated(self):
    ...
```

We want each of the $m$ `VocabReader` instances to be roughly in sync throughout the iterations, while ensuring that we can stop iterating once we exceed the alphabetic order of the token we're looking for. This is where the `peek` and `keep` methods come into play. When choosing a token to process, we can iterate through all the `VocabReader` and use `peek` to find the token with the smallest alphabetic order. Then we use `keep` to fetch its count, increase the pointer by 1, and iterate through all $m-1$ remaining `VocabReader` instances and use `find` to fetch the count of that token in the other lists.

At this point we have thus expended very little memory to storing the vocabularies, while ensuring that the additional processing cost is as little as possible.

## Putting it all together

Ultimatel, all we want is a vocabulary list of tokens with at most length `n_{\text{max_terms}}`. We can process the intermediate vocabularies fairly efficiently, but we still need to track and store the large collated vocabulary somehow. For that we only need two more, relatively simple data structures.

1. **Heap**: quickly insert tokens into a 'sorted' vocabulary, with $\mathcal{O}(1)$ access to the least frequent tokens. Python implements this through the `heapq` module.

2. **LimitedCounter**: Some data structure tracking which tokens we've already processed. A `set` or `dict` wouldn't work, as this would inevitably hash all $n$ tokens, the exact issue we want to avoid. Rather, we define a special instance of a `Counter` that deletes an entry once it has been seen $m$ times. Once we've seen a token $m$ times, we can be certain it has been seen by all $m$ `VocabReader` instances, and won't appear again.

Putting it all together, the vocabulary collation function would look something like this:

```python
def collate_vocabs(
  vocab_fps: typing.List[str],
  min_df: int,
  max_features: int
) -> typing.List[typing.Tuple[int, str]]:
  # Import all the stored dicts as a collection of data streams
  readers = {VocabReader(vocab_fp) for vocab_fp in vocab_fps}

  # Use a limited counter to track which tokens have been seen already
  seen_tokens = LimitedCounter(limit=len(readers))

  # Use a heap to construct the vocabulary
  vocab_heap = []
  heapq.heapify(vocab_heap)

  # Continue until all data streams have been exhausted
  while not all(reader.terminated for reader in readers):
      # Iterate over each data stream, looking only at the next line
      # and cache the result
      possible_continuations = []
      for reader in readers:
          try:
              possible_continuations.append((reader.peek(), reader))
          # Handle a stream ending early
          except StopIteration:
              continue

      # From all the next lines, choose the 'smallest' token
      # This is the token we process in this iteration
      token, cur_reader = min(possible_continuations, key=lambda x: x[0])

      # Tell stream from which the current token came to clear its cache
      count = cur_reader.keep()

      # Check if we've seen the term already
      # If so finish this iteration
      if token in seen_tokens:
          seen_tokens.add(token)
          continue

      # If not, we scan through all the other streams until
      # we find the same token and its count
      for other_reader in readers - {cur_reader}:
          try:
              count += other_reader.find(token)
          except StopIteration:
                  continue

      # If the count is too small to include, don't add it
      if count < min_df:
          continue

      # Finally, add the (count, token) tuple to the heap, removing the
      # lowest count token when necessary
      elif len(vocab_heap) < max_features:
          heapq.heappush(vocab_heap, (count, token))

      elif len(vocab_heap) == max_features:
          heapq.heappushpop(vocab_heap, (count, token))

      seen_tokens.add(token)

  vocab = list(sorted(vocab_heap, key=lambda x: x[0])[::-1])

  return vocab
```

Voilà, we have a dictionary of exactly $n_{\text{max_terms}}$ where the count of each item is the same as if we'd computed on the entire corpus in one go.At no point did the memory consumption exceed the number of tokens present in a single batch, allowing for work on very large datasets without RAM being too large a constraint.

I added one more variable here than strictly necessary: `min_df`. Very infrequent words likely only add noise, so the user can (somewhat arbitrarily) already cull those terms before being added to the heap. As a result, we can also be certain that all tokens in our dictionary occur at least `min_df`.

<!-- That said, this vocabulary is not exact. Tokens that, simply due to chance, occur fewer times than `min_df` in a single batch will get removed early in the process, whether or not it occurs often enough in the entire corpus. However, the error is at most $(\mathtt{max\_df}-1)(m-1)$, and is probably much smaller for frequent of tokens[^2].

[^2]: For a proper analysis, the error rate probably involves the use of a [multinomial](https://en.wikipedia.org/wiki/Multinomial_distribution) or [multivariate hypergeometric](https://en.wikipedia.org/wiki/Hypergeometric_distribution#Multivariate_hypergeometric_distribution) distribution. -->