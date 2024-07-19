---
title: Efficient Vocabulary Generation for Very Large Corpora
date: 2024-04-10
tags:
  - nlp
  - code
draft: true
math: true
---

{{< toc >}}

For an ongoing project, I had to perform topic clustering on a large corpus of diverse and very long news articles. [BERTopic](https://maartengr.github.io/BERTopic/index.html) usually works very well for such use-cases, with a variety of memory saving techniques already being implemented. Where I ran into trouble was a usually innocuous intermediate step.

After embedding the documents, reducing the embedding feature dimensions, and clustering the corpus, a second set of features is estimated for each cluster. Specifically, BERTopic uses the class-based TF-IDF scores to generate a topic-token matrix. The clustering in document embedding space is assumed to be non-convex, making estimation of a central tendency infeasible[^non-convex-central-tendency]. By extension, computing the distance between topics is diffcult or intractable. [Using topic-token TF-IDF representations instead, inter topic distance can be robustly estimated](https://maartengr.github.io/BERTopic/algorithm/algorithm.html#4-bag-of-words). [Another benefit is that we immediately get access to textual representations of our topics](https://maartengr.github.io/BERTopic/algorithm/algorithm.html#5-topic-representation).

[^non-convex-central-tendency]: for example, imagine our cluster is a [thin ring](https://scikit-learn.org/stable/auto_examples/cluster/plot_cluster_comparison.html#sphx-glr-auto-examples-cluster-plot-cluster-comparison-py) (or its $k$-dimensional equivalent). The mean would lie in the middle, far away from the cluster. The mode is spread evenly across the surface of the ring, and the median is not clearly [defined](https://en.wikipedia.org/wiki/Geometric_median). Choosing a single point to represent the cluster remains difficult

But this assumes access to a vocabulary estimated over the entire corpus. For most cases, this vocabulary needs to include higher order n-grams, while simultaneously removing semantically poor words. Human language being what it is, this means the vocabulary size will quickly exceed working memory.

## The Problem

Using the default `CountVectorizer` (a standard [sklearn](https://scikit-learn.org/stable/) function) proved to be a problem, with an intermediate vocabulary that just would not fit into memory. It simply iterates over the corpus and tracks how often each n-gram token occurs. Once ran, it only keeps the top $n$ tokens.

In general, the issue of open-ended vocabularies has been solved by using sub-word tokenization. After all, language models are more or less required to hold two $d\times |\mathcal{V}|$ layers in memory; one to encode each token $t\in\mathcal{V}$ and one to convert the $d$ dimensional contextual embeddings into $|\mathcal{V}|$-sized softmax. Reasonable sized vocabularies are essentially a prerequisite. Sub-word tokenization strikes a trade-off between vocabulary size and sequence length, at the cost of tokens that are (individually) semantically meaningless. This invalidates this approach; I'm interested in semantic topic similarity, not whether or not their counts of sub-word tokens happen to overlap.

Using `HashingVectorizer`, I could retain the full word tokens, and it keeps no state, greatly reducing the memory strain. Specifically, it only retains a token's hash while discarding the actual orthographic form of the tokens. This makes post-hoc inspection of the features also impossible. In the end, I would like to have a representation of each topic available. Not mention, we'd still have to have keep a $|\mathcal{V}|$ dictionary in memory.

To it's merit, BERTopic provides an `OnlineCountVectorizer` meant to solve exactly this problem. Instead of estimating a vocabulary over the entire corpus, it uses small batches sampled from the corpus to learn an intermediate vocabulary, dropping any tokens that occur too infrequently or exceed the desired vocabulary size. Mini-batching alleviates the memory woes, but it results in an inexact token frequencies at the end. Intermediate counts are forgotten, and which words make the vocabulary and which don't largely relies on the order of documents. Ususally, the words that occur less frequently are also exactly the words that carry the most semantic information, and it is here that this approach is most likely to err.

I'd like to think we can do better.

## The Solution (?)

Online estimation, or mini-batching, is not a bad idea altogether, though. If we iterate through our corpus until we have a vocabulary of size $|\mathcal{V}_{i}|=n_{\text{max_terms}}$, we'd be left with $m$ separate relatively small vocabularies. Each of these we can easily store as a list on disk, meaning the memory footprint is as small as possible. In the end, to construct our final vocabulary, we'd just have to search through the $m$ lists and sum their occurrences to get their exact count.

... except that this incurs a $\mathcal{O}\left(m\cdot n \cdot n_{\text{max_terms}}\right)$ cost search operation. For each one of $n$ words, we'd possibly have to look through $n_{\text{max_terms}}$ words in $m$. For infrequent words (again, exactly the class of words we care most about), the probability of a word being in each mini-batch vocabulary list is relatively small, meaning we're bound to hit worst-case performance often.

Luckily, we can sort words alphabetically. Once sorted, we can find the token in each independent list in $\mathcal{O}(1)$ time. The sorted lists can been seen a set of stacks, each with an independent pointer. At the top of the stack we have the smallest value item, i.e. the vocabulary's lowest order token.

```txt
       0    0    0    0
--> [  a | aa |  a | ab ] smallest = a
    [ aa |  b | ab | ac ]
    [ ac | ba | ac | ba ]

processed = []
```

At this point we scan the top of the stacks for the smallest token, 'a', and for stacks that have it at the top we pop it off (or equivalently increase the pointer by 1). All stacks that *did not* have that token at the top are guaranteed to only have higher value tokens, and we ignore those stacks. In other words, if the stack does not have the token on top, it simply does not have it.

```txt
       1    0    1    0
--> [ aa | aa | ab | ab ] smallest = aa
    [ ac |  b | ac | ac ]
    [ ba | ba |  c | ba ]

processed = [a]
```

After processing the tokens, we get to a situation that is remarkably similar to the start. Once again, the smallest value token for each stack is guaranteed to be at the top, and we need only scan the top layer. Rinse, lather and repeat.

```txt
       2    1    1    0
--> [ ac |  b | ab | ab ] smallest = ab
    [ ba | ba | ac | ac ]
    [ bc | bb |  c | ba ]

processed = [a, aa]
```

To process the entire vocabulary, we simply gradually move down each stack until exhausting the values within it. In each iteration, because we started with sorted stacks and we select the smallest element, we have guarantees that eliminate the need for search the rest of the list.

### Implementing a vocabulary reader

The devil lies in the (implementation) details. Let's assume the vocabulary lists are stored as `.csv` files, with each row containing just the token, and its count within the batch. We can *efficiently* achieve the desired behaviour by writing a class with just two methods:[^peek_and_keep]

[^peek_and_keep]: peek and keep being antigrams here is a happy, but unintended coincidence

1. **Peek**: look at the next line of the vocabulary list, return the token on that line, and cache the token count
2. **Keep**: return the peeked token's count and empty the cache

Efficient is the operative term here. We do not want to load all lists into memory. Instead, we want to just load in single rows, and move ever further down the list. Python allows moving along the file stream using the `tell` and `seek` methods. The former returns the current position in the data stream, whereas the latter moves the data stream to the provided position. We can use these to set our pointer.

The Python backbone for this would look something like:

```python
class VocabReader:

    def __init__(self):
        # Set a pointer
        # This is the current location in the data stream
        self.pointer = 0

    def peek(self) -> str:
        # Look at token located at `self.pointer`
        # Returns the token
        ...

    def keep(self) -> int:
        # Returns the count of the peeked token
        ...

    @property
    def terminated(self) -> bool:
        # Whether or not the list has been exhausted
        ...
```

In the background, the `VocabReader` instances can cache the result of peek, only opening the file when the cache is empty. This way, we're only opening the file once per token.

At this point we have offloaded the vocabulary entirely to the disk, while the added compute cost is $\mathcal{O}(|\mathcal{V}|)$. The initial processing has the same cost, so we've essentially doubled the compute cost (neglibe in big-O terms).

## Finishing Touches

Ultimately, all we want is a list with at most `n_{\text{max_terms}}` tokens in it. We can process the intermediate vocabularies efficiently, but we still need to track and store the large collated vocabulary somehow. For that, we only need three more, relatively simple data structures.

1. **Heap**: quickly insert tokens into a 'sorted' vocabulary, with $\mathcal{O}(1)$ access to the least frequent tokens. Python implements this through the `heapq` module.

2. **LimitedCounter**: Some data structure tracking which tokens we've already processed. A `set` or `dict` wouldn't work, as this would inevitably hash all $n$ tokens, the exact issue we want to avoid. Rather, we define a special instance of a `Counter` that deletes an entry once it has been seen $m$ times. Once we've seen a token $m$ times, we can be certain it has been seen by all $m$ `VocabReader` instances, and won't appear again.

3. **token2reader**: a mapping from all tokens on top of a stack to the readers that have that token on top. This way, we can quickly fetch which `VocabReader` instances need updating. This is easily implemented using a `collections.defaultdict[list]`

Putting it all together, the vocabulary collation function would look something like this:

```python
def collate_vocabs(
  vocab_fps: typing.List[str],
  min_df: int,
  max_features: int
) -> typing.List[typing.Tuple[int, str]]:
    # Gather all of the `VocabReader` instances
    readers = [VocabReader(vocab_fp) for vocab_fp in vocabs]

    # Use a limited counter to track which tokens have been seen already
    seen_tokens = LimitedCounter(limit=len(readers))

    # Use a heap to keep track of the most frequent tokens
    # This allows for fast inserts and fast access to minimum
    vocab_heap = []
    heapq.heapify(vocab_heap)

    # Iterate until we've exhausted every vocabulary stack
    while not all(reader.terminated for reader in readers):
        # Create a new token2reader instance
        token2reader = create_token2reader()

        # Peek at the next token on each reader's stack
        for reader in readers:
            try:
                reader_token = reader.peek()
            except StopIteration:
                continue

            # Add the token and reader to the `token2reader` mapping
            token2reader[reader_token].append(reader)

        # Find the token with the minimum value across all readers
        min_val_token = min(token2reader.keys())

        # Fetch all the readers that have the min_val_token on top
        vocab_readers_with_matches = token2reader[min_val_token]

        # Sum up the counts for the min_val_token in each stack
        token_count = 0
        for reader in vocab_readers_with_matches:
            token_count += reader.keep()

        # If the count is too small to include it in the final vocab,
        # remove it
        if token_count < min_df:
            continue

        # Finally, add the (count, token) tuple to the heap,
        # removing the lowest count token when necessary
        elif len(vocab_heap) < max_features:
            heapq.heappush(vocab_heap, (token_count, min_val_token))

        elif len(vocab_heap) == max_features:
            heapq.heappushpop(vocab_heap, (token_count, min_val_token))

        # Add the token to the seen tokens collection
        seen_tokens.add(min_val_token)

    # Finally construct and output the final vocabulary
    # `vocab_heap` stores the tokens as (count, token) tuples
    vocab = {term: i for i, term in enumerate(sorted(map(lambda x: x[1], vocab_heap)))}

    return vocab
```

Voilà, we have a dictionary of exactly $n_{\text{max_terms}}$ where the count of each item is the same as if we'd computed on the entire corpus in one go. At no point did the memory consumption exceed the number of tokens present in a single batch, allowing for work on very large datasets without RAM being a constraint. I added one more variable here than strictly necessary: `min_df`. Very infrequent words likely only add noise, so the user can (somewhat arbitrarily) cull those terms before being added to the heap. As a result, we can also be certain that all tokens in our dictionary occur at least `min_df` times.
