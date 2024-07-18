---
title: Efficient Vocabulary Generation for Very Large Corpora
date: 2024-04-10
tags:
  - nlp
  - code
toc: false
draft: true
math: true
---

{{< toc >}}



At a first glance, this sounds like a particularly bad idea. For a set of $m$ lists, sequentially searching through its $n$ entries gives us a time complexity of $\mathcal{O}\left(m\cdot n^2\right)$. Natural language is (in principle) infinitely productive, and for even moderately sized corpora, the number of distinct token n-grams will explode.

Luckily, we can exploit two properties:

1. We tend to repeat ourselves
2. alphabetic ordering exists

Neither of these should be a revelation to you, but both will prove crucial. The latter provides a convenient early termination condition. While scanning through the auxiliary lists, as soon as we find a token that has an alphabetic value greater than the token we are searching for, we can stop, as the token is not present in the list. In the example below, we scan from 'aa' to 'ac'. Upon reaching 'ac', we can be certain we would have matched 'ab' if it had been in the list.

```txt
Token | List
      | aa
      | aaa
      | aab
      | ...
   ab | ac
```

Of course, any number of tokens might appear between 'aa' and 'ab', after all 'antidisestablishmentarianism' has a lower alphabetic order than 'aorta', and allowing n-grams only worsens this.

This is where the former property comes into play. While each list contains the vocabulary of a random subset of the entire corpus, each list individually likely introduces few new tokens, purely because effective communication requires using common enough words. Tokens that are unique to a subset, are likely less important to the overall corpus (again, assuming a random distribution of documents across the batches). By removing closed class or stop words, we can strengthen this effect further, as non-semantically relevant token n-grams get culled from the list, and the sparsity increases.

These two facts put together, the average case performance is likely sub-quadratic, i.e., a much more tractable $\mathcal{O}\left(m\cdot n\cdot n_{\text{search}}\right)$, where $n_{\text{search}}$ is the average length of the list to traverse before a match, which should ideally be $n_{\text{search}}<<< n$.

## Implementing a vocabulary reader

The devil, as they say, lies in the details. Let's assume the vocabulary lists are stored as `.csv` files, with each row containing just the token, and its count within the batch. We can *efficiently* achieve the desired behaviour by writing a class with three methods:[^1]

1. **Peek**: look at just the next line of the vocabulary list, return the token on that line, and cache the token count
2. **Keep**: return the peeked token's count and empty the cache
3. **Find**: scan through the remaining rows in the list, and return the count of the token in that list if it exists, or return 0 if the token is definitely not in the list

Efficient is the operative term here. We do not want to load the entire list into memory. Instead, we want to just load in single rows, and then move ever further down the list. Python allows moving down the file stream using the `tell` (returns the current position in the data stream) and `seek` methods (moves the data stream to the provided position). The caching prevents repeating work when choosing which token to process at each iteration.

### Subheader

The Python backbone for this [[snippets/VocabReader]] would look something like:

```python
class VocabReader:
  def peek(self):
    # Look at the next line, and caches it
    # Returns (str), the next token
    ...

  def keep(self):
    # Empty the cache
    # Returns (int), the cached token's count
    ...

  def find(self, token):
    # Iterate through the vocab list, looking for a specific token
    # Args:
    #   token (str), the token to match
    # Returns (int), the token's count if it exists, or 0
    ...

  @property
  def terminated(self):
    ...
```

We want the `VocabReader` instances to be roughly in sync throughout the iterations, while ensuring that we can stop iterating once we exceed the alphabetic order of the token we're looking for. This is where the `peek` and `keep` methods come into play. When choosing a token to process, we can iterate through all the `VocabReader` and use `peek` to find the token with the smallest alphabetic order. Then we use `keep` to fetch the count, and allow the reader to move to the next line, and finally iterate through all remaining `VocabReader` instances and use `find` to fetch the count of that token in the other lists.

## Putting it all together

We now have most of the tools necessary to collate all the independent vocab lists together. Remember, ultimately we want a vocabulary list of tokens of at most length `max_features`, and where each token occurs in at least `min_df` documents.

To get there, we need two more, relatively simple data structures.

1. A heap to quickly insert tokens into a 'sorted' vocabulary, and $\mathcal{O}(1)$ access to the least frequent token. Python implements this through the `heapq` module.

2. Some data structure tracking which tokens we've already processed. A `set` or `dict` wouldn't work, as this would inevitably cache all found tokens, the exact issue we want to avoid. Rather, we define a special instance of a `Counter` that deletes an entry once it has been seen a certain number of times. In our case, that is the number of vocab lists we have; if a token has been seen $n$ times, we can be certain it has been seen by all `VocabReader` instances, and won't appear again.

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

The returned list now contains a sorted collection of `(count, token)` tuples, from which one can easily construct the final vocabulary. At no point did the memory consumption exceed the number of tokens present in a single batch, allowing for work on very large datasets without RAM being too large a constraint.

That said, this vocabulary is not exact. Tokens that, simply due to chance, occur fewer times than `min_df` in a single batch will get removed early in the process, whether or not it occurs often enough in the entire corpus. However, the error is at most $(\mathtt{max\_df}-1)(m-1)$, and is probably much smaller for frequent of tokens[^2].

[^1]: Peek and keep effectively being antigrams here is a happy coincidence.

[^2]: For a proper analysis, the error rate probably involves the use of a [multinomial](https://en.wikipedia.org/wiki/Multinomial_distribution) or [multivariate hypergeometric](https://en.wikipedia.org/wiki/Hypergeometric_distribution#Multivariate_hypergeometric_distribution) distribution.