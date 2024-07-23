---
title: VocabReader
date: 2024-04-10
tags:
  - nlp
  - code
math: true
draft: false
---

The expanded code snippet found in [[blog/vocab_generation#Implementing a vocabulary reader]]

```python
import csv
from copy import deepcopy
from functools import cache


class VocabReader:
    def __init__(self, fp: str):
        self.fp = fp

        self.pointer = 0

        self.peeked_token = None
        self.peeked_count = None
        self.peek_offset = None

    def peek(self):
        """Look at the next line, and put into cache

        Raises:
            StopIteration: next line does not exist

        Returns:
            str: the next token
        """
        if self.terminated:
            raise StopIteration

        if self.peeked_token is None:
            with open(self.fp) as f:
                f.seek(self.pointer)

                line = f.readline()

                self.peek_offset = f.tell()

            token, count = next(csv.reader([line]))

            self.peeked_token = token
            self.peeked_count = int(count)

            return token
        else:
            return self.peeked_token

    def keep(self):
        """Empty the cache

        Returns:
            int: count
        """
        self.pointer = self.peek_offset

        count = deepcopy(self.peeked_count)

        self.peeked_token = None
        self.peeked_count = None
        self.peek_offset = None

        return count

    def find(self, token):
        """Iterate through the csv, looking for a specific token

        Args:
            token (str): the token to match

        Raises:
            StopIteration: list has been completely traversed already

        Returns:
            int: the count of the token in this file
        """
        if self.terminated:
            raise StopIteration

        with open(self.fp) as f:
            f.seek(self.pointer)

            reader = csv.reader(f)

            for cur_token, count in reader:
                # Token is larger than the token we're looking for
                # So token does not exist
                if cur_token > token:
                    return 0

                elif token != cur_token:
                    continue

                # If we have a match, return the count of the token
                return int(count)

    @cache
    def __len__(self):
        with open(self.fp) as f:
            for _ in f:
                pass

            reader_len = f.tell()
        return reader_len

    def __hash__(self) -> int:
        return hash(self.fp)

    @property
    def terminated(self):
        return self.pointer >= len(self)

```
