---
title: A (More) Realistic Evaluation Setup for Generalisation of Community Models on Malicious Content Detection
date: 2024-04-02
tags:
  - misinformation
  - meta-learning
  - graph-neural-networks
  - generalisation
  - PhD
---

{{< icons/icon vendor=simple name="arxiv" className="logo arxiv" >}}[arXiv](https://arxiv.org/abs/2404.01822) | {{< icons/icon vendor=bootstrap name="link" className="logo" >}}[ACL Anthology](https://aclanthology.org/2024.findings-naacl.30/) | {{< icons/icon vendor=simple name="github" className="github logo" >}}[Github](https://github.com/rahelbeloch/meta-learning-gnns)

## Abstract

Community models for malicious content detection, which take into account the context from a social graph alongside the content itself, have shown remarkable performance on benchmark datasets. Yet, misinformation and hate speech continue to propagate on social media networks. This mismatch can be partially attributed to the limitations of current evaluation setups that neglect the rapid evolution of online content and the underlying social graph. In this paper, we propose a novel evaluation setup for model generalisation based on our few-shot subgraph sampling approach. This setup tests for generalisation through few labelled examples in local explorations of a larger graph, emulating more realistic application settings. We show this to be a challenging inductive setup, wherein strong performance on the training graph is not indicative of performance on unseen tasks, domains, or graph structures. Lastly, we show that graph meta-learners trained with our proposed few-shot subgraph sampling outperform standard community models in the inductive setup. We make our code publicly available.

## Citation

```bibtex
@inproceedings{verhoeven-etal-2024-realistic,
    title = "A (More) Realistic Evaluation Setup for Generalisation of Community Models on Malicious Content Detection",
    author = "Verhoeven, Ivo  and
      Mishra, Pushkar  and
      Beloch, Rahel  and
      Yannakoudakis, Helen  and
      Shutova, Ekaterina",
    editor = "Duh, Kevin  and
      Gomez, Helena  and
      Bethard, Steven",
    booktitle = "Findings of the Association for Computational Linguistics: NAACL 2024",
    month = jun,
    year = "2024",
    address = "Mexico City, Mexico",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.findings-naacl.30",
    doi = "10.18653/v1/2024.findings-naacl.30",
    pages = "437--463",
}
```
