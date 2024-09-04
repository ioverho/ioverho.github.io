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

{{< icons/icon vendor=simple name="arXiv" className="logo arxiv" >}}[ArXiv](https://arxiv.org/abs/2404.01822) | {{< icons/icon vendor=bootstrap name="link" className="logo" >}}[ACL Anthology](https://aclanthology.org/2024.findings-naacl.30/) | {{< icons/icon vendor=simple name="github" className="github logo" >}}[Github](https://github.com/rahelbeloch/meta-learning-gnns)

## Abstract

Community models for malicious content detection, which take into account the context from a social graph alongside the content itself, have shown remarkable performance on benchmark datasets. Yet, misinformation and hate speech continue to propagate on social media networks. This mismatch can be partially attributed to the limitations of current evaluation setups that neglect the rapid evolution of online content and the underlying social graph. In this paper, we propose a novel evaluation setup for model generalisation based on our few-shot subgraph sampling approach. This setup tests for generalisation through few labelled examples in local explorations of a larger graph, emulating more realistic application settings. We show this to be a challenging inductive setup, wherein strong performance on the training graph is not indicative of performance on unseen tasks, domains, or graph structures. Lastly, we show that graph meta-learners trained with our proposed few-shot subgraph sampling outperform standard community models in the inductive setup. We make our code publicly available.

## Citation

```bibtex
@misc{verhoeven2024morerealisticevaluationsetup,
      title={A (More) Realistic Evaluation Setup for Generalisation of Community Models on Malicious Content Detection},
      author={Ivo Verhoeven and Pushkar Mishra and Rahel Beloch and Helen Yannakoudakis and Ekaterina Shutova},
      year={2024},
      eprint={2404.01822},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2404.01822},
}
```
