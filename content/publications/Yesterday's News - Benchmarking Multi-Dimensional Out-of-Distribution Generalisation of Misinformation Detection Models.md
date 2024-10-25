---
title: "Yesterday's News: Benchmarking Multi-Dimensional Out-of-Distribution Generalisation of Misinformation Detection Models"
date: 2024-10-12
tags:
  - misinformation
  - generalisation
  - dataset
  - PhD
---

{{< icons/icon vendor=simple name="arxiv" className="logo arxiv" >}}[arXiv](https://arxiv.org/abs/2410.18122) | {{< icons/icon vendor=simple name="github" className="github logo" >}}[Github](https://github.com/ioverho/misinfo-general) | {{< icons/icon vendor=simple name="huggingface" className="logo hf" >}}[Data](https://huggingface.co/datasets/ioverho/misinfo-general)

## Abstract

This paper introduces `misinfo-general`, a benchmark dataset for evaluating misinformation models' ability to perform out-of-distribution generalisation. Misinformation changes rapidly, much quicker than moderators can annotate at scale, resulting in a shift between the training and inference data distributions. As a result, misinformation models need to be able to perform out-of-distribution generalisation, an understudied problem in existing datasets. We identify 6 axes of generalisation-time, event, topic, publisher, political bias, misinformation type-and design evaluation procedures for each. We also analyse some baseline models, highlighting how these fail important desiderata.

## Citation

```bibtex
@misc{verhoeven2024yesterdaysnewsbenchmarkingmultidimensional,
      title={Yesterday's News: Benchmarking Multi-Dimensional Out-of-Distribution Generalisation of Misinformation Detection Models},
      author={Ivo Verhoeven and Pushkar Mishra and Ekaterina Shutova},
      year={2024},
      eprint={2410.18122},
      archivePrefix={arXiv},
      primaryClass={cs.IR},
      url={https://arxiv.org/abs/2410.18122},
}
```
