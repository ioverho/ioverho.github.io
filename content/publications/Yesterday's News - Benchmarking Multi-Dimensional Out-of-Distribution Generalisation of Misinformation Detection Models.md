---
title: "Yesterday's News: Benchmarking Multi-Dimensional Out-of-Distribution Generalisation of Misinformation Detection Models"
date: 2025-11-23
tags:
  - misinformation
  - generalisation
  - PhD
---

<img src="https://upload.wikimedia.org/wikipedia/commons/1/10/MIT_Press_logo.svg" style="background-color: white;margin: 0;height: 1em;display: inline;"></img> [Computation Linguistics](https://direct.mit.edu/coli/article/doi/10.1162/COLI.a.585/134523/Yesterday-s-News-Benchmarking-Multi-Dimensional)  | {{< icons/icon vendor=simple name="arxiv" className="logo arxiv" >}}[arXiv](https://arxiv.org/abs/2410.18122) | {{< icons/icon vendor=simple name="github" className="github logo" >}}[Github](https://github.com/ioverho/misinfo-general) | {{< icons/icon vendor=simple name="huggingface" className="logo hf" >}}[Data](https://huggingface.co/datasets/ioverho/misinfo-general)

## Abstract

This article introduces misinfo-general, a benchmark dataset for evaluating misinformation models’ ability to perform out-of-distribution generalization. Misinformation changes rapidly, much more quickly than moderators can annotate at scale, resulting in a shift between the training and inference data distributions. As a result, misinformation detectors need to be able to perform out-of-distribution generalization, an attribute they currently lack. Our benchmark uses distant labelling to enable simulating covariate shifts in misinformation content. We identify time, event, topic, publisher, political bias, misinformation type as important axes for generalization, and we evaluate a common class of baseline models on each. Using article metadata, we show how this model fails desiderata, which is not necessarily obvious from classification metrics. Finally, we analyze properties of the data to ensure limited presence of modelling shortcuts. We make the dataset and accompanying code publicly available at: {{< icons/icon vendor=simple name="github" className="github logo" >}}[ioverho/misinfo-general](https://github.com/ioverho/misinfo-general)

## Citation

```bibtex
@article{verhoeven2025yesterday,
  title={Yesterday’s News: Benchmarking Multi-Dimensional Out-of-Distribution Generalization of Misinformation Detection Models},
  author={Verhoeven, Ivo and Mishra, Pushkar and Shutova, Ekaterina},
  journal={Computational Linguistics},
  pages={1--58},
  year={2025},
  publisher={MIT Press 255 Main Street, 9th Floor, Cambridge, Massachusetts 02142, USA~…},
  url={https://direct.mit.edu/coli/article/doi/10.1162/COLI.a.585/134523/Yesterday-s-News-Benchmarking-Multi-Dimensional},
}
```
