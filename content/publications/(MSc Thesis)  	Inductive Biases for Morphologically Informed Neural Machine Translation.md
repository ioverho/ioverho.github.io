---
title: (MSc Thesis) Inductive Biases for Morphologically Informed Neural Machine Translation
date: 2022-08-03
tags:
  - meta-learning
  - machine-translation
  - morphology
  - MSc
---

{{< icons/icon vendor=bootstrap name="journal-text" className="logo" >}}[UvA Dare](https://scripties.uba.uva.nl/search?id=record_50777) | {{< icons/icon vendor=simple name="github" className="github logo" >}}[Github](https://github.com/ioverho/morph_tag_lemmatize)

## Abstract

The words which comprise human language, are themselves complex units. Each carries meaning in isolation, but their structure is often altered to accommodate larger compositions, according to some set of grammatical rules. The processes that determine how the word is formed, and for which occasions it applies, are heavily patterned. Human beings, whether conscious of it or not, can leverage these patterns to quickly generate new word-forms when situations necessitate them. Neural language models, as used in neural machine translation systems, likely do not learn these patterns and show limited capacity in decomposing words as humans would. Instead, they merely learn to associate certain string segments with others, imitating the data used to train them. Morphologically rich languages, with very complex word-formation processes, have word-forms that occur only in rare situations. As a result, translation performance suffers when neural language models are required to produce word forms it has not seen before. This thesis explores these morphological word formation processes, and how neural language models interact with them. Ultimately, it seeks to adapt pre-trained neural machine translation models, towards greater understanding of morphology. The requirement of pre-trained systems, a practical necessity for many researchers, invalidates previous techniques presented for this task. As such, a novel gradient-based meta-learning framework is formulated, which only alters the sampling method to incorporate morphological information implicitly. This process is coined 'morphological cross-transfer', and separates meaning from function in the learning phase. For this, strong automated morphological analyzers are required. This is covered in detail, and neural systems for this task are re-implemented, before verifying their usage on natural language corpora. A second required component is a measurable notion of morphological competence. This too is covered in some detail, presenting a novel methodology that extends easily to designing task samplers typically found in meta-learning setups. Finally, experiments with morphological cross-transfer indicate slightly improved translation systems, and slightly improved dedicated morphological inflectors, although the objectives are not achieved simultaneously. This opens up new avenues of research into post-hoc adaptation techniques for providing neural language models with desired inductive biases.

## Citation

```bibtex
@thesis{verhoevenInductiveBiasesMorphologically2022,
  type = {{{MSc}}},
  title = {Inductive {{Biases}} for {{Morphologically Informed Neural Machine Translation}}},
  author = {Verhoeven, Ivo},
  year = {2022},
  month = aug,
  address = {Amsterdam},
  school = {University of Amsterdam},
}
```
