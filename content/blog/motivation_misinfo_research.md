---
title: Why Work on Misinformation Detection and Generalization?
date: 2025-03-11
tags:
  - phd
  - generalisation
  - misinformation
math: false
---

The following is a transcript of a brief research pitch I gave to colleagues at our research institute.

---

In the past decade, we have entered a new political era: the era of post-truth. What is said has become less important than how often something is repeated. While misinformation is by no means a new phenomenon, the rise of social media sites has made it easier than ever before to share deliberately false or misleading content. When such content goes viral, recommendation algorithms ensure these malicious posts are seen by potentially millions of people within a very short timespan.

Misinformation posts that infect the social media feeds of this many people in this a short time-span, stop being merely an online nuisance and instead becomes a serious threat with dire real-life consequences. 

At present, our only weapon against online misinformation is manual moderation by expert journalists and fact-checkers. These moderators can remove posts, *before* they propagate through social media networks, thereby preventing future harm. While effective, these moderators are fighting a losing battle, akin to draining the oceans with a thimble.

There are simply too many social media posts to manually monitor each. My field, Natural Language Processing (NLP), offers a solution: automated moderation systems using language models trained on vast collections of prior content.

These systems, completely autonomously, would provide an estimate of the reliability of some input information, based on the reliability of similar examples the model had seen during training.

While these systems work well on academic benchmark datasets, my work has shown that under more realistic settings, these models fail, and their moderation decisions quickly become unreliable. This is not entirely unexpected. We expect these models to classify completely unseen information, from unseen authors, about unseen events. Their training data, however, consists entirely of yesterday’s news. Because social media networks are constantly trying to serve their users the latest, greatest, newest novelty, a gap between the input during the model’s training and inference quickly arises.

Bridging this gap, generalizing from seen to unseen examples, is a difficult task even for trained human experts. My work seeks to help models be better at bridging this generalization gap.

One strategy for this has been to analyse the properties of online content and misinformation, and using this to develop more realistic evaluation protocols of moderation systems. By incorporating elements of these evaluation protocols into the model training phase, we get models that are inherently more robust to changes in content than before.

Another strategy we have explored is the use meta-learning. Under this paradigm models don’t just learn; they learn to learn. In theory, these models can adapt to completely unseen input with minimal additional instructive examples. These moderation systems aren’t just robust to changes in the input, but can actively adapt when misinformation is produced about unseen events or entities.

Through my work, I hope to make automated moderation systems more reliable in realistic application settings. The hope is that these models can lighten the load for expert moderators, leveraging yesterday’s news to reduce future harms.
