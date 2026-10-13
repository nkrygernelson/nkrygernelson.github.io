---
layout: default
title: My Projects
---

## Here is what I've been working on

### 1. [POD: a modular diffusion research codebase](./POD.html)

A research codebase for training and sampling diffusion models, built so that
the noise schedule, prediction target (noise / clean sample / score / velocity),
time coordinate, sampler and training procedure are all independent components
swapped from the command line via Hydra.

The idea is to make "does this design choice actually matter?" a question you can
answer with a config override instead of a rewrite. It implements four
parameterizations behind a single matrix inversion, four samplers (Euler,
Euler-Maruyama, DDIM, DEIS), log-SNR time reparameterization, classifier-free
guidance and reflow.

*Python, PyTorch, Hydra* · **[Read more](./POD.html)** · **[Source on GitHub](https://github.com/nkrygernelson/POD)**

---

More projects to come.

---
[⬅ Back to Home](./index.html)
