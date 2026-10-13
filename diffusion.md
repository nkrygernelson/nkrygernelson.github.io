---
layout: default
title: "A Diffusion Crash Course"
---

# A Diffusion Crash Course

A crash course in diffusion: the fundamental math and concepts behind diffusion
models. The goal is to present the fundamental mathematical framework of modern
diffusion models.

### [Part 1: DDPM](./diffpart1.html)

The problem diffusion models actually solve, stated probabilistically. The fixed
forward process and its simulation-free marginal, why the true reverse kernel is
intractable, and how conditioning on a clean sample makes the objective
tractable without changing the minimizer.

### [Part 2: Score and SDEs](./diffpart2.html)

Moving to continuous time. ODEs and SDEs, a derivation of the Fokker-Planck
equation, the Langevin SDE and why the score exactly compensates for noise
injection, reversing an SDE via the probability-flow ODE, and score matching.

### [Part 3: Flow Matching](./diffpart3.html)

Flows, pushforward densities and the continuity equation. Why the marginal
velocity field is intractable, and how conditional flow matching recovers it
with a closed-form regression target that shares the same optimum.

### [Part 4: Unification](./diffpart4.html)

Writing the forward process in one form that covers all of the above. *Work in
progress.*

### [Part 5: Solvers](./diffpart5.html)

The numerical solvers used at sampling time. *Work in progress.*

---

[⬅ Back to Home](./index.html)
