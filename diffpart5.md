---
layout: default
title: "Diffusion Crash Course, Part 5: Solvers"
---

# Part 5: Solvers

*Part 5 of [A Diffusion Crash Course](./diffusion.html). Previous: [Part 4, Unification](./diffpart4.html).*

> **Work in progress.** This part has not been written yet. It will cover the
> numerical solvers used to integrate the probability-flow ODE and the reverse
> SDE at sampling time: DDIM, DEIS, and Euler-Maruyama among them.

In the meantime, [POD](./POD.html) implements four of these solvers behind a
common interface, so the code is a reasonable preview of what this part will
cover.

---

[⬅ Part 4: Unification](./diffpart4.html) · [Series index](./diffusion.html)
