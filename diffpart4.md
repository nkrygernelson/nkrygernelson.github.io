---
layout: default
title: "Diffusion Crash Course, Part 4: Unification"
---

# Part 4: Unification

*Part 4 of [A Diffusion Crash Course](./diffusion.html). Previous: [Part 3, Flow Matching](./diffpart3.html).*

> **Work in progress.** This part is still being written.

To unify the different parameterizations we write the forward process in the following way:

$$
\mathbf{x}_t = \alpha_t \mathbf{x}_0 + \sigma_t\epsilon
$$

This is essentially the flow matching view for the flow between the target distribution $$p_{\text{data}}(\mathbf{x}_0)$$ and the Gaussian distribution $$\epsilon \sim \mathcal{N}(0, \mathbf{I})$$.

---

[⬅ Part 3: Flow Matching](./diffpart3.html) · [Series index](./diffusion.html) · [Part 5: Solvers ➡](./diffpart5.html)
