---
layout: default
title: "Diffusion Crash Course, Part 1: DDPM"
---

# Part 1: DDPM

*Part 1 of [A Diffusion Crash Course](./diffusion.html).*

This is a crash course in diffusion: the fundamental math and concepts behind diffusion models. The goal is to present the fundamental mathematical framework of modern diffusion models. Topics covered: DDPM, SDEs and Score-Based diffusion, Fokker-Planck equation and Flow Matching.

## DDPM

### Motivation

So what problem do diffusion models actually solve? Unless you're an ML researcher, your main point of contact with diffusion is probably AI video or image generation, where you write a prompt describing an image/video and the model produces it. But can we state the problem in a way that makes the design of the model more intuitive? A lot of progress comes from stating, in probabilistic terms, what diffusion models really accomplish.

What diffusion models do is learn the underlying distribution $$p_{\text{data}}$$ of a dataset of observations. Take the classic MNIST handwritten-digit dataset as an example. A diffusion model trained only on these images learns the probability distribution underlying them, and is then able to generate new samples from that distribution. That is the problem diffusion models solve.

![MNIST digit distribution](/imgs/dist_mnist.jpeg)

![Image generation](/imgs/generation.jpeg)

It takes a little work to reconcile this statement with what you normally picture as image generation. The simplest way to think about it is to consider a fixed-size image, that is a fixed number of pixels, described by a very complicated joint probability distribution over all pixel values. The pixel values that correspond to a valid digit have a probability significantly higher than, say, pure noise, under this massive joint distribution. It's abstract to imagine a probability distribution describing every $$28\times 28$$ image of a handwritten digit (or every $$32\times 32$$ photo of a cat), but that is exactly the case.

Stated another way: we want to learn the probability distribution underlying the data in a way that lets us generate new samples from it. We learn the "rules" of handwritten-digit image, in other words what the joint distribution over all pixel values is, so that we can generate a sample with high joint probability, meaning a realistic, plausible digit. Two things matter: learn the distribution, and learn it so that we can sample from it.

To bring this down to a tractable number of dimensions, consider a 2D Gaussian-mixture "blob" distribution with four blobs. We want to generate new points that would be realistic under it: points that land in one of the four blobs.

![2D Gaussian blobs](/imgs/GaussBlob.png)

*Figure: 2D Gaussian blobs.*

### The forward process

A technical introduction to diffusion usually leads with DDPM (Denoising Diffusion Probabilistic Models), because it's historically important and conceptually lighter than what we'll build up to later. DDPM has two key components, the two stages most people list if you ask how diffusion works:

1. **Forward process.** A Markov chain of fixed forward transition kernels $$p(x_i\mid x_{i-1})$$. The data evolves into an isotropic Gaussian distribution. This process is fixed, not learned.
2. **Reverse / generative process (learnable).** A neural network learns to reverse the noise corruption through a parameterized distribution $$p_\theta(x_{i-1}\mid x_i)$$. Starting from pure noise, the model iteratively denoises it to generate realistic samples.

![DDPM forward and backward process](/imgs/DDMP.png)

*Figure: DDPM forwards and backwards process visualized. Diagram taken from "The Principles of Diffusion Models."*

The first is the forward process. We take a data sample (say, an MNIST-style digit) and over a number of steps we add Gaussian noise, so the image gets grainier at every iteration, until all information is destroyed and we're left with pure noise. This is a stochastic process: a Markov chain, where reaching the next step requires only the previous step. We do this by applying fixed forward transition kernels, and the goal is to carry the original data distribution to pure Gaussian noise:

$$
p(x_i \mid x_{i-1}) = \mathcal{N}\!\left(x_i;\, \sqrt{1-\beta_i^2}\, x_{i-1},\; \beta_i^2 I\right).
\tag{1}
$$

You don't need to memorize every formula, but it helps with the reading: the fixed forward transition kernel in equation (1) is written in terms of $$\beta_i$$ values that follow a fixed schedule $$\{\beta_i\}_{i=1}^{L}$$. At each step we sample from the kernel, conditioned on the previous realization.

I find another view more intuitive: to get $$x_i$$, the next latent, we scale $$x_{i-1}$$ and add noise, where both the scaling and the amount of noise are set by the schedule. In this reparameterized (sampling) form,

$$
x_i = \sqrt{1-\beta_i^2}\, x_{i-1} + \beta_i\, \epsilon,
\qquad \epsilon \sim \mathcal{N}(0, I),
\tag{2}
$$

and writing $$\alpha_i := \sqrt{1-\beta_i^2}$$ this is just $$x_i = \alpha_i x_{i-1} + \beta_i \epsilon$$.

Applying these kernels one step at a time all the way to noise is inconvenient. It's much nicer to have a *simulation-free* way to sample, meaning we don't have to simulate the successive application of kernels. We can define a kernel that, given the index $$i$$ we care about, lets us sample $$x_i$$ directly. Unrolling the recursion equation (2) one step,

$$
\begin{aligned}
x_i
&= \sqrt{1-\beta_i^2}\, x_{i-1} + \beta_i\, \epsilon \\
&= \sqrt{1-\beta_i^2}\left(\sqrt{1-\beta_{i-1}^2}\, x_{i-2} + \beta_{i-1}\, \epsilon'\right) + \beta_i\, \epsilon \\
&= \sqrt{1-\beta_i^2}\,\sqrt{1-\beta_{i-1}^2}\; x_{i-2}
   \;+\; \sqrt{1-\beta_i^2}\,\beta_{i-1}\, \epsilon'
   \;+\; \beta_i\, \epsilon .
\end{aligned}
$$

The two independent Gaussian terms merge into a single Gaussian with variance $$(1-\beta_i^2)\beta_{i-1}^2 + \beta_i^2$$. Iterating all the way down to $$x_0$$ and defining

$$
\alpha_k := \sqrt{1-\beta_k^2},
\qquad
\bar{\alpha}_i := \prod_{k=1}^{i} \sqrt{1-\beta_k^2} = \prod_{k=1}^{i} \alpha_k,
$$

the marginal collapses to the simulation-free form

$$
\boxed{\,x_i = \bar{\alpha}_i\, x_0 + \sqrt{1-\bar{\alpha}_i^2}\;\epsilon\,},
\qquad \epsilon \sim \mathcal{N}(0, I),
$$

or equivalently

$$
p(x_i\mid x_0) = \mathcal{N}\!\left(x_i;\, \bar{\alpha}_i\, x_0,\, (1-\bar{\alpha}_i^2) I\right).
\tag{3}
$$

Equivalently, instead of sampling from that Gaussian, we can think of it as scaling $$x_0$$ by an amount determined by $$\bar{\alpha}_i$$ and adding a corresponding amount of noise. I find this version more intuitive: for a fixed $$x_0$$, as $$i$$ grows we scale $$x_0$$ down and add more noise, until at the end there's essentially no $$x_0$$ information left and only noise remains.

For a suitable schedule, as the number of steps grows the final latent is described by a pure Gaussian distribution:

$$
p(x_L \mid x_0) \to \mathcal{N}(0, I)
\qquad \text{as } L \to \infty \quad(\text{since } \bar{\alpha}_i \to 0).
$$

So we have effectively brought the data to noise, which is the whole point.

Returning to the blobs (using $$t$$ to track time instead of $$i$$): we start with the blob distribution, evolve it under some schedule, and in the end we have something that looks like total noise: a 2D normal distribution.

![DDPM forward process](/imgs/forward.png)

*Figure: DDPM forward process.*

### The reverse process

We have two stages, forward and generative. The question is how to learn the reverse process so we can iteratively denoise starting from the prior $$x_L \sim \mathcal{N}(0, I)$$. What's a good learning objective? A natural choice is the KL divergence, which measures how different two distributions are. We want the learned reverse conditional distribution to match the true one, so a good measure of how well the model is doing is how close these two distributions are. We sample from the marginal $$p_i(x_i)$$, take an expectation, and we have a learning objective:

$$
\mathcal{L}=\mathbb{E}_{p_i(x_i)}\!\left[\, D_{\mathrm{KL}}\!\big(p(x_{i-1}\mid x_i)\,\|\,p_\theta(x_{i-1}\mid x_i)\big) \right].
$$

It seems fine, but it isn't, because we don't actually know the true reverse transition kernel. Applying Bayes' theorem, we can rewrite the reverse kernel in terms of the forward kernel, which we do know:

$$
p(x_{i-1}\mid x_i) = p(x_i\mid x_{i-1})\,\frac{p_{i-1}(x_{i-1})}{p_i(x_i)}.
$$

But the denominator contains a classic intractable term: an integral over all clean samples, the entire $$p_{\text{data}}$$ distribution,

$$
p_i(x_i) = \underbrace{\int p(x_i\mid x_0)\, p_{\text{data}}(x_0)\, dx_0}_{\text{hard to approximate}},
$$

and similarly for $$p_{i-1}(x_{i-1})$$. That makes it very hard to work with.

### Conditioning on the clean sample

Here's the trick, and you'll see it again in this series: we condition the reverse transition kernel on a clean sample $$x_0$$. The moment we do this, things become tractable. Rewriting with Bayes' theorem, the term that was an unconditional distribution becomes a conditional one:

$$
p(x_{i-1}\mid x_i,x_0)
= p(x_i\mid x_{i-1},x_0)\frac{p(x_{i-1}\mid x_0)}{p(x_i\mid x_0)}
= p(x_i\mid x_{i-1})\frac{p(x_{i-1}\mid x_0)}{p(x_i\mid x_0)},
$$

and we know $$x_i$$ given $$x_0$$ because we have the forward process. The same holds for the denominator. Concretely:

- $$p(x_i\mid x_{i-1},x_0) = p(x_i\mid x_{i-1})$$ by the Markov property.
- $$p(x_{i-1}\mid x_0)$$ and $$p(x_i\mid x_0)$$ are both Gaussian, from equation (3).

Conditioning gives us a tractable reverse transition kernel.

But can we get away with conditioning on $$x_0$$? Yes, and here's why. We can write a new conditional learning objective: the KL divergence between the reverse transition kernel conditioned on a clean $$x_0$$ (the target the model learns toward) and what the model has learned. Now that $$x_0$$ appears in the objective, we take the expectation over the samples available in our dataset (samples from $$p_{\text{data}}$$) and from each clean sample $$x_0$$, for a given $$i$$, we produce the needed samples and evaluate the term. The remarkable part is that this conditional objective equals the original unconditional objective up to a constant $$C$$:

$$
\mathbb{E}_{p_i(x_i)}\!\left[D_{\mathrm{KL}}\!\big(p(x_{i-1}\mid x_i)\,\|\,p_\theta(x_{i-1}\mid x_i)\big)\right]
=
\mathbb{E}_{p_{\text{data}}(x_0)}\,\mathbb{E}_{p(x_i\mid x_0)}\!\left[D_{\mathrm{KL}}\!\big(p(x_{i-1}\mid x_i, x_0)\,\|\,p_\theta(x_{i-1}\mid x_i)\big)\right] + C.
$$

At the gradient level, learning one is the same as learning the other. In particular, the minimizer satisfies

$$
p^\ast(x_{i-1}\mid x_i)
= \mathbb{E}_{p(x_0\mid x_i)}\!\left[p(x_{i-1}\mid x_i, x_0)\right]
= p(x_{i-1}\mid x_i).
\tag{4}
$$

### Learning along a path

Is something deep happening? When we condition on $$x_0$$, the mean and variance functions become defined. From the conditional marginal

$$
p(x_i\mid x_0) = \mathcal{N}\!\left(x_i;\, \bar{\alpha}_i x_0,\, (1 - \bar{\alpha}_i^2) I\right),
$$

once $$x_0$$ is fixed, both

$$
\mu_{x_i\mid x_0}(i) = \bar{\alpha}_i\,x_0,
\qquad
\mathrm{Var}_{x_i\mid x_0}(i) = (1-\bar{\alpha}_i^2) I
$$

are pinned down. So conditioning defines a learnable path for the model; once that path is defined, we can learn along it in order to go the other way.

![Learnable path for conditional diffusion](/imgs/path.jpeg)

Fixing $$x_0$$ fixes the evolution of the conditional density, and it also defines the reverse transition kernel. I won't derive it, but the key fact is that the reverse transition kernel is also Gaussian, with its own mean and variance functions, defined thanks to fixing $$x_0$$:

$$
p(x_{i-1}\mid x_i, x_0) = \mathcal{N}\!\left(x_{i-1};\, \mu(x_i, x_0, i),\, \mathrm{Var}(i)\cdot I\right),
$$

where

$$
\mu(x_i, x_0, i) := \frac{\bar{\alpha}_{i-1}\beta_i^2}{1 - \bar{\alpha}_i^2}\,x_0
+ \frac{(1 - \bar{\alpha}_{i-1}^2)\alpha_i}{1 - \bar{\alpha}_i^2}\,x_i,
\qquad
\mathrm{Var}(i) := \frac{1 - \bar{\alpha}_{i-1}^2}{1 - \bar{\alpha}_i^2}\beta_i^2.
$$

### The simplified objective

Because the transition kernels are Gaussian, the learning objective simplifies. The KL-divergence objective reduces to a (variance-weighted) mean-squared-error regression on the mean function, summed across the steps:

$$
\mathcal{L}_{\text{DDPM}}(\phi) := \sum_{i=1}^{L} \frac{1}{2\cdot\mathrm{Var}(i)}\,
\mathbb{E}_{x_0}\, \mathbb{E}_{p(x_i\mid x_0)}
\!\left[ \left\| \mu_\phi(x_i, i) - \mu(x_i, x_0, i) \right\|_2^2 \right].
$$

We can rewrite this in two different but equivalent ways, in which the model predicts different things: a loss where the model predicts the noise at each step, and one where it predicts the clean target at each step. Both come from rewriting the same expression: two parametrizations of the same problem. When generating, we can either predict the noise at each step or predict the clean sample at each step, and the two are equivalent.

Finally, sampling: start from Gaussian noise and run from $$L$$ down to $$t = 0$$. The points we generate look a lot like the correct blob distribution.

![DDPM generative process](/imgs/genproc.png)

*Figure: DDPM generative process.*

![Before and after generation](/imgs/beforeaftergen.png)

### Recap

That was a lot, so here's where we are. We defined the problem of diffusion: learn the underlying probability distribution of a dataset so that we can generate new samples. We do this in two stages, and along the way we established a handful of ideas that carry over to every other view of diffusion:

- **A fixed forward path** through the latents $$x_i$$. It is *simulation-free*: any latent can be written directly in terms of $$x_0$$, a noise $$\epsilon$$, and the schedule, $$x_i = \bar{\alpha}_i x_0 + \sqrt{1-\bar{\alpha}_i^2}\,\epsilon$$, without applying the kernel step by step.
- **Conditioning on $$x_0$$**, which makes the target (and hence the problem) tractable. This defines everything and gives us something we can actually optimize.
- **A conditional training objective that is gradient-equivalent** to the unconditional one (equal up to a constant $$C$$).
- The minimizer of the original loss satisfies equation (4), i.e. $$p^\ast(x_{i-1}\mid x_i)$$ is the average over all $$x_0$$ of the conditional reverse transition kernel; the minimizers of the conditional and unconditional objectives are the same.

---

[Series index](./diffusion.html) · [Part 2: Score and SDEs ➡](./diffpart2.html)
