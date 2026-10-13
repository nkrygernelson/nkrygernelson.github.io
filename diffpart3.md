---
layout: default
title: "Diffusion Crash Course, Part 3: Flow Matching"
---

# Part 3: Flow Matching

*Part 3 of [A Diffusion Crash Course](./diffusion.html). Previous: [Part 2, Score and SDEs](./diffpart2.html).*

Much like in DDPM and in the score view of diffusion, Flow Matching (FM) offers a way to evolve a source distribution $$p_{\text{source}}$$ to a target distribution $$p_{\text{target}}$$. However, unlike DDPM, the $$p_{\text{target}}$$ distribution is arbitrary instead of Gaussian. For the purpose of these notes, we first consider that the source distribution is $$\mathcal{N}(0, \mathbf{I})$$.[^convention]

[^convention]: In flow matching we consider that the target distribution is our $$p_{\text{data}}$$ distribution while our source is the Gaussian distribution. This convention is a bit confusing considering that we have previously stated that *we want to bring the data distribution to noise* in the context of DDPM and in the score-based diffusion contexts. Concretely, we take $$t=0$$ to correspond to the data (target) and $$t=1$$ to the noise (source); some references reverse this. The forward path therefore runs $$0 \to 1$$, and generation is the reverse flow $$1 \to 0$$.

## The flow, the pushforward, and the continuity equation

Consider a general velocity field:

$$
\frac{\mathrm{d}\mathbf{x}_t}{\mathrm{d}t} = \mathbf{v}_t(\mathbf{x}_t,t).
\tag{1}
$$

This ODE defines a flow $$\mathbf{x}(t)$$ which is a solution to the initial value problem defined by equation (1) with $$\mathbf{x}(0)=\mathbf{x}_0$$:

$$
\boldsymbol{\Psi}_{0\rightarrow t}(\mathbf{x}_0) = \mathbf{x}_0 + \int_0^t \mathbf{v}_s(\mathbf{x}_s,s)\,\mathrm{d}s.
\tag{2}
$$

The *pushforward density* is the result of transforming the density $$p_0(\mathbf{x}_0)$$ by applying the flow in equation (2) to $$\mathbf{x}_0$$:

$$
p_t^{\text{fwd}}(\mathbf{x}) = \int \delta\left(\mathbf{x} - \boldsymbol{\Psi}_{0 \to t}(\mathbf{x}_0)\right) p_0(\mathbf{x}_0)\, \mathrm{d}\mathbf{x}_0 =: \boldsymbol{\Psi}_{0 \to t} \# p_0,
$$

where, by our convention, $$p_0 = p_{\text{target}}$$ is the data distribution.

The law tying a density path to the velocity that generates it is the continuity equation. Given a prescribed density path $$p_t$$, the flow-induced density $$p_t^{\text{fwd}}$$ exactly matches $$p_t$$ for all $$t \in [0, 1]$$ if and only if the pair $$(p_t, \mathbf{v}_t)$$ satisfies the continuity equation:

$$
\partial_t p_t(\mathbf{x}) + \nabla_{\mathbf{x}} \cdot \big(p_t(\mathbf{x})\,\mathbf{v}_t(\mathbf{x})\big) = 0,
$$

for all $$t \in [0, 1]$$ and $$\mathbf{x}$$. We will use this twice below: once to recover the marginal velocity from conditional ones, and once to justify generation by the probability-flow ODE.

## Paths between distributions: diffusion flow-matching

In DDPM the conditional path from $$x_0$$ to Gaussian noise was defined as:

$$
p(x_i\mid x_0) = \mathcal{N}\!\left(x_i;\, \bar{\alpha}_i x_0,\, (1 - \bar{\alpha}_i^2) I\right).
$$

We can consider a similar path in continuous time:

$$
p_t(\mathbf{x}_t\mid\mathbf{x}_0) = \mathcal{N}(\alpha_t\mathbf{x}_0,\, \sigma_t^2\mathbf{I}),
$$

written alternatively:

$$
\mathbf{x}_t = \alpha_t\mathbf{x}_0 + \sigma_t\epsilon.
$$

From this conditional path we can define the conditional velocity field by differentiating the sample path:

$$
\begin{aligned}
\mathbf{v}_t(\mathbf{x}_t\mid\mathbf{x}_0) = \frac{d\mathbf{x}_{t}}{dt} &= \frac{d}{dt}\left[\alpha_t\mathbf{x}_0 + \sigma_t\epsilon \right]\\
& = \alpha_t'\mathbf{x}_0+\sigma_t'\epsilon.
\end{aligned}
$$

Fixing $$\mathbf{x}_0$$ defines a conditional velocity field $$\mathbf{v}_t(\mathbf{x}_t\mid\mathbf{x}_0)$$. The density $$p_t(\mathbf{x}_t)$$ can be obtained by marginalization:

$$
p_t(\mathbf{x}_t) = \int p_t(\mathbf{x}_t\mid\mathbf{x}_0)\,p_{\text{data}}(\mathbf{x}_0)\,d\mathbf{x}_0 = \mathbb{E}_{\mathbf{x}_0 \sim p_{\text{data}}}[p_t(\mathbf{x}_t\mid\mathbf{x}_0)].
$$

The marginal velocity, however, is a posterior average of the conditional velocities:

$$
\mathbf{v}_t(\mathbf{x}_t) = \int \mathbf{v}_t(\mathbf{x}_t\mid\mathbf{x}_0)\,\frac{p_t(\mathbf{x}_t\mid\mathbf{x}_0)\,p_{\text{data}}(\mathbf{x}_0)}{p_t(\mathbf{x}_t)}\,d\mathbf{x}_0 = \mathbb{E}_{\mathbf{x}_0 \sim p_t(\mathbf{x}_0\mid\mathbf{x}_t)}[\mathbf{v}_t(\mathbf{x}_t\mid\mathbf{x}_0)].
$$

This posterior weighting is exactly what makes the marginal velocity hard, as we discuss below.

## Constructing a path via a latent z

As evidenced by the pushforward above, if we have a defined $$\mathbf{v}$$, FM implicitly defines a continuous family of intermediate densities $$\{p_t\}_{t \in [0,1]}$$ interpolating between these endpoints. More generally than in the diffusion flow-matching example in the previous section, each marginal $$p_t$$ is expressed via a latent variable $$\mathbf{z}$$ drawn from a known distribution $$\pi(\mathbf{z})$$ and a conditional distribution $$p_t(\mathbf{x}_t \mid \mathbf{z})$$:

$$
p_t(\mathbf{x}_t) = \int p_t(\mathbf{x}_t \mid \mathbf{z})\, \pi(\mathbf{z})\, \mathrm{d}\mathbf{z},
$$

with $$\bigl(\pi(\mathbf{z}), \{p_t(\cdot \mid \mathbf{z})\}\bigr)$$ chosen to satisfy the boundary conditions $$p_0(\mathbf{x}) = p_{\text{target}}$$ and $$p_1(\mathbf{x}) = p_{\text{source}}$$. In general, the marginal densities $$p_t$$ are not tractable, since they require integrating over $$\pi(\mathbf{z})$$, and both $$\pi(\mathbf{z})$$ and the conditional distributions $$p_t(\mathbf{x}_t \mid \mathbf{z})$$ can be complex. Conditioning can be categorized into:

- **Two-sided conditioning:** $$\mathbf{z} = (\mathbf{x}_0, \mathbf{x}_1) \sim p_{\mathrm{src}}(\mathbf{x}_0)\, p_{\mathrm{tgt}}(\mathbf{x}_1)$$, where $$\pi$$ couples source and target distributions. This allows FM to define transport between arbitrary distributions.
- **One-sided conditioning:** $$\mathbf{z} = \mathbf{x}_0$$ or $$\mathbf{z} = \mathbf{x}_1$$. It especially recovers diffusion-like setups when the source distribution is chosen to be Gaussian.

In the case of DDPM and in score-based diffusion, once we have defined the forward process that transports $$p_{\text{data}}$$ to $$\mathcal{N}(0, \mathbf{I})$$, we developed ways to train a model to reverse it. The same is true here. Let's say that we are able to construct a velocity field that induces the desired transformation; to reverse this transformation, it suffices to learn the velocity field $$\mathbf{v}_{\phi}(\mathbf{x}, t)$$. If this can be done, generation is simply a matter of sampling $$\mathbf{x}_1$$ from the $$p_{\text{source}}$$ distribution and integrating backwards in time to $$t=0$$ to produce $$\mathbf{x}_0^*$$.

## The marginal velocity is intractable

Given that we only need to learn $$\mathbf{v}_t(\mathbf{x}_t,t)$$, the MSE error between the true and approximate velocity fields seems appropriate:

$$
\mathcal{L}_{\text{FM}}(\phi) = \mathbb{E}_{t,\mathbf{x}_t}\left[\|\mathbf{v}_\phi(\mathbf{x}_t,t) - \mathbf{v}_t(\mathbf{x}_t)\|^2_2\right].
$$

But $$\mathbf{v}_t(\mathbf{x}_t)$$ requires averaging over all latents that could produce $$\mathbf{x}_t$$, precisely the posterior $$p_t(\mathbf{z}\mid\mathbf{x}_t)$$ derived above, whose normalizer is the intractable marginal $$p_t(\mathbf{x}_t)$$.

## Conditioning makes it tractable

The conditional velocity $$\mathbf{v}_t(\mathbf{x}_t\mid\mathbf{z})$$ is closed-form, and recovers the marginal via the continuity-equation argument:

$$
\mathbf{v}_t(\mathbf{x}_t) = \mathbb{E}_{\mathbf{z}\sim p(\cdot\mid\mathbf{x}_t)}\big[\mathbf{v}_t(\mathbf{x}_t\mid\mathbf{z})\big].
$$

This motivates the conditional flow-matching objective, which replaces the intractable target by the closed-form conditional one:

$$
\mathcal{L}_{\text{CFM}}(\phi) = \mathbb{E}_{t,\,\mathbf{z},\,\mathbf{x}_t\sim p_t(\cdot\mid\mathbf{z})}\left[\|\mathbf{v}_\phi(\mathbf{x}_t,t) - \mathbf{v}_t(\mathbf{x}_t\mid\mathbf{z})\|^2_2\right].
$$

Because regressing onto a target and onto its conditional expectation share the same minimizer, the two objectives differ only by a constant and share the same optimum:

$$
\mathcal{L}_{\text{FM}}(\phi) = \mathcal{L}_{\text{CFM}}(\phi) + C, \qquad \mathbf{v}^*(\mathbf{x}_t,t) = \mathbf{v}_t(\mathbf{x}_t).
$$

## Underlying rules

We now have a way to transport source to target via the probability-flow ODE:

$$
\frac{\mathrm{d}\mathbf{x}_t}{\mathrm{d}t} = \mathbf{v}_t(\mathbf{x}_t,t).
$$

By the continuity equation introduced at the outset, integrating this ODE transports the density along the prescribed path $$p_t$$, so generating a sample reduces to drawing $$\mathbf{x}_1 \sim p_{\text{source}}$$ and integrating the learned field $$\mathbf{v}_\phi$$ backwards to $$t=0$$. This is the same essence as the other views of diffusion.

---

[⬅ Part 2: Score and SDEs](./diffpart2.html) · [Series index](./diffusion.html) · [Part 4: Unification ➡](./diffpart4.html)
