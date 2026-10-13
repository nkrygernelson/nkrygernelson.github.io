---
layout: default
title: "Diffusion Crash Course, Part 2: Score and SDEs"
---

# Part 2: Score-Based Diffusion and SDEs

*Part 2 of [A Diffusion Crash Course](./diffusion.html). Previous: [Part 1, DDPM](./diffpart1.html).*

The big switch now is to move into the continuous-time picture. So far we have a transformation from $$p_{\mathrm{data}}$$ to Gaussian noise by applying a transition kernel or, equivalently, by transforming the clean sample according to a schedule and injecting Gaussian noise. Huge insight and flexibility are gained by treating the forward process as a continuous stochastic process defined by an SDE.

## ODEs

In the case of an ODE, the evolution of a system is described by

$$
\frac{dx(t)}{dt} = f\big(x(t),t\big), \qquad x(0)=x_0,
$$

with solution obtained by integrating along the vector field,

$$
x(t) = x_0 + \int_{0}^{t} f\big(x(s),s\big)\, ds,
$$

and we recover the initial point by integrating backward,

$$
x_0 = x(t) + \int_{t}^{0} f\big(x(s),s\big)\, ds
    = x(t) - \int_{0}^{t} f\big(x(s),s\big)\, ds.
$$

An ODE with sufficient smoothness admits a unique solution, a flow in the vector field defined by the ODE, so we can integrate along the field to move freely between $$x(t)$$ and $$x_0$$.

## SDEs

Keeping things simple, work in 1D:

$$
dx = \underbrace{f(x_t,t)}_{\text{deterministic drift}}dt + \overbrace{g(t)\,dW}^{\text{Brownian motion}},
$$

where $$dW \approx \sqrt{dt}\,\epsilon$$, $$\epsilon \sim \mathcal{N}(0, I)$$. This defines a single trajectory $$x(t)$$. In contrast with ODEs, $$x(t)$$ cannot be known definitively.

![Single simulation of an SDE](/imgs/SDE_onepath.jpeg)

*Figure: Single simulation of an SDE.*

![Many simulations of an SDE](/imgs/many_sims.jpeg)

*Figure: Many simulations of an SDE.*

## The Fokker-Planck equation

Given an SDE $$dx = f(x_t,t)\,dt + g(t)\,dW$$, what can we say about the evolution of the marginal densities $$p_t(x_t)$$? The initial probability distribution evolves instantaneously due to two mechanisms: a drift term $$f(x_t, t)$$, and an injection of scaled Gaussian noise $$g(t)\,dW_t=g(t)\sqrt{dt}\,\epsilon$$. The following (heuristic) derivation of the Fokker-Planck equation in 1D hinges on treating the two terms in the SDE as independent for an infinitesimal $$dt$$.

### Evolution due to drift

With $$dx_t = f(x_t,t)\,dt$$ and $$x_0 \sim p(x_0)$$, the density obeys the continuity equation

$$
\frac{\partial p_t(x_t)}{\partial t} = -\frac{\partial}{\partial x}\!\left[f(x,t)\,p_t(x_t)\right].
$$

### Evolution due to noise injection

We consider $$dx_t = g(t)\,dW_t$$. To derive the $$\frac{\partial p_t(x_t)}{\partial t}$$ we proceed by discretizing time. At time $$t$$ we have $$X_t\sim p_t(x_t)$$ and we add noise $$Z \sim P_Z =\mathcal{N}(Z\vert 0, g^2(t)\Delta t)$$. The new state $$x_{t+\Delta t} = x_t + Z$$ is a sum of two independent random variables, so

$$
p_{t+\Delta t}(x,t) = \int_{-\infty}^{\infty} p_t(x-z)\, P_{Z}(z)\, dz.
$$

Taylor-expanding $$p(x-z,t)$$ around $$x$$,

$$
\begin{aligned}
p_t(x-z) &= p_t(x) - ((x-z)-x)\frac{\partial p_t}{\partial x} + \frac{1}{2}((x-z)-x)^{2}\frac{\partial^2 p_t}{\partial x^2} + O(z^3)\\
&=p_t(x) - z\frac{\partial p_t}{\partial x} + \frac{1}{2}z^{2}\frac{\partial^2 p_t}{\partial x^2} + O(z^3)
\end{aligned}
$$

We plug the Taylor expansion into the integral:

$$
\begin{aligned}
p_{t+\Delta t}(x) &\approx\int_{-\infty}^{\infty}\left[p_t(x) - z\frac{\partial p_t}{\partial x} + \frac{1}{2}z^{2}\frac{\partial^2 p_t}{\partial x^2}\right]  P_{Z}(z)\, dz.\\
&= \int_{-\infty}^{\infty}p_t(x)P_{Z}(z)dz - \int_{-\infty}^{\infty}z\frac{\partial p_t}{\partial x}P_Z(z)\,dz + \int_{-\infty}^{\infty}\frac{1}{2}z^{2}\frac{\partial^2 p_t}{\partial x^2} P_{Z}(z)\, dz.\\
&=p_t(x)\int_{-\infty}^{\infty}P_{Z}(z)dz - \frac{\partial p_t}{\partial x}\int_{-\infty}^{\infty}zP_Z(z)\,dz + \frac{1}{2}\frac{\partial^2 p_t}{\partial x^2}\int_{-\infty}^{\infty} z^{2}P_{Z}(z)\, dz
\end{aligned}
$$

The right-hand side is now $$p_t(x)$$ and its derivatives as factors in front of moments of the Gaussian random variable $$Z$$. Using the moments $$\int p_Z(z)\,dz = 1$$, $$\mathbb{E}[Z]=0$$, $$\mathbb{E}[Z^2]=g^2(t)\Delta t$$ (because $$\mathbb{E}[Z^2]=\mathrm{Var}[Z]$$ since $$\mathbb{E}[Z]=0$$), we get:

$$
\begin{aligned}
p_{t+\Delta t}(x) &\approx p_t(x)\cdot 1 - \frac{\partial p_t}{\partial x}\mathbb{E}[z]+ \frac{1}{2}\frac{\partial^2 p_t}{\partial x^2}\mathbb{E}[z^2]\\
p_{t+\Delta t}(x) &= p(x,t) + \frac{1}{2}g^{2}(t)\Delta t\,\frac{\partial^2 p}{\partial x^2} + O(\Delta t^4)
\end{aligned}
$$

The residual term is $$O(\Delta t^4)$$ because the 3rd moment $$\mathbb{E}[Z^3]$$ is $$0$$, since symmetric distributions have $$0$$ odd moments. The fourth moment will contribute a $$\Delta t^4$$ term.

Rearranging and taking $$\Delta t \to 0$$ yields the heat equation

$$
\frac{\partial p_t(x_t)}{\partial t} = \frac{1}{2}g^{2}(t)\frac{\partial^{2}}{\partial x^2}p_t(x_t).
$$

### Combining both mechanisms

Combining both mechanisms gives the Fokker-Planck equation

$$
\frac{\partial p_t(x_t)}{\partial t} = -\frac{\partial}{\partial x}\!\bigl[f(x,t)\,p_t(x_t)\bigr]
+ \frac{1}{2}g^{2}(t)\frac{\partial^{2}}{\partial x^2}p_t(x_t).
$$

So the SDE $$dx = f(x_t,t)\,dt + g(t)\,dW$$ has deterministically evolving marginals $$p_t(x_t)$$.

## The Langevin SDE

Consider the SDE with drift equal to the *score* $$\nabla_x \log p_t(x)$$:

$$
dx_t = \frac{1}{2}\frac{\partial}{\partial x}\log p_t(x)\,dt + dW.
$$

Applying Fokker-Planck,

$$
\frac{\partial p_t(x)}{\partial t}
= -\frac{\partial}{\partial x}\!\left[\frac{1}{2}\frac{\partial \log p_t}{\partial x}\, p_t\right]
+ \frac{1}{2}\frac{\partial^2 p_t}{\partial x^2}
= -\frac{1}{2}\frac{\partial^2 p_t}{\partial x^2} + \frac{1}{2}\frac{\partial^2 p_t}{\partial x^2} = 0.
$$

The score drift exactly cancels the diffusion, so $$p_t(x) = p_0(x)$$ for all $$t$$. The discrete update makes this concrete:

$$
X_{t+\Delta t} = X_t + \frac{1}{2}\underbrace{\frac{\partial}{\partial x}\log p_t(x_t)}_{\text{score: pushes } X_t \text{ toward } p_0}\Delta t
+ \underbrace{\sqrt{\Delta t}\,\epsilon}_{\text{destroys } p_0},\quad \epsilon\sim\mathcal{N}(0,1).
$$

The score perfectly compensates for the noise injection.

For intuition, take a 2D Gaussian $$p(x) = \mathcal{N}(\mu, \Sigma)$$. The score is a vector field

$$
s(x) = \nabla_{x}\log p(x) = \Sigma^{-1}(\mu-x).
$$

![Score vector field of a 2D Gaussian](/imgs/scorefield.png)

![Langevin dynamics following the score](/imgs/langevinscore.png)

![Approximating a Gaussian with Langevin dynamics](/imgs/approxgausslang.png)

## Reversing an SDE

Given the SDE $$dx_t = f(x_t,t)\,dt + g(t)\,dW$$ with $$x_0 \sim p_0$$, and a sample $$x_T \sim p_T$$, can we recover $$x_0$$ by reversing time?

- For an ODE: yes, just flip the sign of $$t$$.
- For an SDE: the noise injection *destroys* information.

But Langevin showed us that the score can compensate for noise injection.

### Probability flow ODE

Consider the ODE with modified drift

$$
dx_t = h(x_t,t)\,dt, \qquad h(x,t) := f(x,t) - \frac{g^2(t)}{2}\frac{\partial}{\partial x}\log q_t(x).
$$

Applying the continuity equation (Fokker-Planck with $$g=0$$),

$$
\frac{\partial q_t}{\partial t} = -\frac{\partial}{\partial x}\!\left[h\, q_t\right]
= -\frac{\partial}{\partial x}\!\left[f\, q_t - \frac{g^2}{2}\frac{\partial q_t}{\partial x}\right]
= -\frac{\partial}{\partial x}[f\, q_t] + \frac{g^2}{2}\frac{\partial^2 q_t}{\partial x^2},
$$

which is exactly the Fokker-Planck equation for the original SDE, so $$q_t(x) = p_t(x)$$ for all $$t$$. The ODE shares the same marginals as the SDE.

### Time-reversed Langevin

Under time reversal (flip the sign of $$t$$), the score term flips sign. It now *pushes away* from $$p_0$$, giving the backward Langevin

$$
dx_t = -\frac{1}{2}\frac{\partial}{\partial x}\log p_t(x)\,dt + dW.
$$

### From PF-ODE to reverse SDE

The PF-ODE drift is

$$
h(x,t) = f(x,t) - \frac{g^2(t)}{2}\frac{\partial}{\partial x}\log p_t(x).
$$

Adding the forward Langevin SDE recovers the original SDE,

$$
dx_t = \left[h(x,t) + \frac{g^2(t)}{2}\frac{\partial}{\partial x}\log p_t(x)\right]dt + g(t)\,dW
      = f(x,t)\,dt + g(t)\,dW,
$$

while adding the *backward* Langevin instead gives the reverse SDE

$$
dx_t = \left[h(x,t) - \frac{g^2(t)}{2}\frac{\partial}{\partial x}\log p_t(x)\right]dt + g(t)\,d\bar{W}
      = \left[f(x,t) - g^2(t)\frac{\partial}{\partial x}\log p_t(x)\right]dt + g(t)\,d\bar{W}.
$$

## Score matching

We have shown that all we need to perform the forward and generative processes is the score

$$
\nabla_{x}\log p_t(x) = s(x,t),
$$

approximated by $$s_{\theta}(x,t)$$. Turning this into a learning objective gives score matching,

$$
\mathcal{L}_{SM}(\theta) = \mathbb{E}_{t}\,\mathbb{E}_{x_t\sim p_t}\!\left[\|s_\theta(x_t,t)-\nabla_{x}\log p_t(x_t)\|^2_2\right].
$$

But what is the true score? As before, once we specify $$x_0$$, the PF-ODE deterministically specifies the path the model must learn in order to denoise the data, and the objective becomes tractable as denoising score matching,

$$
\mathcal{L}_{DSM}(\theta) = \mathbb{E}_{t}\,\mathbb{E}_{x_0}\,\mathbb{E}_{p(x_t\mid x_0)}\!\left[\|s_\theta(x_t,t)-\nabla_{x}\log p_t(x_t\mid x_0)\|^2_2\right].
$$

Notice the same move as in Part 1: conditioning on a clean sample $$x_0$$ turns an intractable target into a closed-form one.

---

[⬅ Part 1: DDPM](./diffpart1.html) · [Series index](./diffusion.html) · [Part 3: Flow Matching ➡](./diffpart3.html)
