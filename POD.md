---
layout: default
title: "POD — a modular diffusion research codebase"
---

# POD

**[github.com/nkrygernelson/POD](https://github.com/nkrygernelson/POD)** · Python, PyTorch, Hydra

POD is a research codebase for training and sampling diffusion models on 2D toy
distributions. The point of it is *modularity*: the noise schedule, the
prediction target, the time coordinate, the sampler, the training procedure and
the dataset are all independent, swappable components, wired together at runtime
by [Hydra](https://hydra.cc/) config groups.

That design exists for a reason. Most diffusion implementations hard-code one
choice of parameterization and schedule, which makes the question *"does this
change actually matter?"* hard to ask. Here, comparing a noise-prediction model
against a score-prediction model is a command-line override, not a rewrite:

```bash
python main.py parameterization=score_pred solver=ddim time_param=logsnr
```

## Architecture

Everything is composed from the config, then handed to a single orchestrating
class, `DiffusionProcess`, which owns the training loss and all prediction
conversions.

<pre class="mermaid">
flowchart TB
  CFG["configs/base.yaml<br/>Hydra defaults list"]

  CFG --> SCH["schedule<br/>alpha(t), sigma(t), derivatives"]
  CFG --> TP["time_param<br/>bijection t ↔ u"]
  CFG --> TD["t_distribution<br/>samples (t, u)"]
  CFG --> PM["parameterization<br/>what the net predicts"]
  CFG --> W["weight_func(t)"]
  CFG --> NET["model<br/>MLP or residual net"]
  CFG --> DS["dataset<br/>tree or blobs"]

  SCH --> TP
  SCH --> PM
  TP --> TD

  SCH --> DP["DiffusionProcess"]
  TP --> DP
  TD --> DP
  PM --> DP
  W --> DP
  NET --> DP

  DS --> TRP["train_proc<br/>standard / cfg / reflow"]
  DP --> TRP
  TRP --> FIT["fit_one training loop"]
  FIT --> CKPT["checkpoint:<br/>weights + model/param/schedule/time cfg"]
  CKPT --> GEN["generation.py<br/>solver + prior → samples"]
</pre>

A detail I like: the checkpoint stores the *config* of the model,
parameterization, schedule and time coordinate alongside the weights. At
generation time those are re-instantiated from the checkpoint rather than from
the live config, so you cannot accidentally load a `v_pred` network and sample
it as if it were `score_pred`. If the live config disagrees with the checkpoint,
it warns.

## The central abstraction: parameterizations

This is the part of the codebase I'd point at first. A diffusion network can be
trained to predict any of several quantities — the noise, the clean sample, the
score, or the velocity — and every sampler needs some specific one of them.
Writing four independent classes means four places to update every time you add
a feature.

Instead there's one base class and one inversion. Observe that the noised sample
and the network's target are both linear in $$(x_0, \epsilon)$$:

$$
\begin{bmatrix} x_t \\ \texttt{model\_out} \end{bmatrix}
=
\begin{bmatrix} \alpha_t & \sigma_t \\ a & b \end{bmatrix}
\begin{bmatrix} x_0 \\ \epsilon \end{bmatrix}
$$

Each parameterization is then nothing but a choice of the coefficients
$$(a, b)$$:

| Parameterization | $$a$$ | $$b$$ | class |
|---|---|---|---|
| noise | $$0$$ | $$1$$ | `NoisePred` |
| clean sample $$x_0$$ | $$1$$ | $$0$$ | `CleanPred` |
| score | $$0$$ | $$-1/\sigma_t$$ | `ScorePred` |
| velocity $$v$$ | $$\alpha_t'$$ | $$\sigma_t'$$ | `VPred` |

Recovering $$(x_0, \epsilon)$$ from whatever the network emitted is just
inverting that $$2\times 2$$ matrix:

$$
\begin{bmatrix} x_0 \\ \epsilon \end{bmatrix}
=
\frac{1}{\det}
\begin{bmatrix} b & -\sigma_t \\ -a & \alpha_t \end{bmatrix}
\begin{bmatrix} x_t \\ \texttt{model\_out} \end{bmatrix},
\qquad \det = \alpha_t b - \sigma_t a
$$

Once you have $$(x_0, \epsilon)$$, every other quantity any sampler could want
follows from one place:

<pre class="mermaid">
flowchart LR
  MO["model_out"] --> INV
  XT["x_t, t"] --> INV["to_x0_eps<br/>one 2x2 inversion<br/>det = alpha*b - sigma*a"]

  INV --> X0["x_0"]
  INV --> EPS["epsilon"]

  EPS --> SC["to_score<br/>-eps / sigma"]
  X0 --> VV["to_v<br/>alpha' x_0 + sigma' eps"]
  EPS --> VV

  VV --> PF["pfode<br/>probability-flow ODE drift"]
  VV --> RS["rev_sde<br/>v - 0.5 gamma^2 score"]
  SC --> RS

  PF --> S1["Euler"]
  PF --> S2["DEIS"]
  RS --> S3["Euler-Maruyama"]
  X0 --> S4["DDIM"]
  EPS --> S4
</pre>

The honest caveat is written into the code: the inversion divides by
$$\det$$, so it degrades wherever $$\det \to 0$$. Rather than hide that, the
repo ships a `singularity` report that plots $$1/|\det|$$ and the target
magnitude $$\max(|a|,|b|)$$ over the time grid for each parameterization, so you
can *see* where a given choice blows up before blaming the model. It depends
only on the schedule and the parameterization — not the weights — so it's a
static sanity check.

## Two clocks: schedule time and network time

The other design decision worth explaining is that POD distinguishes the time
the *schedule* speaks from the time the *network* sees. A `TimeParam` defines a
bijection $$u = f(t)$$; the schedule and parameterization are always evaluated
at $$t$$, while the network and the sampling grid live in $$u$$.

The default is log-SNR time, $$u = 2(\log \alpha_t - \log \sigma_t)$$, inverted
by $$t = \sigma(-u/2)$$. This matters because uniform spacing in $$t$$ is not
uniform in difficulty — log-SNR spacing concentrates steps where the signal-to-noise
ratio is actually changing. Because `u_bounds` is the single source of truth for
the usable range, the training time distribution and the sampler's grid
automatically cover the same support, buffered inward by `eps` to stay off the
singular endpoints.

## Training

<pre class="mermaid">
flowchart TB
  TDS["t_distribution.sample()"] --> T["t — schedule coordinate"]
  TDS --> U["u — network coordinate"]

  X0["x_0 from DataLoader"] --> FS
  NZ["noise: fresh N(0,I)<br/>or a curated partner x_1"] --> FS
  T --> FS["forward_sample<br/>x_t = alpha(t) x_0 + sigma(t) noise"]

  FS --> NET["network(x_t, u, cond)"]
  U --> NET

  X0 --> TGT["training_target<br/>a(t) x_0 + b(t) eps"]
  NZ --> TGT
  T --> TGT

  T --> WF["weight_func(t)"]

  NET --> L["weighted MSE<br/>mean( w * (pred - target)^2 )"]
  TGT --> L
  WF --> L

  L --> BP["Adam + LR scheduler"]
  NET --> DIAG["clean_loss: same error<br/>mapped back to x_0 space<br/>(comparable across params)"]
</pre>

That `clean_loss` is a small but useful idea: raw loss values aren't comparable
between parameterizations, because each regresses a different target. Projecting
the error back into $$x_0$$ space through `to_x_0` gives a number you *can*
compare across runs.

Three training procedures share this loop behind one interface:

- **standard** — plain denoising regression.
- **cfg** — classifier-free guidance. Labels are randomly replaced by a null
  token (`num_classes` is used as the null index) with probability `p_uncond`,
  so one network learns both the conditional and unconditional model. At
  sampling time a `GuidedModel` wrapper runs both passes per step and returns
  $$m_\text{uncond} + w\,(m_\text{cond} - m_\text{uncond})$$.
- **reflow** — straightens the transport map, below.

## Reflow

Reflow retrains on *coupled* noise/data pairs instead of independently sampled
noise, which straightens the ODE trajectories and makes few-step sampling
viable. The pairs come from running the teacher's own probability-flow ODE
backwards — the same Euler solver, with the time grid flipped:

<pre class="mermaid">
flowchart LR
  DATA["x_0 — real data"] --> REV
  TCH["teacher<br/>standard-trained or loaded"] --> REV["EulerSolver, flipped time grid<br/>integrate PF-ODE data → noise"]
  REV --> X1["x_1 — the noise this<br/>teacher maps x_0 to"]
  DATA --> PAIR
  X1 --> PAIR["TensorDataset(x_0, x_1)"]
  PAIR --> STU["student: same loss, but<br/>noise := x_1 instead of randn"]
  STU --> PROMOTE["student becomes the<br/>new teacher, repeat n_steps"]
  PROMOTE -.-> REV
</pre>

Only `ReflowProc.training_loss` differs from the standard one — it passes the
paired `x_1` into the existing `training_loss` as the noise argument, rather
than letting it sample fresh noise. Everything downstream is unchanged.

## Samplers

Four samplers consume the parameterization interface, so any of them works with
any trained parameterization:

| Solver | Uses | Notes |
|---|---|---|
| **Euler** | `pfode` | Deterministic ODE integration; also used in reverse to build reflow pairs. |
| **Euler–Maruyama** | `rev_sde` | Stochastic; noise scale from `schedule.gamma(t)`, with the final step taken noise-free. |
| **DDIM** | `to_x_0` + `to_noise` | Re-noises directly to the next time level: $$x_t = \alpha_t \hat{x}_0 + \sigma_t \hat{\epsilon}$$. |
| **DEIS** | `to_noise` | Multistep exponential integrator. Keeps a buffer of past noise predictions and weights them by Lagrange-polynomial coefficients computed with Gauss–Legendre quadrature. |

## Data

Two 2D targets, both standardized to unit variance so they match the
standard-normal prior:

- **blobs** — a Gaussian mixture via scikit-learn, with labels (so it's the one
  used for classifier-free guidance).
- **tree** — the 2D fractal mixture from NVIDIA's *"Guiding a Diffusion Model
  with a Bad Version of Itself"*, which has exact ground-truth density and score.
  Useful precisely because you can check against truth rather than eyeballing a
  scatter plot.

Staying in 2D is deliberate: every quantity can be plotted, so the questions
being asked are about the *diffusion machinery* rather than about network
capacity.

## Status and next steps

The repo trains, samples, and produces diagnostic figures (loss curves, prior
check, transport trajectories, density comparison, singularity report) into
per-run Hydra output directories. From the author's own notes, the open threads
are better pair selection for velocity-based training, and seeing how
parameterizations compare as the data gets higher-dimensional.

---

[⬅ Back to Projects](./projects.html) · [Home](./index.html)
