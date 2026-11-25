
# What components should be learned?

Sadly, so far none at BNN size. We systematically tested the components written in <span style="color:#3b82f6">blue</span> below.

## Setup / sampler components we tried to learn

Given the posterior $\pi[\theta]\propto e^{-U[\theta]}$ with $U[\theta]=-\log p[\mathcal D\mid\theta]-\log p[\theta]$ and $\theta\in\mathbb R^D$.
Every sampler we tried is defined by:

1. The <span style="color:#3b82f6">coordinates</span> $T$: a diffeomorphism $T:\mathbb R^D\to\mathbb R^D$. Sample $z$ with $\theta=T[z]$ (default: $T = \mathrm{id}$), so the chain sees

$$
\pi_z[z]=\pi\big[T[z]\big]\,\big|\det\nabla T[z]\big|,
$$

where $\nabla T$ is the Jacobian of $T$. An affine map $T[z]=\mu+C^{1/2}z$ is the same as a constant metric $C$ (below). Learned maps are coupling flows ([Dinh et al., 2017](https://arxiv.org/abs/1605.08803)), used as coordinates as in NeuTra ([Hoffman et al., 2019](https://arxiv.org/abs/1903.03704)) trained using reverse KL

2. The local transition, HMC ([Neal, 2011](https://arxiv.org/abs/1206.1901)), where one transition is $L$ leapfrog steps: Draw a momentum $r\sim\mathcal N[0,M]$, then repeat $L$ times

$$
r\leftarrow r-\tfrac{\epsilon}{2}\,g[\theta],\qquad
\theta\leftarrow\theta+\epsilon\,C\,r,\qquad
r\leftarrow r-\tfrac{\epsilon}{2}\,g[\theta],
$$

and accept the endpoint $(\theta',r')$ with probability $\min\{1,e^{-\Delta H}\}$, where $\Delta H=H[\theta',r']-H[\theta,r]$ and $H[\theta,r]=U[\theta]+\tfrac12 r^\top C\,r$ uses the *exact* $U$. Here,

- $g\approx\nabla U$ is the <span style="color:#3b82f6">force</span>. Any $g$ that depends only on $\theta$ keeps the leapfrog reversible and volume-preserving, so the chain stays exact and an error in $g$ costs only acceptance.
- $C$ is the <span style="color:#3b82f6">metric</span> (preconditioner), strictly the *inverse* metric: $C=M^{-1}$ for the mass matrix $M$. In HMC it is constant.
- $\epsilon$, $L$ are tuned.

With $L=1$ this is MALA (Roberts and Tweedie, 1996, Bernoulli 2:341), the Gaussian proposal

$$
\theta'\sim\mathcal N\!\big[\theta-\tfrac{\epsilon^2}{2}C\,g[\theta],\ \epsilon^2C\big].
$$

A position-dependent metric $C[\theta]$ ([Girolami and Calderhead, 2011](https://doi.org/10.1111/j.1467-9868.2010.00765.x)) would break the reversibility of the leapfrog above. We used it only in this proposal, $Q[\theta'\mid\theta]=\mathcal N\big[\theta'\mid\theta-\tfrac{\epsilon^2}{2}C[\theta]\,g[\theta],\ \epsilon^2C[\theta]\big]$, accepted with probability $\min\big[1,\ \pi[\theta']\,Q[\theta\mid\theta']\,/\,(\pi[\theta]\,Q[\theta'\mid\theta])\big]$.

The unadjusted samplers drop the accept step. SGHMC and L2E add friction $\gamma$, temperature $\tau$ and fresh noise $\xi\sim\mathcal N[0,I]$ at every step, and then leave $\pi^{1/\tau}$ only approximately invariant (so $\pi$ only approximately at $\tau=1$):

- SGHMC ([Chen et al., 2014](https://arxiv.org/abs/1402.4102)), here with full gradients and $M=I$:

$$
r\leftarrow r-\epsilon\,[g+\gamma r]+\sqrt{2\gamma\epsilon\tau}\,\xi,\qquad
\theta\leftarrow\theta+\epsilon\,r.
$$

- L2E ([Kim et al., 2024](/Users/manuel/Documents/Library/kimLearningExploreStochastic2024.pdf)) generalizes SGHMC's update with two coordinatewise <span style="color:#3b82f6">kinetic fields</span> $\alpha,\beta$ of $(\theta_i,r_i,g_i)$ and running averages of $g_i$:

$$
r\leftarrow r-\epsilon\,[g+\alpha+\gamma\beta]+\sqrt{2\gamma\epsilon\tau}\,\xi,\qquad
\theta\leftarrow\theta+\epsilon\,\beta,
$$

where $\beta$ in the second update is evaluated at the new $r$, and $\alpha=0$, $\beta=r$ is SGHMC. The fields stand for $\nabla_\theta$ and $\nabla_r$ of a learned kinetic energy. The paper uses $\tau=1$.

3. Global moves: With probability $s$, replace the local step by a <span style="color:#3b82f6">global proposal</span> $\theta'\sim q$, an independence Metropolis–Hastings move ([Tierney, 1994](/Users/manuel/Documents/Library/tierneyMarkovChainsExploring1994.pdf)) with acceptance $\min\Big[1,\ \frac{w[\theta']}{w[\theta]}\Big], w=\pi/q.$ Not part of this page, see [here](https://app.notion.com/p/oct-nov-41d85150a6ea8062b00ee1e1b41b645d) for detailed results.

## Experiments

For each component, we first tried synthetic targets (a Gaussian, a banana shaped distribution, a funnel, two mode mixtures) in varying dimensions D (D up to 128), then BNNs: three UCI MLPs (D = 658-754) and small networks (D = 25 to 66).

### Synthetic targets

Each synthetic target is a standard normal pushed through a known map $T$ (corresp. to the funnel, banana): $\theta=T[z]$ with $z\sim\mathcal N[0,I]$. For these, we compared three arms:

- Classical, i.e., the best (or rather a good) performing rule that needs no learning, e.g., $g=\nabla U$; $C$ = covariance of warmup draws, or the inverse of a mean curvature (the Hessian on synthetic targets, $\bar F+I$ on BNNs, see below), $\epsilon,L,\gamma,\tau$ tuned on a grid.
- Oracle: the ideal value for that component. E.g., $T$ = the exact map to $\mathcal N[0,I]$, $q=\pi$ (exact draws).
- Learned: A fitted $g_\phi$, $C_\phi$, $T_\phi$, $(\alpha_\phi,\beta_\phi)$, ..., with learned parameters $\phi$.

As we cannot easily measure how close a chain is to $\pi$, we use

$$
h[\theta]=\frac{\theta_k^2}{\mathbb E_\pi[\theta_k^2]},\qquad \mu_h=\mathbb E_\pi[h]=1 \qquad
\mathbb E_\pi[\theta_k^2]=\mathbb E_z\big[T_k[z]^2\big].
$$

as a proxy ($\theta$ is just a sample / point). We choose the $k$-th axis of that point so that a chain that doesn't explore enough is most wrong (i.e., the widest direction for the Gaussian, the scale coordinate of the funnel, the bent of the banana).
The hope is that a 'good' sampler estimates this well (e.g., because it appropriately samples the funnel), while a 'bad' sampler does not.

We evaluate $h$ at every draw $\theta^{(t)}$ a chain produces and average over the $n$ draws it makes within budget $B$,

$$
\hat h[B]=\frac1n\sum_{t=1}^n h\big[\theta^{(t)}\big],
$$

the chain's estimate of $\mu_h$. Run $R$ independent chains $j$, each giving its own $\hat h_j[B]$, and compute

$$
\mathrm{RMSE}[B]=\Big(\tfrac1R\textstyle\sum_{j=1}^R\big(\hat h_j[B]-\mu_h\big)^2\Big)^{1/2}.
$$

$B$ is counted in gradient equivalents and includes everything:

$$
B=n_{\nabla U}+\tfrac{c_U}{c_{\nabla U}}\,n_U+\frac{c_\phi}{c_{\nabla U}}\,n_\phi+B_\text{fit},
$$

Here $n_{\nabla U}$, $n_U$ and $n_\phi$ count the evaluations of the gradient, of $U$ and of the learned part, and $c_{\nabla U}$, $c_U$ and $c_\phi$ are the costs of one evaluation of each. $B_\text{fit}$ is the cost of fitting the learned part (also in gradients).

Invert the curve to $B_\text{arm}[e]$, the budget an arm needs for $\mathrm{RMSE}=e$. We can then define

$$
\text{headroom}[e]=\frac{B_\text{classical}[e]}{B_\text{oracle}[e]},\qquad
\text{realized}[e]=\frac{B_\text{classical}[e]}{B_\text{learned}[e]}.
$$

We read both ratios at the $e$ the oracle (learned) arm reaches at its largest budget.

Interpreting:

- headroom $\approx1$: no learner of this component can help on this target.
- headroom $\gg1$ but realized $<1$: the component is worth something, but the learned form or the learner fails to reach it.
- realized $>1$: worth learning under those conditions.

### Results by component (synthetic)

Headroom and Realized: times less compute than classical (>1 = better); ranges span targets and $D$.

| Component                  | Classical                  | Oracle (ceiling)                     | Learned                    |                                         Headroom |                           Realized | Why it fails                                                            |
| -------------------------- | -------------------------- | ------------------------------------ | -------------------------- | -----------------------------------------------: | ---------------------------------: | ----------------------------------------------------------------------- |
| Force $g$                  | exact $\nabla U$           | exact $\nabla U$ at zero cost        | per-target MLP             |     $\le24$ |                               $<1$ | fit costs far more than it saves                                        |
| Metric $C[\theta]$ in MALA | tuned, adapted constant $C$ | exact local metric                   | not fitted                 | 0.09–0.4 (oracle *slower*); 1.4–8 on the funnel |                                  — | no headroom                                                             |
| Coordinates $T$            | tuned, adapted constant $C$ | exact Gaussianizing $T$              | coupling flow, reverse KL  |                                          8–1,400 | 0.8–3.2, falling to $<0.3$ at $D=128$ | flow fit degrades with dimension; fit costs about what it saves |

So every component has at least one of three problems:

- it has no headroom (metric $C[\theta]$),
- clearly does not scale well with dimension (coordinates $T$),
- the learner reaches part of the headroom, but its fit costs too much (force, also $T$).

### Results by component (BNNs)

Naturally, the UCI MLPs have no reference. The small networks only have a sampled one (but judging by prediction $\widehat R$, seemingly converged). The verdict stays the same everywhere: No learned component improves over the classical component.

- Force $g$: no. The learned force models in Representing and Learning BNN Samplers cost 22 to 234 gradients per evaluation and do not improve anything.
- Metric $C[\theta]$: no. On the UCI MLPs the [local metric](https://rss.onlinelibrary.wiley.com/doi/10.1111/j.1467-9868.2010.00765.x) is $F[\theta]+I$ where $F[\theta]=J^\top J/\sigma^2$ is the Fisher information of the Gaussian likelihood (with $J$ the Jacobian of the network's outputs at the training inputs), and $I$ is the precision of the standard normal prior. Random-walk Metropolis shaped by the exact local metric has 26 to 37 times the expected squared jump distance of one without a metric.
- But even better: constant metric $C$: The mean $\bar F+I$ over 32 draws of the chain, used as $C=(\bar F+I)^{-1}$, gets a chain in 7,500 to 22,000 gradients to where the chain without a metric is after 400,000 (still not stationary at that point, so only burn-in tested).
- Disappointingly, a flow started at the constant metric $(\bar F+I)^{-1}$ to the coordinates $T$ seems to get worse the longer it is fitted. On airfoil it moves predictions 0.86 times as fast after 16,000 gradients of fit, and 0.18 times as fast after 131,000.
- Kinetic fields $\alpha,\beta$: no. Trained as published on small regression networks ($D=49$), L2E ties tuned SGHMC in held out RMSE but ends up farther from the posterior (measured in terms of the predictive distribution on probes).

## Takeaways / other things we stumbled upon

- Adding a preconditioner seems to speed up warmup by quite a lot.
- integrating out the Gaussian readout exactly (the exact case of the readout reparametrization of [Hron et al., 2022](/Users/manuel/Documents/Library/hronWideBayesianNeural2022.pdf)) helps sampling (so we sample only hidden weights and noise), and can repair generative models (e.g., what we tried when experimenting with BNN samples as a data modality) a bit.
