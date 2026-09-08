# `SweptTube`: fixing the n-plane gauge of `rate_of_strain`

**Date:** 2026-09-08 **Status:** approved design, not yet implemented **Closes:** #829. Follow-up to #828.

## Problem

`rate_of_strain(chart_at_time, point, t)` computes $K_{ij} = \tfrac12\,\partial_t\gamma_{ij}$ by differentiating a family of `TubularChart`s across time. The documented idiom builds each slice independently:

```python
lambda t: TubularChart(BishopBuilder(AtTime(curve, t), "km"), tau_bounds=...)
```

`BishopBuilder` seeds its transport from `_auto_initial_normal(T0)`, which picks the world axis least aligned with the tangent (`argmin |T0|`) and Gram--Schmidts it. The seed is therefore anchored to the world frame, not to the curve. When the tangent at $\tau_0$ rotates with time, the labels $(n_1, n_2)$ denote a _different physical point_ on each slice, and $\partial_t\gamma_{ij}$ differentiates the frame's gauge along with the geometry.

### Measured

A rigid rotation is an isometry, so $K_{ij}$ must vanish identically. It does not.

| curve / motion | seed | $\vert K\vert_{\max}$ |
| --- | --- | --- |
| helix (pitch 0.4), rigid $\hat y$ rotation | world axis | `0.117767` |
| helix (pitch 0.3), rigid $\hat y$ rotation | world axis | `0.099216` |
| helix (pitch 0.3), rigid rotation about $(2,-1,3)$ | world axis | `0.002662` |
| helix, rigid translation | world axis | `0.0` |
| helix, rigid rotation about $\hat x$ or $\hat z$ | world axis | `0.0` |

Two consequences for testing:

- **Translations are equivariant even when the seed is broken** ($T_0$ does not move, so its Gram--Schmidt against a world axis does not either). A rigid-translation test proves nothing.
- **The bug is axis-dependent.** Rotations that fix $e_k$, or act within $\mathrm{span}(T_0, e_k)$, are exactly equivariant. Any regression test **must** use a generic axis.

Localising it confirms the cause is the seed and nothing else:

| point | $\gamma_{\tau\tau}$ | $K_{\tau\tau}$ |
| --- | --- | --- |
| $n = (0,0)$ — on axis, gauge-free | 1.160000 | `-0.000000` |
| $n = (0.2, 0.1)$ | 1.588498 | `-0.117767` |
| $n = (0.2, 0.1)$, seed carried with the body | — | `6.7e-08` |

### Second failure mode: a discontinuous gauge

`argmin` is a discrete choice, so the seed jumps when two tangent components cross. On the rigidly rotating helix, at $\theta = \pi/4$:

| t      | $\gamma_{\tau\tau}$ | reported $K_{\tau\tau}$ |
| ------ | ------------------- | ----------------------- |
| 0.7800 | 1.506062            | `-0.175294`             |
| 0.7900 | 0.849303            | `-0.126384`             |

$\gamma$ jumps by 0.66. The true $\partial_t\gamma_{\tau\tau}$ across that point is $\approx -66\,\mathrm{s}^{-1}$; `jacfwd` reports $-0.35$. Forward-mode AD differentiates the branch it landed in and never sees the jump, so the function returns a number that is not the derivative of anything, with no diagnostic.

## Why this cannot be fixed by transporting the seed

The obvious repair — transport the seed rotation-minimisingly in $t$ — does not work, and the reason is structural rather than a gap in the implementation.

Decompose the angular velocity in the frame $\{T, U, V\}$ as $\omega = a\,T + b\,U + c\,V$. A material carry gives $\dot U = aV - cT$; rotation-minimising transport gives $\dot U = -cT$. They differ by $aV$, where $a = \omega\cdot T$ is the spin about the tangent. Measured on the rotating helix ($a = 0.928\ \mathrm{rad/s}$):

| seed                    | $K_{\tau\tau}$ |
| ----------------------- | -------------- |
| material carry          | `-0.000000`    |
| RMF-in-$t$              | `-0.112967`    |
| world axis (status quo) | `-0.117767`    |

RMF-in-$t$ buys determinism and continuity; it does not buy correctness.

**The n-plane material labelling is not contained in $r(\tau, t)$.** A rod spinning about its own axis and one at rest trace the same curve. Rate of strain of a _tube_ is a property of a **framed** curve — a Cosserat/director structure — not of a curve. Any design that derives the gauge from the tangent alone is deriving data that is not there.

## Key lemma: where the seed does and does not matter

In a **Bishop** frame the induced metric is exactly block-diagonal,

$$\gamma = \mathrm{diag}\!\left(|r'|^2\,(1 - n_1k_1 - n_2k_2)^2,\; 1,\; 1\right),$$

because $U_i' = -k_i T$ kills the cross terms (unlike Frenet, which carries a torsion term). The seed therefore enters $\gamma$ through exactly **one scalar**: the angle $\varphi$ of the curvature vector in the $(U_1, U_2)$ plane. Hence

> $\gamma$ is seed-independent **iff** $\boldsymbol\kappa \cdot (n_1, n_2) = 0$.

Verified numerically: on a tilted straight line, the full $3\times3$ metric is the identity to $5.4\times10^{-17}$ for five different seeds (auto, $e_y$, $e_z$, two random), with maximum inter-seed deviation $2.2\times10^{-16}$; under rigid rotation $\vert K\vert_{\max} \le
1.1\times10^{-16}$ for all five.

This lemma is what the whole design hinges on: it identifies the set on which a degenerate seed is provably harmless. Note the design still _raises_ on that set rather than falling back (see below) — what the lemma buys is the guarantee that raising there costs the caller only convenience, never correctness, because on that set any normal they supply is as good as any other.

## Rejected: the geometric (Frenet-normal) seed

Seeding from $\widehat{d\mathbf{T}/d\tau}$ is equivariant under rigid motion and essentially free (`_transport_start` already builds `dTangent_fn`). It gives $\vert K\vert = 1.4\times
10^{-16}$ on the rotating helix. It was still rejected:

**It flips sign through an inflection, and the flip locus is exactly $\kappa = 0$** — which a $\kappa > \varepsilon$ guard steps over. On $r(s,t) = (s,\ (t-0.5)s^2 + s^3/3,\ 0)$, with $\kappa(\tau_0) = 2\times10^{-3} \ggg \varepsilon$ at both sample times:

| t     | seed       | $\gamma_{\tau\tau}$ |
| ----- | ---------- | ------------------- |
| 0.499 | `[0,-1,0]` | 4.395369            |
| 0.501 | `[0,+1,0]` | 3.353607            |

$\gamma$ jumps by 1.04 over $\Delta t = 0.002$; `rate_of_strain` returns a smooth-looking 4.2615 and 4.4714 where the true $\Delta\gamma/2\Delta t \approx -260$. **The world-axis seed is continuous there** — the geometric seed would be _worse_ than the status quo at inflections, reproducing exactly the failure mode above.

Two further findings from the same review:

- A `if kappa > eps` guard is **not implementable**: `kappa` is traced, so it raises `TracerBoolConversionError` under `jit` and `vmap`. It appears to work eagerly only because `jacfwd`'s JVPTracer leaks a concrete primal.
- Conditioning near an inflection is _not_ a problem — `normalize` is homogeneous of degree zero, and equivariance stays exact ($\le 1.1\times10^{-16}$) down to $\varepsilon = 10^{-8}$. The sign flip is the real hazard.

## Rejected: the curvature-weighted mean seed

Averaging the curvature direction over $\tau$ cancels on a helix, whose curvature vector rotates in the Bishop frame at the torsion rate. Conditioning ratio (weighted-mean magnitude over mean $|\kappa|$):

| $\tau$ range | 3.0   | 6.0   | 11.4  | 21.86      | 45.6  |
| ------------ | ----- | ----- | ----- | ---------- | ----- |
| ratio        | 0.967 | 0.881 | 0.609 | **0.0016** | 0.041 |

The ratio oscillates rather than decaying — it passes through a null each torsion period — so the $L=45.6$ entry being above the $L=21.86$ one is expected, not an error. The first null is at $L = 21.8624$. Note the rotation rate is the torsion **per arc length** times $|r'|$: $0.275229 \times \sqrt{1.09} = 0.287348\ \mathrm{rad/km}$, so the period is $21.86$, not $22.83$. It also makes the seed depend on `tau_bounds`, so the same curve charted over a different $\tau$ extent would get different $(n_1, n_2)$ coordinates.

## Design

### The seed rule

Any single continuous rotation-equivariant unit normal field over curve space must vanish somewhere, so no formula avoids a degeneracy locus. The design goal is therefore not to avoid degeneracy but to **align the degeneracy locus with the set where the lemma proves the seed does not matter**.

The **centroid seed** does that:

```
w = P_perp_T0 ( mean_{tau in tau_bounds} r(tau)  -  r(tau_0) )
```

then through the existing `_orthonormalize`. Measured properties:

- equivariant under rigid rotation: $\vert K\vert = 5.6\times10^{-17}$ (helix), $2.2\times10^{-16}$ (curve with $r''(\tau_0) = 0$);
- **continuous through an inflection**: $\gamma_{\tau\tau} = 3.335749 \to 3.353607$, matching the world-axis seed exactly, where the geometric seed jumps;
- well defined at $\kappa(\tau_0) = 0$, with no derivative-order selection;
- no data-dependent Python branch, so it survives `jit` and `vmap`.

**Degeneracy set — larger than "straight".** The centroid coincides with $r(\tau_0)$ for any curve odd-symmetric about $\tau_0$, not only straight ones:

| case | $\vert w_\perp\vert$ | verdict |
| --- | --- | --- |
| cubic $s^3$, $\tau_0$ centred $[-1,1]$ | 1.42e-17 | degenerate (not straight) |
| sine, $\tau_0$ centred $[-\pi,\pi]$ | 1.61e-16 | degenerate (not straight) |
| cubic $s^3$, $\tau_0$ at end $[0,3]$ | 6.75 | ok |
| helix, $\tau_0$ at end $[0,3]$ | 0.983 | ok |
| straight line | 0.0 | degenerate |

This is harmless in practice because `tau_0` defaults to the lower bound rather than the centre, but that is a property of the default, not a theorem. It must be documented as such.

**Degeneracy is a raise, never a fallback.** A `jnp.where` blend to the world axis is rejected: its AD derivative is identically zero, so it would be silently wrong near the boundary. The rejection routes through `_orthonormalize`'s existing `eqx.error_if` (bishop.py:164), which is `jit`-safe and whose message already names `initial_normal`.

> **Guard against NaN.** `normalize(0)` yields `[nan nan nan]` silently. The seed must be passed to `_orthonormalize` _unnormalised_ so the existing `error_if` fires, rather than pre-normalised.

### `SweptTube`

```python
class SweptTube(eqx.Module):
    """One-parameter family of tubular slices: t -> TubularChart."""

    curve: Any  # two-argument gamma(tau, t)
    tau_unit: ... = eqx.field(static=True)
    tau_bounds: tuple[Any, Any]  # kw_only; the seed rule needs this
    builder: type = eqx.field(static=True, default=BishopBuilder)
    director: Callable[[Any], Any] | None = None
    # solver / n_seed passthrough

    def __call__(self, t) -> TubularChart: ...
```

- **The seed rule lives here, not on `BishopBuilder`** — forced, because it needs `tau_bounds`, which the builder does not own. `__call__` computes the centroid seed and passes it down as `initial_normal`.
- **`builder` carries the choice.** `FrenetSerretBuilder` needs no seed and is _already_ equivariant ($\vert K\vert = 6.0\times10^{-8}$ vs Bishop's $2.7\times10^{-3}$ on the same generic-axis rigid rotation), because $\mathbf{N}$ and $\mathbf{B}$ are fixed pointwise by the curve with no transport. The gauge bug is Bishop-specific. Keeping Frenet preserves working behaviour and is required for the non-vacuous symmetry test below.
- **`director` is a callable of $t$**, defaulting to the centroid seed. A fixed vector cannot express a materially spinning rod — the Cosserat case where the caller knows the gauge and the library provably cannot derive it.

### `rate_of_strain` narrows to `SweptTube`

The raw-callable overload is removed. An opaque `Callable[[Any], Any]` cannot be inspected, so the gauge contract could only ever be advisory; taking a `SweptTube` makes it structural. `rate_of_strain` is new in #828 and unreleased, so there is no deprecation burden.

### `BishopBuilder` requires an explicit seed

`_auto_initial_normal` is removed. A `BishopBuilder` with no `initial_normal` raises. This leaves no arbitrary gauge anywhere in the library.

> **Scale, stated plainly.** This is a hard breaking change across ~258 references in 25+ files, including doctests. A measurement of a _different_ seed change produced 116 failures against 991 passes; this change touches every construction site rather than only those whose value moves, so it is larger. It was chosen deliberately over the non-breaking alternative (leave the bare default alone, since a single chart's gauge is arbitrary but harmless — the bug exists only when differencing two independently-seeded charts).

## Acceptance criteria

| # | Check | Rationale |
| --- | --- | --- |
| 1 | Rigid rotation, **generic axis**, $\vert K\vert < 10^{-9}$ | The headline. Axis-aligned rotations pass even when broken |
| 2 | $\gamma$ continuous through a $\kappa=0$ crossing; $K$ matches finite differences | The failure that killed the geometric seed |
| 3 | Closed form: uniform stretch of a straight line, $K_{\tau\tau} = c(1+ct)$ | Analytic anchor; verified to 1.5e-8 |
| 4 | $\gamma^{ij}K_{ij} = \partial_t\ln\sqrt{\det\gamma}$ | Independent invariant; verified exact |
| 5 | Frenet family: off-diagonal $K \neq 0$ and $K = K^\mathsf{T}$ | Bishop's metric is always diagonal, so the existing symmetry test is vacuous |
| 6 | Degenerate seed raises, naming `initial_normal` | Straight curve, and odd-symmetric-about-$\tau_0$ |
| 7 | `jit` and `vmap` over scalar `t` both work | What killed the `if kappa > eps` rule |
| 8 | Bishop-with-centroid and Frenet agree on-axis ($n = 0$) | Cross-builder check where the gauge cancels |

Rigid _translation_ is worth one line but proves nothing alone.

## Sequencing

**PR 1 — cleanup.** Stops the corrupted equation shipping immediately.

- Fix `$$K*{ij}$$` / `\partial*t\gamma*{ij}` in `docs/curve-charts.md`. **Root cause proven:** the repo's own `prettier-markdown-no-wrap` hook, pinned at `v3.8.1` with `--prose-wrap=never`, rewrites `_` to `*` inside a single-line `$$…$$`. Reproduced byte-for-byte from clean input; prettier 3 latest does not do it, so it is specific to the pinned version. The author wrote correct LaTeX and the hook silently mangled it, which is why it shipped and why `nox -s docs` stayed green. **Verified fix:** put the `$$` delimiters on their own lines — that form round-trips through prettier 3.8.1 unchanged. Any other display math in the docs must use the same form or it will be re-corrupted on the next commit.
- State the sign convention: this is Wald's $+$, opposite to Baumgarte--Shapiro and Alcubierre.
- `strain.py:81` `.matrix.value` -> `ustrip(unit0)`; `strain.py:75` remove the dead `t.value if t_unit is not None else t` branch.

**PR 2 — `SweptTube` and the gauge fix.** The type, the centroid seed, the `BishopBuilder` breaking change, and all eight acceptance tests. Closes #829.

**PR 3 — documentation.** Rewrite the derivation as $K_{ij} = \tfrac12(\mathcal{L}_T\gamma)_{ij}$ with $T$ the tube's declared time flow, replacing "the Lie-drag term vanishes by construction". State that $K_{ij}$ is a slice 2-tensor only under _time-independent_ spatial relabelling, and that rate of strain of a tube is a property of a framed curve.

## Behaviour regressions to declare

1. `BishopBuilder` without `initial_normal` now raises (~258 call sites).
2. `rate_of_strain` no longer accepts a raw callable.
3. A `SweptTube` whose curve is straight, or odd-symmetric about $\tau_0$, raises and requires an explicit `director`.
