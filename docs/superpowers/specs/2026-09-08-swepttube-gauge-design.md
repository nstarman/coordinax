# `SweptTube`: the n-plane gauge of `rate_of_strain`, and four defects beneath it

**Date:** 2026-09-08 **Status:** approved design, not yet implemented **Closes:** #829. Follow-up to #828.

## Summary

`rate_of_strain` reports the drift of an arbitrary frame gauge as physical strain. The fix is **not** a better seed rule: four were tried and all four fail, for a reason that is now quantified rather than guessed. The n-plane gauge is Cosserat director data, absent from $r(\tau, t)$, so `SweptTube` requires the caller to supply it and refuses to guess.

An independent audit of the surrounding package found three further defects, one of which (`nearest_tau` returning a local _maximum_) is a silently wrong answer through the public `pt_map` and is unrelated to the gauge work.

## Problem

The documented idiom builds each slice independently:

```python
lambda t: TubularChart(BishopBuilder(AtTime(curve, t), "km"), tau_bounds=...)
```

`BishopBuilder` seeds its transport from `_auto_initial_normal(T0)` (`argmin |T0|` over the world axes). The seed is anchored to the world frame, not the curve, so when the tangent at $\tau_0$ rotates with time the labels $(n_1, n_2)$ denote a _different physical point_ on each slice, and $\partial_t\gamma_{ij}$ differentiates the gauge along with the geometry.

### Measured

A rigid rotation is an isometry, so $K_{ij}$ must vanish identically. It does not.

| curve / motion                                     | $\vert K\vert_{\max}$ |
| -------------------------------------------------- | --------------------- |
| helix (pitch 0.4), rigid $\hat y$ rotation         | `0.117767`            |
| helix (pitch 0.3), rigid $\hat y$ rotation         | `0.099216`            |
| helix (pitch 0.3), rigid rotation about $(2,-1,3)$ | `0.002662`            |
| helix (pitch 0.3), rigid rotation about $\hat z$   | `0.019980`            |
| helix, rigid rotation about $\hat x$               | `5.55e-17`            |
| helix, rigid translation                           | `0.0`                 |

Two consequences for testing:

- **Translations are equivariant even when the seed is broken** — $T_0$ does not move, so its Gram--Schmidt against a world axis does not either. A rigid-translation test proves nothing.
- **The bug is axis-dependent.** Only rotations that fix $e_k = \arg\min|T_0|$, or act within $\mathrm{span}(T_0, e_k)$, are equivariant — here that is $\hat x$ alone. $\hat z$ is **not** safe. Any regression test must use a generic axis.

Localising it confirms the seed is the whole cause:

| point | $\gamma_{\tau\tau}$ | $K_{\tau\tau}$ |
| --- | --- | --- |
| $n = (0,0)$ — on axis, gauge-free | 1.160000 | `-0.000000` |
| $n = (0.2, 0.1)$ | 1.594483 | `-0.117767` |
| $n = (0.2, 0.1)$, seed carried with the body | — | `6.7e-08` |

### Second failure mode: a discontinuous gauge

`argmin` is a discrete choice, so the seed jumps when two tangent components cross. On the rigidly rotating helix at $\theta = \pi/4$, $\gamma_{\tau\tau}$ jumps `1.506062` -> `0.849303` between $t = 0.78$ and $t = 0.79$. The true $\partial_t\gamma_{\tau\tau}$ across that point is $\approx -66\,\mathrm{s^{-1}}$; `jacfwd` reports `-0.35`. Forward-mode AD differentiates the branch it landed in and never sees the jump.

## Why no automatic seed rule works

Four rules were tried against the real library. All four fail, and the failures share a structure.

### 1. World axis (`argmin |T0|`, the status quo) — not equivariant, and discontinuous

Both failures above.

### 2. RMF-in-$t$ — smooth, but not equivariant

Decompose the angular velocity in $\{T, U, V\}$ (right-handed, $T \times U = V$) as $\omega = a\,T + b\,U + c\,V$. The frame angular velocity is pinned by $\Omega \times T =
\dot{T}$ to $\Omega = b\,U + c\,V + \alpha\,T$; a material carry keeps $\alpha = a$, giving $\dot U = a\,V - c\,T$, while RMF sets $\alpha = 0$, giving $\dot U = -c\,T$. They differ by $aV$, where $a = \omega\cdot T$ is the spin about the tangent. Measured ($a = 0.928\ \mathrm{rad/s}$):

| seed           | $K_{\tau\tau}$ |
| -------------- | -------------- |
| material carry | `-0.000000`    |
| RMF-in-$t$     | `-0.112967`    |
| world axis     | `-0.117767`    |

### 3. Geometric (Frenet-normal) seed — equivariant, but flips through an inflection

$\widehat{d\mathbf{T}/d\tau}$ gives $\vert K\vert = 1.4\times10^{-16}$ under rigid rotation. But it sign-flips through an inflection, and the flip locus is exactly $\kappa = 0$. On $r(s,t) = (s,\ (t-0.5)s^2 + s^3/3,\ 0)$, with $\kappa(\tau_0) = 2\times10^{-3}$ at both times:

| t     | seed       | $\gamma_{\tau\tau}$ |
| ----- | ---------- | ------------------- |
| 0.499 | `[0,-1,0]` | 1.484697            |
| 0.501 | `[0,+1,0]` | 0.709800            |

evaluated at $(\tau, n_1, n_2) = (0.5, 0.2, 0)\,\mathrm{km}$, `tau_bounds` $= (0,1)\,\mathrm{km}$. measured through the library. The single-point cross-check an earlier draft gave does **not** derive them: at $s = 0.5$, $t = 0.5$ the curvature is $0.913076$, and $1.0625(1 \mp 0.2\kappa)^2$ yields 1.48599 / 0.70987. The quoted values need the _$t$-dependent_ pair — $(|r'|^2, \kappa) = (1.062001, 0.911893)$ at $t = 0.499$ and $(1.063001, 0.914255)$ at $t = 0.501$. (An earlier draft recorded 4.395369 / 3.353607, unattainable at the stated point — that family never reaches $|r'| \approx 1.96$ in bounds.) $\gamma$ jumps by 0.775 across $\Delta t = 0.002$, a _spurious_ rate of $\approx -390$. **The world-axis seed is continuous there** — this rule is _worse_ than the status quo at inflections.

### 4. Centroid seed — equivariant and inflection-continuous, but its own locus is worse

$w = P_{\perp T_0}(\langle r\rangle_{\tau} - r(\tau_0))$ is equivariant under rotation ($3.7\times10^{-16}$), translation ($5.4\times10^{-16}$) and reflection (exactly 0), and is continuous through an inflection. It still fails:

- Its degeneracy locus is the _vector condition_ $\langle r\rangle - r(\tau_0) \parallel
  T(\tau_0)$ — **codimension 1** for planar curves, not a symmetry. Counterexample: $r(s) = (s,\ s^2 - \tfrac43 s^3,\ 0)$ on $[0,1]$, $\tau_0 = 0$: curved ($\kappa = 2$), not odd-symmetric, $\tau_0$ at the lower bound, and degenerate.
- Codimension 1 means ordinary time-dependent families **cross it transversally**: $\gamma_{\tau\tau}$ jumps `1.958321` -> `0.359281` across $\Delta t = 0.002$ where the true rate is $\approx -800\,\mathrm{s^{-1}}$, with $\kappa \approx 2$ throughout so nothing about curvature flags it. Near the locus $K$ diverges as $-0.399/\varepsilon$ (`-79.74` against a true `+0.84`), and **central differences confirm the wrong value to 5 digits**.
- It depends on `tau_bounds` — 47.5° gauge rotation between bounds $(0,1)$ and $(0,8)$ — which is the objection used to reject the curvature-weighted mean. `tau_bounds` is documented in `chart.py` as a numerical scan range, so widening it for solver robustness would silently change $K$.
- It is not a centroid: $\int r\,d\tau$ is parametrisation-dependent, so a time-dependent reparametrisation with fixed image rotates the seed and injects spurious $K$.

### 5. Geometric seed plus a relative guard — no tolerance separates

The natural repair is to keep rule 3 and refuse near its locus, guarding on a _relative_ scale, $\rho = |d\mathbf{T}/d\tau(\tau_0)| \,/\, \langle |d\mathbf{T}/d\tau|\rangle_{\tau}$, via `eqx.error_if` (which, unlike a Python branch on a traced value, survives `jit`, `vmap` and `jacfwd` — verified). This does fix rule 3's specific hole: $\rho = 2.5\times10^{-3}$ on the family above, so a relative guard fires where an absolute $\varepsilon = 10^{-8}$ did not.

It fails anyway. On the 3-D near-miss $r(s,t) = (s,\ (t-0.5)s^2 + s^3/3,\ \varepsilon s^2)$, where the seed sweeps ~180° continuously and $|d\mathbf{T}/d\tau| \ge 2\varepsilon$ never vanishes:

| $\varepsilon$ | $\rho$ at $t = 0.5$ | guard fires? | $K$ (geometric) | $K$ (fixed director) | rel. err |
| --- | --- | --- | --- | --- | --- |
| 0.050 | 1.2624e-01 | no | -3.5519 | -0.0381 | 92.2 |
| 0.020 | 5.0909e-02 | no | -9.3777 | -0.0382 | 244.5 |
| 0.010 | 2.5494e-02 | no | -19.0809 | -0.0382 | 498.4 |
| 0.005 | 1.2753e-02 | no | -38.4847 | -0.0382 | 1006.2 |

The product `rel.err` $\times\ \rho$ is `11.6, 12.4, 12.7, 12.9` — constant _for this family, this reference director and these bounds_. So

$$
\text{relative error} \;\approx\; \frac{12.7}{\rho}.
$$

The guard measures how flat $\tau_0$ is; the error scales as the **inverse** of that same quantity.

**The exponent is the result; the coefficient is not.** Independent review reproduced $\rho$ and $K$ to every digit above, but measured the product as **1.88–1.96** with a different reference director and **3.94–4.10** with `tau_bounds` widened to $(0,4)$. So `12.7` is a property of one family, one director and one bounds choice. Do not read it as "$\rho_{\text{tol}} > 12.7$ would work": the coefficient is unbounded over families, so **no finite tolerance works at all** — the stronger statement. A circle sits at $\rho = 1.0$, a helix at $1.0$, a catenary at $2.04$, so any tolerance large enough to catch a bad case rejects ordinary curves. This is not a tuning problem.

### The structural result

**The gauge is not a function of the curve at all.** Take a curve with _no_ $t$-dependence whatsoever — a static helix $(\cos s, \sin s, 0.4s)$ — and spin the director about $\mathbf{T}(\tau_0)$ at $a = \omega\cdot\mathbf{T} = 1/\sqrt{1.16}$. Measured at $(\tau, n_1, n_2) = (0.5, 0.2, 0.1)\,\mathrm{km}$, `tau_bounds` $= (-1,2)\,\mathrm{km}$:

| director    | $K_{\tau\tau}$       |
| ----------- | -------------------- |
| fixed       | **0.000000** exactly |
| spun at $a$ | **−0.067526**        |

Same $r(\tau, t)$, two different correct answers. No function of the curve — however clever, however conditioned — can distinguish them, because the curve is identical in the two cases. That is an information-theoretic obstruction, and unlike the conditioning arguments above it is unconditional: no degeneracy locus, no tolerance, no family of counterexamples required.

The four rejected rules are then evidence of a weaker but useful kind: that the plausible substitutes are _also_ individually broken, so nobody needs to re-derive them.

> An earlier draft generalised this as **"equivariance and smoothness are mutually exclusive for a curve-derived gauge"**. That is **false as written**, and independent review falsified it: the Frenet-normal seed on a rigidly rotating 3-D helix gives $\vert K\vert_{\max} = 5.03\times10^{-17}$ against the material carry's $5.55\times10^{-17}$ — equivariant, smooth in $t$, well conditioned. Its locus $\kappa(\tau_0)=0$ is codimension 2 in 3-D, which is the standard §4 uses to _accept_; §3 condemns it only by choosing a planar family, where the same locus drops to codimension 1. The rule-3 rejection stands on its conditioning table and on the physics, not on that generalisation.

The physical statement: a rod spinning about its own axis and one at rest trace the same curve. Rate of strain of a _tube_ is a property of a **framed** curve — a Cosserat director structure — not of a curve. The library cannot derive it and must not guess.

## Key lemma

In a **Bishop** frame the induced metric is exactly block-diagonal:

$$
\gamma = \mathrm{diag}\!\left(|r'|^2\,(1 - n_1k_1 - n_2k_2)^2,\; 1,\; 1\right),
$$

because $U_i' = -k_i T$ kills the cross terms (unlike Frenet, which carries a torsion term). This is exact, not generic — verified to $\le 2.9\times10^{-11}$ (ODE-solver residual) at offsets up to $|n| = 1.05$, inside and outside the focal distance.

**$\gamma$ is seed-independent iff $|\kappa(\tau)| = 0$ or $n = 0$** — chart-wide, iff the curve is straight. It is _not_ enough that $\kappa\cdot(n_1,n_2) = 0$: rotating the seed by $\psi$ rotates $(k_1,k_2)$ by $-\psi$ while $(n_1,n_2)$ are held fixed, so $n\cdot\kappa$ sweeps a full cosine. At a point where $n\cdot\kappa = 1.4\times10^{-17}$ for one seed, $\gamma_{\tau\tau}$ across five seeds spans `1.0900 … 2.0368` — spread **1.600**.

Two structural corollaries, neither previously stated:

- $\gamma_{n_in_j} = \delta_{ij}$ for **every** builder, so $K_{n_in_j} \equiv 0$ always. The formalism cannot represent radial inflation or cross-sectional shear of the tube — only longitudinal stretch and (Frenet only) $\tau$–$n$ shear. This must be documented; "rate of strain of a tube" invites the opposite expectation.
- $K = \tfrac12\partial_t(J^\mathsf{T}J)$ is symmetric by construction for both builders, so any test asserting $K = K^\mathsf{T}$ is vacuous.

## Design

### `SweptTube`

```python
class SweptTube(eqx.Module):
    """One-parameter family of tubular slices: t -> TubularChart."""

    curve: Any  # two-argument gamma(tau, t)
    director: Callable[[Any], Any] | None = None  # required iff builder is Bishop
    tau_unit: ... = eqx.field(static=True)
    tau_bounds: tuple[Any, Any]  # kw_only
    builder: type = eqx.field(static=True, default=BishopBuilder)
    # solver / n_seed passthrough

    def __call__(self, t) -> TubularChart: ...
```

`director` is **required and has no default.** It is a callable of $t$, not a fixed vector, because a materially spinning rod needs a $t$-dependent frame. Omitting it raises, with a message stating that the gauge is director data the library cannot derive and pointing at the rejected-alternatives section.

> **Day-1 blocker, found by building the sketch.** A `Quantity`-valued director — the natural typing, and what the rest of this library uses — raises `TracerArrayConversionError` inside `rate_of_strain`'s `jacfwd`. `_float` at `bishop.py:132` calls `jnp.asarray(x)`, which invokes `__array__` on a traced `Quantity`, reached via `_transport_start`'s `_orthonormalize(_float(self.initial_normal), T0_val)`. A raw `jnp` array works. One line to fix (`ustrip` first), but it must land before or with this work.

`director` is required **only on the Bishop path**. `FrenetSerretBuilder` takes no seed, so passing one is a caller error, rejected at construction rather than silently ignored — as written earlier ("omitting it raises", plus a selectable builder) the two rules contradicted each other.

`builder` carries the choice because `FrenetSerretBuilder` needs no seed and is _already_ equivariant ($7.6\times10^{-17}$, and exactly `0.0` on axis-aligned rotations) — $\mathbf{N}$ and $\mathbf{B}$ are fixed pointwise by the curve. The gauge bug is Bishop-specific. Frenet is also required for the off-diagonal acceptance test.

### `rate_of_strain` narrows to `SweptTube`

The raw-callable overload is removed. Note what that does and does not buy: `director` is itself an opaque `Callable`, so the gauge is no more _inspectable_ than before. Narrowing buys exactly one thing — **omission raises** rather than silently defaulting, which is criterion 5. The earlier claim that it makes the contract "structural rather than advisory" overstated it. `rate_of_strain` is new in #828 and unreleased.

It must additionally either invoke the reach guard or document that it is unvalidated outside the tube: at present, past the focal distance $\gamma_{\tau\tau} = 1.379\times10^{-3}$ with nothing raising, and the trace identity breaks by two orders _at_ the focal distance ($8.6\times10^{-2}$ vs $7.736$), because `metric_matrix`/`rate_of_strain` never call `TubularChart.check_data(values=True)`.

### `BishopBuilder` requires an explicit seed

`_auto_initial_normal` is removed; a `BishopBuilder` with no `initial_normal` raises. By the lemma, a chart's $(n_1, n_2)$ coordinates genuinely depend on the seed whenever $\kappa \neq 0$ and $n \neq 0$, so an automatic seed makes the coordinates themselves arbitrary and undocumented. This is the same principle as `SweptTube`'s required `director`, one layer down.

> **Scale, measured not estimated.** **156 `BishopBuilder(` construction sites across 28 files**, of which 11 already pass `initial_normal` — so **145 need a seed**. Counted by matching balanced parentheses and excluding this document. (Two earlier drafts got this wrong: "~258 references across 25+ files", which counted _references_ rather than construction sites, then "157 / 38 / 8", from a grep that counted this spec and miscounted files.)
>
> **The stated justification was wrong.** An earlier draft called the auto seed "arbitrary and undocumented". Arbitrary yes; undocumented no — it is described at `bishop.py:38-39` and in `_auto_initial_normal`'s own docstring. The honest case is that the gauge is arbitrary _and load-bearing_ for $(n_1, n_2)$ whenever $\kappa \neq 0$ and $n \neq 0$, so a caller who never chose it still depends on it.
>
> **Independent of #829.** `SweptTube` works either way, so this must not ride in the same PR as the new type and the ten criteria — that PR would be unreviewable and unrevertable. Split out.

## Other defects found

Independent of the gauge work, from an audit of `curveframes`.

### D1. `nearest_tau` returned the wrong nearest point (HIGH) — **shipped**

Two distinct defects, recorded together because the first fix was mistaken for a fix of the second.

**D1a — bisection accepted a maximum.** The bracket test asked only whether the residual changed sign, and it crosses positive-to-negative across a minimum but negative-to-positive across a maximum, so `sign(r_lo) != sign(r_hi)` accepted both. Fixed in #841: require the minimum's orientation, plus a post-check that the answer is no worse than the scan's argmin.

**D1b — the bracket was wide enough to hold two minima.** #841 did _not_ close the original finding, and an earlier draft of this spec asserted that it had. The bracket is `2 * spacing` wide; on the reproducer that is 0.31746 against a wiggle period of 0.31416, so it held minima at 8.34701 (distance 0.107) and 8.46367 (distance 0.011), and bisection returned the first. **Both of #841's guards pass on that answer** — it is a genuine minimum (second derivative +59.2) and it is closer than every scan seed.

It is _not_ a wrong-basin failure, as #847 first claimed and I corrected: the argmin seed is within one spacing of the true minimiser and its basin ranks first of three, so searching additional local-minimum seeds would not have helped. Fixed by refining inside the coarse interval before bracketing: the refined interval is $2\,\text{spacing}/(n_{\text{seed}}-1) = 0.00504$ wide and holds one minimum. The refinement is on the **residual**, not on `dist2` — those coincide only when the tangent is the unit tangent of the parametrisation, and on a worldtube they do not. Every crossing with the minimum's orientation is a candidate and the lowest mean `dist2` wins. Tracked as #847, shipped in #848.

Probe: `x = (8.45268, -0.10722, 0)` km on `(t, 0.3 sin 20t, 0)` over `(0, 10)` s at the default `n_seed = 64`. Result 9.62x → 1.01x at float32; at float64, 8.46367448 against Brent's 8.4636744833. The residual float32 gap is the solver's own `sqrt(eps)` tolerance.

> **Lesson, and it generalises to this whole list.** A guard that inspects the _answer_ cannot detect a correct answer to the wrong question. Both of #841's checks are sound, and both passed on a 9.6x-wrong result.

### D2. `FrenetSerretBuilder` returns an all-NaN frame at an inflection (MEDIUM)

`_normalize` (`frenetserret.py:73-74`) has no zero-norm guard, unlike Bishop's `_orthonormalize` (`bishop.py:161`), and is fed the rejection of $\gamma''$, which vanishes wherever $\kappa = 0$. On $(t, t^3, 0)$ at $\tau = 0$, $\mathbf{N}$ and $\mathbf{B}$ are `[nan, nan, nan]`; on a straight line, everywhere. `check_data` catches it, but the builder accessors (`rotation_matrix`, `normal`, `binormal`, `__call__`, `frame_transition`) do not.

### D3. `nearest_tau`'s tolerances are dimensionally overloaded (LOW–MED)

`nearest.py:139-141` derives one scalar $\sqrt{\epsilon}$ and uses it as a tolerance in $\tau$ (Bisection), on a residual that is a **length** (Newton), and again at `nearest.py:171`. The same geometry converges differently in km and m: error `-3.6e-09` vs `+2.1e-12`, ratio tracking the unit scale exactly. The residual tolerance should be scaled by a length scale.

### D4. `nearest_tau` used `rotation_matrix()[0]` where `tangent()` suffices — **shipped**

Bit-identical value, ~110x faster eagerly. Folded into the D1b fix rather than deferred: the turn-counting probe evaluates the residual on a grid, which was unaffordable at the old cost. Targeted tests went from 70 s to 12 s.

### D5. Zero-width `tau_bounds` hangs forever (HIGH)

`nearest.py:127-128` computes `spacing = (hi-lo)/(n_seed-1)`; with `tau_bounds[0] == tau_bounds[1]` that is `0`, so `bracket_lo == bracket_hi == tau0`. `optx.Bisection(..., expand_if_necessary=True)` (`nearest.py:147-157`) grows a bracket by _doubling its width_ — doubling zero never grows it — and that expansion is **not** bounded by `max_steps=64`.

Reproduced without coordinax, so the curve is ruled out: `lo,hi = 0.0,1.0` returns in 0.40 s and one residual evaluation; `lo,hi = 0.3,0.3` was killed at 120 s having never returned. Through the package, `nearest_tau(..., bounds=(Q(3,'s'), Q(3,'s')))` was killed at 200 s where the same call with `(Q(0,'s'), Q(2*pi,'s'))` returns `tau=1.000000` in 4.2 s.

Publicly reachable: `TubularChart.__check_init__` (`chart.py:140-192`) validates bounds _dimensions_ only, so the chart constructs and every inverse `pt_map` then hangs.

Fix: reject `hi == lo` at `chart.py:178`, or guard `spacing == 0` at `nearest.py:127`.

### D6. `s_max` turns a documented graceful degradation into a hard error (HIGH)

`chart.py:123-126` states that a point whose true nearest curve point lies outside `tau_bounds` "does not raise". With `ArcLength(..., s_max=...)` it does — and `s_max` set exactly as its own docstring instructs (`>= tau_bounds[1]`) is enough to trigger it: `s_max=None` returns `1.200000` (true 1.2); `s_max=Q(1.0,"km")` raises `EquinoxRuntimeError` / `_MSG_S_OUT_OF_DOMAIN`.

Cause: the unconstrained Newton fallback (`nearest.py:163-164`) is **always evaluated**, whichever branch the `jnp.where` at `nearest.py:173` later selects, and it probes arbitrarily far — tripping `_eval_tau_dense`'s `error_if` at `arclength.py:217`. `_S_MAX_MARGIN = 0.05` (`arclength.py:186`) was sized against `nearest_tau`'s _bracket_ slack of one seed spacing; the margin comment does not account for the fallback.

### D7. Closed curves: Bishop holonomy tears the chart at the seam (MEDIUM-HIGH)

On a trefoil over `tau_bounds=(0, 2*pi)`, `|T(0) - T(2*pi)| = 2.94e-16` — the same point with the same tangent — but the Bishop normal rotates by **-2.225041 rad = -127.485 deg** over the period. So `pt_map({tau: 1e-7, n1: 0.2, n2: 0})` and `pt_map({tau: 2*pi - 1e-7, n1: 0.2, n2: 0})` are **0.3587 km apart** while naming the same station and the same normal offset.

`tau_bounds`' docstring (`chart.py:119-121`) warns only about spanning _more_ than one period; it never says the frame fails to close on exactly one. Easy to miss, because a **planar** closed curve has zero holonomy — the obvious probe measures `0.000000 rad` and the seam points coincide.

### D8. Station-pinned one-argument builder is rank-deficient, unrejected (MEDIUM)

`BishopBuilder(circle, "s", station=Q(0.5,"s"))` inside a `TubularChart` builds cleanly: `is_time_dependent` is `False` for a one-argument curve, so `__check_init__`'s worldtube guards do not apply. The forward `pt_map` then **ignores its first coordinate** — `tau=1.0` and `tau=2.5` both give `(0.9653408180793798, 0.5273680924646786, 0.0)` — and `jacobian_factor` is `nan` (0/0 at `chart.py:327`). `check_data(values=True)` catches the NaN, but `pt_map` never calls it, so the map is silently non-injective and the inverse solve degenerate.

### D9. Worldtube reach guard rejects the tube axis with the wrong diagnosis (MEDIUM)

For a rigidly rotating rod (`sigma * (cos t, sin t, 0)`, `station=Q(1.3,'km')`), `jacobian_factor` is `0.0` on the **plane $n_2 = 0$** — including `n1=n2=0` on the axis — but not everywhere: it is `0.07692308` at $n=(0,0.1)$ and $(0.2,0.1)$, and `0.38461538` at $(0.5,0.5)$, and `check_data` passes off that plane. (An earlier draft said every point.) On the singular plane, `check_data` raises "point lies outside the reach of the curve: the tubular coordinates are not locally injective there". Refusing is correct — the station's velocity lies in the normal plane, so `dx/dt` is in `span(U1,U2)` and the determinant really is zero — but the message names a focal/reach failure that did not happen. A transversely-moving worldtube has no `(t,n1,n2)` chart at _any_ offset, and nothing says so at construction. Its longitudinal sibling gives `jacobian_factor = 0.96317125` and passes.

Relatedly, `jacobian_factor`'s docstring claim that it "equals `1-k1*n1-k2*n2` at _any_ parametrisation" is false on this branch: at `n=0` it equals `cos(angle(velocity, spatial tangent))`.

### D11. #841's post-check refused correct worldtube inversions (HIGH) — **shipped**

The "no worse than the scan's argmin" check added in #841 assumes the answer minimises `dist2`. On a station-pinned worldtube the chart inverse is the perpendicular foot, whose distance can exceed the sampled minimum -- the builder's tangent is the curve's _spatial_ tangent while `tau` is a time, so the residual's root and `dist2`'s minimum part company.

Currently harmless: measured 0.000400 against a coarse `d_seed` of 0.000478, so it passes. A worldtube whose foot lies farther than the coarse minimum would be refused wrongly. Found while fixing D1b, where tightening `d_seed` with a finer grid made exactly this fire.

### D10. Reversed `tau_bounds` silently accepted (LOW)

`TubularChart(b, tau_bounds=(Q(2*pi,'s'), Q(0,'s')))` builds, and `nearest_tau` happens to return the right answer (`1.000000`, true 1.0). Undocumented, and `_solve_tau_dense` and `_S_MAX_MARGIN` both assume ascending bounds.

## Acceptance criteria

### Corrections from the second pass

- The reach guard fires at `1 - k1*n1 - k2*n2 = 0` **for Bishop only**. Frenet-Serret's factor is `1 - kappa*n1`, independent of `n2` (`n=(1.5,0)` and `n=(1.5,0.7)` both give `-0.29310345`). Still the correct singular set for that chart, but not the quoted formula — criterion 7 must not assert the Bishop form for both builders.
- **A verification that lands where the truth is zero proves nothing.** The first audit's `_tau_of_s` t-direction check sat at `t=2.0`, where the true derivative is analytically exactly zero; AD and finite differences agreed to 3.4e-11 and confirmed nothing. Re-run at `t=1.0` and `t=3.0` it genuinely matches (7.1e-12, 1.4e-11). Every criterion in the table above must be checked for this: the same defect made four of the original eight vacuous.

The previous eight were audited: **four had provably zero power** over the gauge (#3 was on a straight line where $\kappa = 0$; #4 was an algebraic identity true for any $\gamma(t)$ and passes on the broken implementation; #5's symmetry half is vacuous for both builders; #8 was at $n = 0$ where the gauge cancels). The proposed implementation passed all eight and was wrong. Revised:

| # | Check | Power |
| --- | --- | --- |
| 1 | Rigid rotation, **generic axis**, materially correct director: $\vert K\vert < 10^{-9}$ | The headline |
| 2 | Same rotation, constant director $\perp \mathbf{T}(\tau_0)$: assert $K_{\tau\tau}$ against $\vert r'\vert^2(1-n\cdot k)\,a\,(n\times k)_z$ with $a = \boldsymbol\omega\cdot\mathbf{T}(\tau_0)$. **Both conditions are load-bearing**: $\mathbf{T}(\tau)$ instead of $\mathbf{T}(\tau_0)$ gives 0.814815 against 0.928477, a 14% error, and a non-perpendicular director breaks the identity (0.104803 vs 0.108417). State the configuration alongside the expected number — an earlier draft quoted 0.108856 with none, so it was unreproducible | Pins that the caller's gauge is reported faithfully. **Not** "$\neq 0$", which passes on 1e-14 solver noise — the exact vacuity this table replaced |
| 3 | Breathing circle $R(t) = 1 + t/2$ **in the angle parametrisation** $r = R(t)(\cos\tau, \sin\tau)$: $K_{\tau\tau} = (R + n_1)\dot R$ at $n_1 = 0, \pm0.3$ — measured 0.500000 / 0.650000 / 0.350000 — director the outward radial at $\tau_0$. The formula is **wrong under arc length**, where the same criterion yields 0.000000 / −0.195 / +0.105 | Analytic. **Narrower than it looks**: $T(\tau_0)$ does not rotate in $t$, so the status-quo auto seed is constant and passes this. It discriminates against a rotated director, not against the seed-drift bug |
| 4 | Frenet off-diagonals against closed form $\gamma_{\tau n_1} = -\lVert\gamma'\rVert\sigma n_2$, $\gamma_{\tau n_2} = +\lVert\gamma'\rVert\sigma n_1$ | Reference values; "$\neq 0$" passes on a sign error |
| 5 | Missing `director` raises **on the Bishop path**; supplying one with `builder=FrenetSerretBuilder` also raises | The design's core contract. A dataclass field cannot be "required iff builder is Bishop", so `SweptTube` needs its own conditional check — `FrenetSerretBuilder` rejects `initial_normal` with a `TypeError` one layer down, which is a different error and arrives too late |
| 6 | `jit` and `vmap` over scalar `t` | Killed the earlier guard rule |
| 7 | Past the focal distance it **raises** — the "or documented unvalidated" escape is deleted, since an implementation passed either way. Assert the singular set per builder — Bishop `1-k1n1-k2n2=0`, Frenet `1-kappa*n1=0` (independent of `n2`) | D-series gap |
| 8 | $K_{n_in_j} \equiv 0$ pinned as a known structural limit | Documents rather than tests |
| 9 | Uniform stretch of a straight line, $K_{\tau\tau} = c(1+ct)$ | Plumbing only — no gauge power; keep, labelled as such |
| 10 | $\gamma^{ij}K_{ij} = \partial_t\ln\sqrt{\det\gamma}$ | Plumbing only — labelled as such |
| 11 | **Static curve, spinning director.** Helix $(\cos s, \sin s, 0.4s)$ with no $t$-dependence, `tau_bounds` $=(-1,2)$ km, at $(\tau,n_1,n_2)=(0.5,0.2,0.1)$ km, director spun about $\mathbf{T}(\tau_0)$ at $a = 1/\sqrt{1.16}$: assert $K_{\tau\tau} = -0.067526$, against **exactly 0.0** for a fixed director | **The criterion the design exists for.** No curve-derived rule can pass it — the curve is identical in both cases. Without it the design rests on criterion 1 alone, a zero-truth $\vert K\vert < 10^{-9}$ check |

## Sequencing

Revised — several of these have shipped since the first draft.

1. ~~**PR 1 — D1 alone.**~~ **Done.** D1a shipped as #841; D1b, which #841 did not fix, is #847, and carries D4 with it.
2. **#844 — D5 and D6.** Open. The zero-width hang (code) and the `s_max` documentation.
3. **#845 / #846 — unrelated core defects** found by the same audit: `angle_between` collapsing to zero at float32, and the prolate metric NaN on the `nu = 0` plane.
4. **Next — D2, D3, D7–D10.** The remaining guards, diagnoses and docs.
5. **Then — cleanup.** Fix `$$K*{ij}$$` / `\partial*t\gamma*{ij}` at `packages/coordinaxs.curveframes/docs/curve-charts.md:340`. **Root cause proven:** the repo's own `prettier-markdown-no-wrap` hook, pinned at `v3.8.1` with `--prose-wrap=never`, rewrites `_` to `*` inside a single-line `$$…$$`; reproduced byte-for-byte from clean input, and prettier 3 latest does not do it. **Verified fix:** put the `$$` delimiters on their own lines. Also: state the sign convention; `strain.py` `.matrix.value` -> `ustrip(unit0)` and the dead `t.value` branch.
6. **Then — the `BishopBuilder` breaking change, on its own.** ~149 construction sites. Split out deliberately: it is independent of #829, and bundling it with the new type would make that PR unreviewable and unrevertable.
7. **Then — `SweptTube`.** The type, the required director, and criteria 1–10. Closes #829.
8. **Finally — documentation.** $K_{ij} = \tfrac12(\mathcal{L}_T\gamma)_{ij}$ with $T$ the tube's declared time flow. Say **"by analogy with ADM"**, not "the lapse is 1": Galilean spacetime has no non-degenerate 4-metric, so lapse and shift are not defined. State that $K$ is a slice 2-tensor only under _time-independent_ spatial relabelling, that rate of strain of a tube is a property of a framed curve, and that $K_{n_in_j} \equiv 0$.

> **Line numbers in the D-series are 2–5 low** relative to `main` after #841 and #847. Re-anchor each before writing its PR rather than trusting the number here.

## Behaviour regressions to declare

1. `BishopBuilder` without `initial_normal` raises (145 construction sites need a seed; see the scale note above).
2. `rate_of_strain` no longer accepts a raw callable.
3. `SweptTube` requires a `director`; there is no automatic gauge.
4. `nearest_tau` may route to the fallback (or raise) where it previously returned a confidently wrong maximum.
5. `TubularChart` rejects zero-width (and, if we choose, reversed) `tau_bounds` that it previously accepted before hanging.

## Sign convention

$K_{ij} = +\tfrac{1}{2\alpha}(\partial_t\gamma_{ij} - \mathcal{L}_\beta\gamma_{ij})$ is Wald's sign ($K_{ab} = \nabla_a n_b$). Baumgarte--Shapiro (2.134) and Alcubierre (2.3.11) both write $\partial_t\gamma_{ij} = -2\alpha K_{ij} + D_i\beta_j + D_j\beta_i$, the opposite sign, following MTW. For a function named `rate_of_strain` the positive sign is right, but the convention must be stated or anyone assembling evolution equations from a numerical-relativity text inherits a sign error.

> **Verify before shipping.** This attribution was made from memory of the standard texts and has not been checked against the books. PR 5 must confirm it against the actual editions.

## Corrections to earlier drafts of this spec

Recorded so they are not reintroduced:

- "$\gamma$ is seed-independent iff $\kappa\cdot(n_1,n_2) = 0$" — only "if"; the true condition is $|\kappa| = 0$ or $n = 0$.
- "the centroid seed degenerates exactly on {straight} $\cup$ {odd-symmetric}" — false; the locus is a codimension-1 vector condition.
- "harmless because `tau_0` defaults to the lower bound" — `tau_0` defaults to `Q(0.0, tau_unit)` (`bishop.py:400`), _not_ `tau_bounds[0]`; on bounds like $(-3,3)$ it is the centre.
- "rotations about $\hat x$ or $\hat z$ are exactly equivariant" — $\hat z$ gives $2.0\times10^{-2}$.
- Torsion-period arithmetic: $21.8624 \to 21.8654$; and the rate is the torsion **per arc length** times $|r'|$, $0.275229\sqrt{1.09} = 0.287348\ \mathrm{rad/km}$.
- "the seed must be passed unnormalised so `error_if` fires" — not a requirement; both the unnormalised zero and `normalize(0) = nan` raise, since `~(nan > tol)` is True.
- Guard comparisons must use `~(x > tol)`, never `x <= tol` or `x < tol`: NaN is False for both, so a NaN would pass. `bishop.py:161` documents this; an earlier draft of the guard above walked into it anyway.

## Second-pass verdicts on the first audit

- `jacobian_factor` equals `det(J)/||gamma'||` at six points including `n2 != 0`, mixed signs, and past the focal distance — ratio `1.000000` for both builders.
- AD through `nearest_tau`'s implicit differentiation is correct and **extends to second order**: against a `scipy.brentq` reference, `dtau*/da = -0.4290480985` (fwd and rev) vs `-0.4290481925`, and `d2tau*/da2 = -0.9999608376` vs `-0.99995952`. A naive central difference of the second derivative returns `-1.703`, which is amplified solver noise, not a defect.
- `rate_of_strain` matches a central difference of the metric to 1.27e-11.
- `rotation_matrices` batching agrees with per-tau to <= 2.6e-11 across sorted, unsorted, duplicated, batch-of-one and all-negative batches; `vmap`/`jit` agree with eager.
- `ArcLength` and `AtTime` composed in either order agree to **0.0** once `t` is bound, and their `d/dt` to 1.4e-11 — corroborating #828's own measurement that the ordering hazard is not real for this quantity. `LagrangianArcLength` is the reading that genuinely differs.
