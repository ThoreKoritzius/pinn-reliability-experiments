# Reasoning through the Taylor–Green experiment

## 1. The equations

For density one, no external force and kinematic viscosity ν:

```text
u_t + u u_x + v u_y + p_x - ν(u_xx + u_yy) = 0
v_t + u v_x + v v_y + p_y - ν(v_xx + v_yy) = 0
u_x + v_y = 0
```

We use x,y in [0,2π), periodic boundaries, and t in [0,2].
The default ν=0.1 gives visible but moderate decay.
Our length scale is inverse wavenumber L=1 and velocity scale U=1, hence
Re=UL/ν=10 with that convention. Using box width as L changes that reported Re
by 2π; always state the convention. Lower viscosity does not make this particular
single-mode solution develop turbulence.

## 2. An exact solution we can differentiate by hand

Let a(t)=exp(-2νt). Then

```text
u =  sin(x) cos(y) a(t)
v = -cos(x) sin(y) a(t)
p = ¼ [cos(2x) + cos(2y)] a(t)² + C(t)
```

1. **Incompressibility:** u_x=cos(x)cos(y)a and v_y=-cos(x)cos(y)a.
2. **Diffusion:** Δu=-2u, Δv=-2v; u_t=-2νu and v_t=-2νv.
   The time derivative and viscous term cancel in each momentum equation.
3. **Advection:** u u_x + v u_y = sin(x)cos(x)a² and
   u v_x + v v_y = sin(y)cos(y)a².
4. **Pressure gradient:** p_x=-sin(x)cos(x)a² and
   p_y=-sin(y)cos(y)a². These cancel advection.

Any C(t) gives the same equations, because only spatial pressure gradients enter.
The code chooses C(t)=-a²/2 so p(0,0,t)=0, an arbitrary gauge. When plotting or
measuring pressure accuracy we remove its spatial mean separately at each time.

The exact spatial-mean kinetic energy is

```text
E(t) = mean[(u²+v²)/2] = ¼ exp(-4νt).
```

At t=2 and ν=0.1 the velocity amplitude is exp(-0.4)≈0.6703 and
E≈0.1123. The solution has genuinely changed over the interval, without becoming
too small for meaningful relative error comparisons.

## 3. What makes this a PINN?

The neural network maps coordinates to three field values. Autograd supplies
first derivatives in x,y,t and second derivatives in x,y. The residual is the
left-hand side of each equation above, evaluated on sampled interior coordinates.

Training observes the equations, initial velocity and periodicity. It does **not**
observe exact future velocity or pressure labels. The formula for future values
is reserved for diagnostics (except in the explicitly labelled supervised control).

The initial state is essential. u=v=p=0 has exactly zero momentum and divergence
residual, yet 100% relative velocity error for this requested solution. The code
checks that counterexample. It explains why PDE residual alone is insufficient;
it is not evidence that a complete, well-posed PDE problem lacks a unique solution.

## 4. Read the plots as separate questions

- **Loss history:** which individual constraints remain unsatisfied?
- **Fresh residual versus training residual:** does the network satisfy equations
  between the coordinates on which it was optimized?
- **True error versus residual:** are they improving together, and on what scales?
  Their numerical magnitudes have different meanings and should not be equated.
- **Error by time:** does later-time accuracy degrade even if the initial state fits?
- **Energy curve:** is the predicted decay physically correct in aggregate?
  Correct energy alone does not imply a correct velocity field.
- **Amplitude and shape:** project predicted velocity onto the exact velocity
  field at each time. Ratio one means correct projected amplitude; the orthogonal
  remainder measures shape distortion. Both may contribute to error.
- **Local residual/error scatter:** can a point's residual help rank its error?
  Local mismatch is expected: PDE constraints couple points in space and time.
  A scatter plot is not a calibrated failure detector.
- **Final-time fields:** look for symmetry breaking, phase errors and spatial
  distortion that a scalar average can hide.

## 5. What we can and cannot attribute

| Observation | Supported interpretation | Follow-up intervention |
|---|---|---|
| L-BFGS lowers error from the Adam checkpoint | Earlier error was reducible with unchanged architecture and points | Compare more Adam at matched time/cost before crediting optimizer choice |
| Direct supervised fitting works well | This architecture can express a good approximation; PDE training is harder in this experiment | Check gradients, IC weighting and optimization |
| Audit residual greatly exceeds training residual | An observed collocation generalization gap | More points or fresh-sample training, with the rest fixed |
| Initial error remains large | The requested initial state is inadequately enforced | Change IC weight or enforce IC exactly |
| Projected amplitude is wrong but shape error small | Dominant observed discrepancy is decay/amplitude | Inspect temporal derivatives, ν, horizon and IC fit |
| Pressure differs only by spatially constant offset | Gauge mismatch | Center pressure or compare spatial gradients |

These are diagnostic hypotheses. No one toy run proves that all PINN error comes
from a particular cause. Next ablations should change one factor at a time and
include multiple seeds; do not report many checkpoints as independent runs.

## 6. Natural next experiments

1. Repeat seeds to assess how fragile the baseline is.
2. Change IC penalty from 10 to 1 and 100, with identical points and initialization.
3. Vary collocation count, separating sampling from optimization budget.
4. Extend the time horizon, keeping decay/normalization effects explicit.
5. Compare hard periodic features with a plain coordinate MLP and soft periodic
   value/derivative constraints. This code currently implements only the first.
6. Only after this is understood, add richer initial conditions and a numerical
   reference solver. Taylor–Green has no nonlinear mode interactions to test.

For external references and the theory connecting residuals with solution error,
see [the annotated reading list](READING_LIST.md).
