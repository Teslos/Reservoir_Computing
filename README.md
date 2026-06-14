# Reservoir Computing

Reservoir-computing experiments for chaotic-system forecasting and
classification. The Julia code lives in `Reservoir-Computing-in-Julia/` and
shares utilities through `common.jl` (Lorenz data, reservoir helpers, the valid-
prediction-time metric, and the forecast / Lorenz-map plots).

## Lorenz-63 forecasters

| Script | Reservoir |
|---|---|
| `RC.jl` | discrete ESN, ridge readout |
| `RC_NN.jl` | discrete ESN, neural-network readout |
| `RC_FHN_NN.jl` | FitzHugh-Nagumo oscillator network (physical reservoir) |
| `RC_LPCTESN.jl` | **continuous-time** echo state network, linear-projection readout |

### Continuous-time echo state network (LPCTESN)

`Reservoir-Computing-in-Julia/RC_LPCTESN.jl` adds a linear-projection
continuous-time ESN: the reservoir is an ODE
`r' = speed·(−leak·r + tanh(A r + W_in x))` with a ridge linear readout
`x = W_out r`. It teacher-forces the reservoir on the training trajectory, fits
`W_out`, then closes the loop and integrates forward to forecast Lorenz-63.
Reuses `common.jl`; writes `figures/lorenz_LPCTESN.png`,
`figures/lorenz3d_LPCTESN.png`, and `figures/lorenz_map_LPCTESN.png`.

Run from the repo root:

    julia --project=. Reservoir-Computing-in-Julia/RC_LPCTESN.jl [seed] [quick] [nofigs]

Reference: Anantharaman, Ma, Gowda, Laughman, Shah, Edelman, Rackauckas (2021),
"Accelerating Simulation of Stiff Nonlinear Systems using Continuous-Time Echo
State Networks", arXiv:2010.04004 — `docs/2010.04004v6.pdf`.
