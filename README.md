# Reservoir Computing

Reservoir-computing experiments for chaotic-system forecasting and
classification. The Julia code lives in `Reservoir-Computing-in-Julia/` and
shares utilities through `common.jl` (Lorenz data, reservoir helpers, the valid-
prediction-time metric, and the forecast / Lorenz-map plots).

The GPU-accelerated Python/JAX port of the FHN example is in `python-jax/`.
It is packaged with `uv` and targets the two NVIDIA GPUs on `thl06710`; see
`python-jax/README.md` for setup and run commands.

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

## NARMA-10 memory benchmark

`Reservoir-Computing-in-Julia/RC_NARMA10.jl` runs the standard NARMA-10
reservoir benchmark — predicting a 10-lag nonlinear autoregressive target from
its input with a *linear* readout, so all memory and nonlinearity must come from
the reservoir. It compares the discrete ESN, the continuous-time ESN, and the
FHN oscillator network, reporting NRMSE (lower is better) over seeds.

    julia --project=. Reservoir-Computing-in-Julia/RC_NARMA10.jl [seed] [quick] [nofigs]

Result (8 seeds, mean ± std), in `narma10_results.md`:

| Reservoir | NRMSE |
|---|---|
| discrete ESN | 0.262 ± 0.026 |
| continuous ESN | 0.510 ± 0.015 |
| FHN oscillators | 0.988 ± 0.031 |

The discrete ESN hits the canonical NARMA-10 range (~0.2–0.4). NARMA-10 rewards
precise discrete-step *linear* memory, which the continuous reservoirs blur and
the excitable FHN network handles poorly — a useful characterization showing the
FHN reservoir's strengths lie in other temporal tasks (rich nonlinear transients,
spiking / event-driven inputs), not exact-lag memory.

## SHD spiking-digit classification

`Reservoir-Computing-in-Julia/RC_SHD.jl` classifies the **Spiking Heidelberg
Digits** — spoken digits 0–9 (English + German, 20 classes) encoded as spikes
over 700 cochlear channels: a genuinely temporal, spiking task. Spikes are
pre-binned to a dense `(channels × time)` matrix by `prepare_shd.py`; each sample
is driven through a reservoir over time and a linear ridge readout on temporal-
snapshot features classifies the 20 digits.

One-time data prep (downloads ~170 MB from the Zenke lab; needs Python + h5py):

    # download shd_train.h5.gz / shd_test.h5.gz into data/shd/, gunzip, then:
    python Reservoir-Computing-in-Julia/prepare_shd.py --pool 5 --tbins 100 --per-class 300
    julia --project=. Reservoir-Computing-in-Julia/RC_SHD.jl [seed] [quick] [nofigs] [nofhn]

Result (tuned; 3000 train / 2000 test, 20 classes, chance = 0.05), in
`shd_results.md`:

| Method | Test accuracy |
|---|---|
| raw input (no reservoir) | 0.570 |
| discrete ESN | **0.726** |
| FHN oscillators | **0.721** |

This is the task where the reservoir earns its keep: on a real **temporal
spiking** benchmark both reservoirs clearly beat the raw binned input, and the
FHN oscillator network is on par with the standard ESN — the opposite of the
static-digit and NARMA-10 results, where the reservoir added nothing. It
supports the conclusion that the FHN reservoir's advantage is temporal /
event-driven processing.

Tuning matters a lot here: the key lever is **long memory** over the ~1 s
utterance — a low leak (0.05) for the ESN and slow dynamics (`speed`=0.5) for
the FHN — together with a larger reservoir (NR=500), finer 140-channel binning,
8 temporal-snapshot features, and strong ridge (β=1000). This lifted the FHN
from 0.55 to 0.72 and the ESN from 0.52 to 0.73. (Absolute accuracy is still
below SOTA end-to-end SNNs at ~80–90%, but this is an *untrained* reservoir with
a one-line linear readout.)
