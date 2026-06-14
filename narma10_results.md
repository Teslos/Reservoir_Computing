# NARMA-10 reservoir benchmark

NRMSE (mean ± std over 8 seeds), NR=200, train=3000, test=1000, linear ridge readout.

| Reservoir | NRMSE |
|---|---|
| discrete ESN | 0.262 ± 0.026 |
| continuous ESN | 0.510 ± 0.015 |
| FHN oscillators | 0.988 ± 0.031 |

Reference: a well-tuned ESN reaches NRMSE ~0.2–0.4 on NARMA-10 (NRMSE ≥ 1 means no better than predicting the mean).

NARMA-10 rewards precise discrete-step *linear* memory (10 lags) with only mild nonlinearity, which favors the discrete ESN. The continuous reservoirs low-pass the input and lose exact lag memory; the excitable FHN network in particular is poorly suited to this task — its strengths (rich nonlinear transients, spiking/event-driven inputs) lie elsewhere.
