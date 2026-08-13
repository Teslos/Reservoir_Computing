# JAX/GPU FHN reservoir

This is a JAX port of `Reservoir-Computing-in-Julia/RC_FHN_NN.jl`. It includes
the five graph topologies, FHN oscillator network, delayed features, neural and
ridge readouts, partial observation, closed-loop and teacher-forced forecasts,
valid-prediction metrics, climate rollout, and the four plots.

## GPU setup on `thl06710`

The project targets the Ubuntu 24.04 host `thl06710`, whose two TITAN RTX GPUs
(SM 7.5) and NVIDIA 595 driver support JAX's CUDA 13 wheels. CUDA-enabled JAX
is a normal project dependency, so no extra flag or system CUDA toolkit is
needed.

```bash
ssh thl06710
curl -LsSf https://astral.sh/uv/install.sh | sh
# Open a new shell if `uv` is not immediately on PATH.
cd /path/to/ReservoirComputing/python-jax
uv sync
uv run rc-fhn-jax --quick --no-figs --require-gpu
```

For a full run:

```bash
cd python-jax
CUDA_VISIBLE_DEVICES=0 uv run rc-fhn-jax erdos_renyi 42
```

The first line of program output lists both detected `CudaDevice`s. Set
`CUDA_VISIBLE_DEVICES=1` to use the second GPU. Remove `--quick` for the full
200 s training / 25 s test run.

Examples mirroring Julia options:

```bash
uv run rc-fhn-jax erdos_renyi 42 --readout nn --quad
uv run rc-fhn-jax watts_strogatz 7 --readout ridge --beta 1e-2
uv run rc-fhn-jax grid 42 --partial --nodes 256
```

Use `--require-gpu` in automated runs to fail instead of silently using the
CPU. Run `uv run rc-fhn-jax --help` for every parameter.

The defaults use an edge-of-chaos candidate regime: degree-normalized coupling
`0.20` and heterogeneous FHN thresholds `a ~ U(0.98, 1.04)`. This keeps the
population close to the Hopf boundary at `|a| = 1` while reducing collective
synchronization. These are testable candidate settings, not a claim that the
conditional Lyapunov exponent is already optimal; override them with
`--coupling`, `--a-lo`, and `--a-hi` for sweeps.

## Delay-lifted fixed-point experiment

`lorenz-delay-fp` builds delay coordinates directly from Lorenz-63, maps them
through a frozen high-dimensional random `tanh` projection, and learns a
residual vector field. Its loss constrains all three Lorenz equilibria and uses
short, physically consistent true-flow delay trajectories around them to teach
their local stable/unstable dynamics. A physical tangent regularizer additionally
uses the exact discrete propagator `exp(dt J)` and consistent lag perturbations
`exp(-k dt J)` to match the three-dimensional Lorenz tangent dynamics inside
the full delay space. Adam is followed by deterministic full-batch L-BFGS.

```bash
uv run lorenz-delay-fp --quick --no-figs --require-gpu
uv run lorenz-delay-fp 42 --delays 16 --delay-stride 5 --lift 512 --require-gpu
```

Ablate the geometry constraints with
`--fp-weight 0 --local-weight 0 --tangent-weight 0`, or
isolate the optimizer effect with `--no-lbfgs`. The old repeated-delay Jacobian
penalty remains available through `--jac-weight`, but defaults to zero because
it is not the full Jacobian of the autonomous delay-buffer map. The
machine-readable `RESULT` line reports forecast Lyapunov time plus fixed-point,
local-neighborhood, and diagnostic Jacobian residuals.

After Adam and L-BFGS, the default training performs a 10-to-100-step
autonomous rollout curriculum. A 25-second tail is excluded from all fitting
and used for multi-start validation, reporting median, lower-quartile, and
worst-case Lyapunov times. Use `--rollout-epochs 0` for the old one-step-only
workflow. `--projection-seed`, `--network-seed`, and `--local-seed` independently
control the frozen lift, readout initialization, and equilibrium trajectories.
