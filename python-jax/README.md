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
