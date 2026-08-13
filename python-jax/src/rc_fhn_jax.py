"""JAX/GPU port of the Julia FitzHugh-Nagumo Lorenz reservoir example."""

from __future__ import annotations

import argparse
import functools
import math
from pathlib import Path
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

LORENZ_LYAPUNOV = 0.9056


class Standardizer(NamedTuple):
    mean: jax.Array
    std: jax.Array

    def transform(self, x: jax.Array) -> jax.Array:
        return (x - self.mean) / self.std

    def inverse(self, x: jax.Array) -> jax.Array:
        return x * self.std + self.mean


def rk4_step(rhs, state, forcing, dt):
    """RK4 with forcing=(start, midpoint, end), entirely JAX traceable."""
    f0, fm, f1 = forcing
    k1 = rhs(state, f0)
    k2 = rhs(state + 0.5 * dt * k1, fm)
    k3 = rhs(state + 0.5 * dt * k2, fm)
    k4 = rhs(state + dt * k3, f1)
    return state + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6


def lorenz_data(train_seconds: float, test_seconds: float, dt: float):
    sigma, rho, beta = 10.0, 28.0, 8.0 / 3.0

    def rhs(x, _):
        return jnp.array((sigma * (x[1] - x[0]),
                          x[0] * (rho - x[2]) - x[1],
                          x[0] * x[1] - beta * x[2]))

    @functools.partial(jax.jit, static_argnums=(0,))
    def integrate(n):
        def step(x, _):
            xn = rk4_step(rhs, x, (None, None, None), dt)
            return xn, xn
        _, xs = jax.lax.scan(step, jnp.array((10.0, 10.0, 10.0)), None, n)
        return xs

    transient_steps = round(20.0 / dt)
    train_steps = round(train_seconds / dt) + 1
    test_steps = round(test_seconds / dt)
    all_x = integrate(transient_steps + train_steps + test_steps)
    train = all_x[transient_steps - 1:transient_steps - 1 + train_steps]
    test = all_x[transient_steps - 1 + train_steps:]
    return train, test


def build_graph(kind: str, n: int, rng: np.random.Generator) -> np.ndarray:
    adj = np.zeros((n, n), dtype=np.float32)
    if kind == "complete":
        adj[:] = 1
        np.fill_diagonal(adj, 0)
    elif kind == "grid":
        side = math.isqrt(n)
        if side * side != n:
            raise ValueError("grid topology requires --nodes to be a perfect square")
        for i in range(n):
            r, c = divmod(i, side)
            for rr, cc in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
                if 0 <= rr < side and 0 <= cc < side:
                    adj[i, rr * side + cc] = 1
    elif kind == "watts_strogatz":
        # Undirected k=8 ring, then rewire clockwise edges with p=0.25.
        for i in range(n):
            for d in range(1, 5):
                j = (i + d) % n
                if rng.random() < 0.25:
                    choices = np.flatnonzero((adj[i] == 0) & (np.arange(n) != i))
                    j = int(rng.choice(choices))
                adj[i, j] = adj[j, i] = 1
    elif kind == "barabasi_albert":
        m = min(4, n - 1)
        adj[:m+1, :m+1] = 1
        np.fill_diagonal(adj, 0)
        degree = adj.sum(0)
        for new in range(m + 1, n):
            p = degree[:new] / degree[:new].sum()
            targets = rng.choice(new, m, replace=False, p=p)
            adj[new, targets] = adj[targets, new] = 1
            degree = adj.sum(0)
    elif kind == "erdos_renyi":
        upper = rng.random((n, n)) < 0.1
        adj = np.triu(upper, 1).astype(np.float32)
        adj += adj.T
    else:
        raise ValueError(f"unknown topology: {kind}")
    return adj


def weighted_coupling(adj, coupling, rng):
    weights = (0.1 + 0.9 * rng.random(adj.shape)) * adj
    row_sum = weights.sum(1, keepdims=True)
    return np.where(row_sum > 0, weights * coupling / np.maximum(row_sum, 1e-12), 0).astype(np.float32)


def make_fhn_rhs(wc, strength, a_fhn, eps, r0_input, speed):
    n = wc.shape[0]

    def rhs(r, gin):
        u, w = r[:n], r[n:]
        du = wc @ u - strength * u + u - u**3 / 3 - w + gin
        dw = eps * (r0_input * gin + u - a_fhn)
        return speed * jnp.concatenate((du, dw))
    return rhs


def drive_fhn(rhs, inputs, state0, dt):
    mids = 0.5 * (inputs[:-1] + inputs[1:])

    @jax.jit
    def run():
        def step(r, forcing):
            rn = rk4_step(rhs, r, forcing, dt)
            return rn, rn
        _, tail = jax.lax.scan(step, state0, (inputs[:-1], mids, inputs[1:]))
        return jnp.concatenate((state0[None], tail), axis=0)
    return run()


def delay_features(states, delay):
    return jnp.concatenate((states[2*delay:], states[delay:-delay], states[:-2*delay]), axis=1)


def fit_ridge(features, targets, beta):
    # The quadratic feature Gram matrix is poorly conditioned at Julia's small
    # default beta; match Julia's Float64 solve even though dynamics use f32.
    features, targets = features.astype(jnp.float64), targets.astype(jnp.float64)
    mu, sd = features.mean(0), features.std(0) + 1e-8
    fs = (features - mu) / sd
    phi = jnp.concatenate((fs, fs**2), axis=1)
    # Primal solve matches Julia; features are columns there and rows here.
    eye = jnp.eye(phi.shape[1], dtype=phi.dtype)
    weights = jnp.linalg.solve(phi.T @ phi + beta * eye, phi.T @ targets)
    return (mu, sd, weights), jnp.mean((phi @ weights - targets) ** 2)


def adam_init(params):
    zeros = jax.tree.map(jnp.zeros_like, params)
    return zeros, zeros, jnp.array(0)


def fit_nn(key, features, targets, hidden, epochs, batchsize, lr, noise, quad):
    features, targets = features.astype(jnp.float32), targets.astype(jnp.float32)
    if quad:
        features = jnp.concatenate((features, features**2), axis=1)
    k1, k2, key = jax.random.split(key, 3)
    params = (jax.random.normal(k1, (features.shape[1], hidden)) / jnp.sqrt(features.shape[1]),
              jnp.zeros(hidden),
              jax.random.normal(k2, (hidden, targets.shape[1])) / jnp.sqrt(hidden),
              jnp.zeros(targets.shape[1]))

    def predict(p, x):
        w1, b1, w2, b2 = p
        return jnp.tanh(x @ w1 + b1) @ w2 + b2

    @jax.jit
    def update(p, opt, x, y, noise_key, rate):
        def loss_fn(pp):
            xn = x + noise * jax.random.normal(noise_key, x.shape)
            return jnp.mean((predict(pp, xn) - y) ** 2)
        loss, grads = jax.value_and_grad(loss_fn)(p)
        m, v, t = opt
        t = t + 1
        m = jax.tree.map(lambda a, g: 0.9*a + 0.1*g, m, grads)
        v = jax.tree.map(lambda a, g: 0.999*a + 0.001*g*g, v, grads)
        mh = jax.tree.map(lambda a: a / (1 - 0.9**t), m)
        vh = jax.tree.map(lambda a: a / (1 - 0.999**t), v)
        p = jax.tree.map(lambda a, mm, vv: a - rate*mm/(jnp.sqrt(vv)+1e-8), p, mh, vh)
        return p, (m, v, t), loss

    opt = adam_init(params)
    n_batches = math.ceil(features.shape[0] / batchsize)
    for epoch in range(epochs):
        key, kp, kn = jax.random.split(key, 3)
        order = jax.random.permutation(kp, features.shape[0])
        # Fixed-size batches avoid recompilation; the last wraps around.
        order = jnp.resize(order, n_batches * batchsize).reshape(n_batches, batchsize)
        losses = []
        for ids in order:
            kn, kb = jax.random.split(kn)
            rate = lr if epoch < 0.75 * epochs else 1e-4
            params, opt, loss = update(params, opt, features[ids], targets[ids], kb, rate)
            losses.append(loss)
        if (epoch + 1) % 50 == 0 or epoch + 1 == epochs:
            print(f"Epoch {epoch+1}, loss: {float(jnp.mean(jnp.stack(losses))):.6g}")

    def readout(x):
        xx = jnp.concatenate((x, x**2), axis=-1) if quad else x
        return predict(params, xx)
    return readout


def valid_prediction_time(truth, pred, dt, threshold=0.4):
    truth, pred = np.asarray(truth), np.asarray(pred)
    scale = np.sqrt(np.mean(np.sum((truth - truth.mean(0))**2, axis=1)))
    err = np.sqrt(np.sum((truth - pred)**2, axis=1)) / scale
    bad = np.flatnonzero(err > threshold)
    seconds = (bad[0] if len(bad) else len(err)-1) * dt
    return seconds, seconds * LORENZ_LYAPUNOV


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("topology", nargs="?", default="erdos_renyi",
                   choices=("erdos_renyi", "complete", "grid", "watts_strogatz", "barabasi_albert"))
    p.add_argument("seed", nargs="?", type=int, default=42)
    p.add_argument("--nodes", type=int, default=256)
    p.add_argument("--readout", choices=("nn", "ridge"), default="nn")
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--epochs", type=int, default=400)
    p.add_argument("--beta", type=float, default=1e-4)
    p.add_argument("--coupling", type=float, default=0.20,
                   help="degree-normalized FHN coupling (default: 0.20, edge-of-chaos candidate)")
    p.add_argument("--sigma-in", type=float, default=1.5)
    p.add_argument("--a-lo", type=float, default=0.98,
                   help="lower FHN threshold bound (default: 0.98, near Hopf boundary)")
    p.add_argument("--a-hi", type=float, default=1.04,
                   help="upper FHN threshold bound (default: 1.04, near Hopf boundary)")
    p.add_argument("--quad", action="store_true")
    p.add_argument("--partial", action="store_true")
    p.add_argument("--no-figs", action="store_true")
    p.add_argument("--quick", action="store_true", help="small smoke-test configuration")
    p.add_argument("--require-gpu", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    if args.coupling < 0:
        raise ValueError("--coupling must be non-negative")
    if args.a_lo > args.a_hi:
        raise ValueError("--a-lo must be no greater than --a-hi")
    devices = jax.devices()
    print(f"JAX {jax.__version__}; devices: {devices}; backend: {jax.default_backend()}")
    if args.require_gpu and jax.default_backend() != "gpu":
        raise RuntimeError("GPU required but JAX did not detect CUDA; run `uv sync` on thl06710 and inspect nvidia-smi")

    dt, delay = 0.01, 10
    train_s, test_s, washout = (10.0, 2.0, 100) if args.quick else (200.0, 25.0, 500)
    if args.quick:
        args.nodes, args.hidden, args.epochs = min(args.nodes, 32), min(args.hidden, 32), min(args.epochs, 5)
    rng = np.random.default_rng(args.seed)
    train, test = lorenz_data(train_s, test_s, dt)
    scaler = Standardizer(train.mean(0), train.std(0, ddof=1))
    u_train, u_test = scaler.transform(train), scaler.transform(test)

    adj = build_graph(args.topology, args.nodes, rng)
    wc = jnp.asarray(weighted_coupling(adj, args.coupling, rng))
    strength = wc.sum(1)
    print(f"Topology: {args.topology}, {args.nodes} nodes, {int(adj.sum())} directed edges, density {adj.mean():.3f}")
    print(f"FHN regime: coupling={args.coupling:g}, a~U({args.a_lo:g}, {args.a_hi:g})")
    in_dim = 1 if args.partial else 3
    win = jnp.asarray(2 * args.sigma_in * (rng.random((args.nodes, in_dim)) - 0.5), dtype=jnp.float32)
    a_fhn = jnp.asarray(rng.uniform(args.a_lo, args.a_hi, args.nodes), dtype=jnp.float32)
    rhs = make_fhn_rhs(wc, strength, a_fhn, 0.05, 0.5, 20.0)
    observed_train = u_train[:, :1] if args.partial else u_train
    train_inputs = (observed_train @ win.T).astype(jnp.float32)
    r_train = drive_fhn(rhs, train_inputs, jnp.zeros(2*args.nodes, dtype=jnp.float32), dt)
    all_features = delay_features(r_train, delay)
    start = washout - 2*delay
    features = all_features[start:]
    targets = u_train[washout:]

    key = jax.random.PRNGKey(args.seed)
    if args.readout == "ridge":
        ridge, mse = fit_ridge(features, targets, args.beta)
        mu, sd, weights = ridge
        def readout(f):
            fs = (f - mu) / sd
            return jnp.concatenate((fs, fs**2), axis=-1) @ weights
        print(f"Ridge readout (beta={args.beta:g}) training MSE: {float(mse):.6g}")
    else:
        readout = fit_nn(key, features, targets, args.hidden, args.epochs, 256, 1e-3, 0.02, args.quad)
        print(f"NN readout training MSE: {float(jnp.mean((readout(features)-targets)**2)):.6g}")

    @functools.partial(jax.jit, static_argnums=(2,))
    def closed_loop(history, x0, n_steps):
        history, x0 = history.astype(jnp.float32), x0.astype(jnp.float32)
        def step(carry, _):
            hist, x = carry
            gin = win @ (x[:1] if args.partial else x)
            rn = rk4_step(rhs, hist[-1], (gin, gin, gin), dt)
            hist = jnp.concatenate((hist[1:], rn[None]), axis=0)
            feat = jnp.concatenate((hist[-1], hist[-1-delay], hist[-1-2*delay]))
            x = jnp.clip(readout(feat), -5, 5).astype(jnp.float32)
            return (hist, x), x
        return jax.lax.scan(step, (history, x0), None, n_steps)[1]

    history = r_train[-(2*delay+1):]
    pred_closed_n = closed_loop(history, u_train[-1], len(test))
    pred_closed = scaler.inverse(pred_closed_n)

    observed_test = u_test[:, :1] if args.partial else u_test
    r_test = drive_fhn(rhs, (observed_test @ win.T).astype(jnp.float32), r_train[-1], dt)
    extended = jnp.concatenate((r_train[-2*delay:], r_test), axis=0)
    pred_open = scaler.inverse(readout(delay_features(extended, delay)))
    # drive_fhn includes the initial state, giving exactly len(test) features.
    pred_open = pred_open[:len(test)]

    t_valid, t_lyap = valid_prediction_time(test, pred_closed, dt)
    n_short = min(len(test), round(1 / (LORENZ_LYAPUNOV * dt)))
    mse_short = float(jnp.mean((test[:n_short] - pred_closed[:n_short])**2))
    mse_open = float(jnp.mean((test - pred_open)**2))
    print(f"Closed-loop valid prediction time: {t_valid:.2f} s ({t_lyap:.2f} Lyapunov times)")
    print(f"Closed-loop MSE over first Lyapunov time: {mse_short:.6g}")
    print(f"Open-loop (teacher-forced) MSE: {mse_open:.6g}")
    flags = f" beta={args.beta:g}" if args.readout == "ridge" else f" hidden={args.hidden} quad={str(args.quad).lower()}"
    print(f"RESULT topology={args.topology} seed={args.seed} readout={args.readout}{flags} "
          f"coupling={args.coupling:g} a_lo={args.a_lo:g} a_hi={args.a_hi:g} "
          f"partial={str(args.partial).lower()} t_valid_s={t_valid:.3f} t_valid_lyap={t_lyap:.3f} mse_open={mse_open:.4f}")

    if not args.no_figs:
        climate_n = closed_loop(history, u_train[-1], round(100/dt))
        make_plots(args, np.asarray(train), np.asarray(test), np.asarray(pred_closed),
                   np.asarray(pred_open), np.asarray(scaler.inverse(climate_n)),
                   np.asarray(r_train), t_valid, dt)


def make_plots(args, train, truth, closed, opened, climate, states, t_valid, dt):
    import matplotlib.pyplot as plt
    out = Path(__file__).resolve().parents[2] / "figures"
    out.mkdir(exist_ok=True)
    suffix = f"{args.topology}_{args.readout}" + ("_quad" if args.quad else "") + ("_partial" if args.partial else "")
    t = np.arange(1, len(truth)+1) * dt
    fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
    for k, label in enumerate("xyz"):
        axes[k].plot(t, truth[:, k], "k", label="True")
        axes[k].plot(t, closed[:, k], "r", label="Closed-loop")
        axes[k].plot(t, opened[:, k], "b--", alpha=.6, label="Open-loop")
        axes[k].axvline(t_valid, color="gray", ls=":"); axes[k].set_ylabel(label)
    axes[0].legend(); axes[-1].set_xlabel("Time (s)")
    fig.tight_layout(); fig.savefig(out / f"lorenz_FHN_NN_{suffix}.png", dpi=160); fig.savefig(out / f"lorenz_FHN_NN_{suffix}.pdf"); plt.close(fig)

    fig = plt.figure(figsize=(8, 6)); ax = fig.add_subplot(projection="3d")
    ax.plot(*truth.T, color="blue", alpha=.7); ax.plot(*closed.T, color="red", alpha=.7)
    ax.set(xlabel="x", ylabel="y", zlabel="z"); fig.tight_layout()
    fig.savefig(out / f"lorenz3d_FHN_NN_{suffix}.png", dpi=160); fig.savefig(out / f"lorenz3d_FHN_NN_{suffix}.pdf"); plt.close(fig)

    def maxima(x):
        z = x[:, 2]; return z[1:-1][(z[1:-1] > z[:-2]) & (z[1:-1] > z[2:])]
    mt, mp = maxima(train), maxima(climate)
    fig, ax = plt.subplots(figsize=(7, 6)); ax.scatter(mt[:-1], mt[1:], s=8, c="k", alpha=.6); ax.scatter(mp[:-1], mp[1:], s=8, c="r", alpha=.6)
    ax.set(xlabel="$z_n$", ylabel="$z_{n+1}$", title="Lorenz map"); fig.tight_layout()
    fig.savefig(out / f"lorenz_map_FHN_NN_{suffix}.png", dpi=160); fig.savefig(out / f"lorenz_map_FHN_NN_{suffix}.pdf"); plt.close(fig)

    n = min(len(states), round(20/dt)); fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(np.arange(n)*dt, states[:n, :min(8, args.nodes)]); ax.set(xlabel="Time (s)", ylabel="u", title="FHN reservoir states")
    fig.tight_layout(); fig.savefig(out / f"RC_FHN_reservoir_states_{args.topology}.png", dpi=160); plt.close(fig)
    print(f"Figures written under {out}")


if __name__ == "__main__":
    main()
