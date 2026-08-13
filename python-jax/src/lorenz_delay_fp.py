"""Delay-embedded Lorenz model with explicit fixed-point/stability constraints."""

from __future__ import annotations

import argparse
import functools
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jaxopt import LBFGS
from jax.scipy.linalg import expm

jax.config.update("jax_enable_x64", True)
LYAPUNOV = 0.9056


def lorenz_rhs(x):
    return jnp.stack((10.0 * (x[..., 1] - x[..., 0]),
                      x[..., 0] * (28.0 - x[..., 2]) - x[..., 1],
                      x[..., 0] * x[..., 1] - (8.0 / 3.0) * x[..., 2]), axis=-1)


def rk4(x, dt):
    k1 = lorenz_rhs(x)
    k2 = lorenz_rhs(x + dt * k1 / 2)
    k3 = lorenz_rhs(x + dt * k2 / 2)
    k4 = lorenz_rhs(x + dt * k3)
    return x + dt * (k1 + 2*k2 + 2*k3 + k4) / 6


@functools.partial(jax.jit, static_argnums=(0,))
def generate(n, dt=0.01):
    def step(x, _):
        y = rk4(x, dt)
        return y, y
    return jax.lax.scan(step, jnp.array((10.0, 10.0, 10.0)), None, n)[1]


def delay_vectors(x, delays, stride):
    offset = (delays - 1) * stride
    ids = jnp.arange(offset, len(x))[:, None] - stride * jnp.arange(delays)[None, :]
    return x[ids].reshape(len(x) - offset, 3 * delays), offset


def init_model(key, input_dim, lift_dim, hidden):
    kp, k1, k2 = jax.random.split(key, 3)
    # Frozen random nonlinear projection; sqrt scaling avoids immediate saturation.
    projection = jax.random.normal(kp, (input_dim, lift_dim), dtype=jnp.float32) / jnp.sqrt(input_dim)
    params = (jax.random.normal(k1, (lift_dim + 3, hidden), dtype=jnp.float32) / jnp.sqrt(lift_dim + 3),
              jnp.zeros(hidden, jnp.float32),
              jax.random.normal(k2, (hidden, 3), dtype=jnp.float32) / jnp.sqrt(hidden),
              jnp.zeros(3, jnp.float32))
    return projection, params


def predict(params, projection, delay_flat):
    w1, b1, w2, b2 = params
    current = delay_flat[..., :3]
    lift = jnp.tanh(delay_flat @ projection)
    return jnp.tanh(jnp.concatenate((lift, current), axis=-1) @ w1 + b1) @ w2 + b2


def true_fixed_points(mu, sd):
    q = np.sqrt((8.0 / 3.0) * 27.0)
    raw = jnp.array(((0.0, 0.0, 0.0), (q, q, 27.0), (-q, -q, 27.0)))
    return (raw - mu) / sd


def standardized_jacobians(fixed, mu, sd):
    # z' = D^-1 f(mu + Dz), hence J_z = D^-1 J_x D.
    def rhs_z(z):
        return lorenz_rhs(mu + sd * z) / sd
    return jax.vmap(jax.jacfwd(rhs_z))(fixed)


def physical_tangent_maps(target_jac, delays, stride, dt):
    """Exact discrete flow and physically consistent delay tangent embeddings.

    A perturbation q at the current time has lagged copies A^(-k*stride) q,
    where A=exp(dt*J) is the true local Lorenz flow. Thus E maps a current
    three-dimensional perturbation into the full delay coordinates.
    """
    lag_steps = jnp.arange(delays) * stride

    def build(jac):
        advance = expm(dt * jac)
        blocks = jax.vmap(lambda lag: expm(-dt * lag * jac))(lag_steps)
        embedding = blocks.reshape(3 * delays, 3)
        # Backward evolution along forward-stable modes can grow enormously
        # over a long delay window. Normalize each tangent basis column while
        # scaling its target identically; this preserves the linear constraint.
        scale = jnp.linalg.norm(embedding, axis=0)
        return advance / scale[None, :], embedding / scale[None, :]

    return jax.vmap(build)(target_jac)


def losses(params, projection, delays_x, deriv_y, local_x, local_y, fixed,
           target_jac, tangent_advance, tangent_embedding, fp_weight,
           local_weight, tangent_weight, jac_weight, l2_weight):
    pred = predict(params, projection, delays_x)
    data_loss = jnp.mean((pred - deriv_y) ** 2)

    def equilibrium_field(z):
        repeated = jnp.tile(z, delays_x.shape[1] // 3)
        return predict(params, projection, repeated)

    fp_pred = jax.vmap(equilibrium_field)(fixed)
    fp_loss = jnp.mean(fp_pred**2)
    local_loss = jnp.mean((predict(params, projection, local_x) - local_y) ** 2)
    learned_jac = jax.vmap(jax.jacfwd(equilibrium_field))(fixed)
    jac_loss = jnp.mean((learned_jac - target_jac) ** 2)

    def tangent_residual(z, advance, embedding):
        repeated = jnp.tile(z, delays_x.shape[1] // 3)
        # The autonomous update is x+ = x + dt*g(delay). Its derivative along
        # a physically consistent delay perturbation E q must equal A q.
        dg_ddelay = jax.jacfwd(lambda d: predict(params, projection, d))(repeated)
        current_perturbation = embedding[:3, :]
        learned_advance = current_perturbation + 0.01 * (dg_ddelay @ embedding)
        return learned_advance - advance

    tangent_error = jax.vmap(tangent_residual)(fixed, tangent_advance, tangent_embedding)
    tangent_loss = jnp.mean(tangent_error ** 2)
    l2 = sum(jnp.mean(p**2) for p in params)
    total = (data_loss + fp_weight*fp_loss + local_weight*local_loss
             + tangent_weight*tangent_loss + jac_weight*jac_loss + l2_weight*l2)
    return total, (data_loss, fp_loss, local_loss, tangent_loss, jac_loss)


def local_equilibrium_data(key, fixed, mu, sd, delays, stride,
                           perturbations, local_steps, radius_min, radius_max, dt):
    """True-flow delay histories starting close to each Lorenz equilibrium."""
    if not (0 < radius_min <= radius_max):
        raise ValueError("local perturbation radii must satisfy 0 < min <= max")
    count = len(fixed) * perturbations
    kd, kr = jax.random.split(key)
    directions = jax.random.normal(kd, (count, 3))
    directions /= jnp.linalg.norm(directions, axis=1, keepdims=True)
    log_r = jax.random.uniform(kr, (count,), minval=jnp.log(radius_min), maxval=jnp.log(radius_max))
    centers = jnp.repeat(fixed, perturbations, axis=0)
    initial = centers + jnp.exp(log_r)[:, None] * directions
    offset = (delays - 1) * stride

    def rhs_z(z):
        return lorenz_rhs(mu + sd*z) / sd

    def rk4_z(z):
        k1 = rhs_z(z); k2 = rhs_z(z + dt*k1/2)
        k3 = rhs_z(z + dt*k2/2); k4 = rhs_z(z + dt*k3)
        return z + dt*(k1 + 2*k2 + 2*k3 + k4)/6

    def integrate(z0):
        def step(z, _):
            zn = rk4_z(z); return zn, zn
        tail = jax.lax.scan(step, z0, None, offset + local_steps)[1]
        return jnp.concatenate((z0[None], tail), axis=0)

    trajectories = jax.vmap(integrate)(initial)
    lag_ids = jnp.arange(delays)*stride

    def examples(traj):
        current_ids = offset + jnp.arange(local_steps)
        ids = current_ids[:, None] - lag_ids[None, :]
        dx = traj[ids].reshape(local_steps, 3*delays)
        return dx, jax.vmap(rhs_z)(traj[current_ids])

    lx, ly = jax.vmap(examples)(trajectories)
    return lx.reshape(-1, 3*delays).astype(jnp.float32), ly.reshape(-1, 3).astype(jnp.float32)


def adam_train(key, params, projection, x, y, local_x, local_y, fixed,
               target_jac, tangent_advance, tangent_embedding, args):
    zeros = jax.tree.map(jnp.zeros_like, params)
    state = (zeros, zeros, jnp.array(0))

    @jax.jit
    def update(p, opt, xb, yb):
        fn = lambda pp: losses(
            pp, projection, xb, yb, local_x, local_y, fixed, target_jac,
            tangent_advance, tangent_embedding, args.fp_weight, args.local_weight,
            args.tangent_weight, args.jac_weight, args.l2)[0]
        value, grad = jax.value_and_grad(fn)(p)
        m, v, t = opt; t += 1
        m = jax.tree.map(lambda a, g: .9*a + .1*g, m, grad)
        v = jax.tree.map(lambda a, g: .999*a + .001*g*g, v, grad)
        mh = jax.tree.map(lambda a: a/(1-.9**t), m)
        vh = jax.tree.map(lambda a: a/(1-.999**t), v)
        p = jax.tree.map(lambda a, mm, vv: a-args.lr*mm/(jnp.sqrt(vv)+1e-8), p, mh, vh)
        return p, (m, v, t), value

    nb = int(np.ceil(len(x) / args.batch_size))
    for epoch in range(args.adam_epochs):
        key, sub = jax.random.split(key)
        order = jnp.resize(jax.random.permutation(sub, len(x)), nb*args.batch_size).reshape(nb, args.batch_size)
        vals = []
        for ids in order:
            params, state, value = update(params, state, x[ids], y[ids]); vals.append(value)
        if (epoch+1) % 25 == 0 or epoch+1 == args.adam_epochs:
            print(f"Adam epoch {epoch+1}: objective={float(jnp.mean(jnp.stack(vals))):.6g}")
    return params


def valid_time(truth, pred, dt):
    scale = jnp.sqrt(jnp.mean(jnp.sum((truth-truth.mean(0))**2, axis=1)))
    error = jnp.sqrt(jnp.sum((truth-pred)**2, axis=1))/scale
    bad = np.flatnonzero(np.asarray(error) > .4)
    seconds = (bad[0] if len(bad) else len(truth)-1)*dt
    return seconds, seconds*LYAPUNOV


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("seed", nargs="?", type=int, default=42)
    p.add_argument("--delays", type=int, default=16)
    p.add_argument("--delay-stride", type=int, default=5, help="samples between delays; dt=0.01 s")
    p.add_argument("--lift", type=int, default=512)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--adam-epochs", type=int, default=150)
    p.add_argument("--lbfgs-iterations", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--fp-weight", type=float, default=.1)
    p.add_argument("--local-weight", type=float, default=.05)
    p.add_argument("--tangent-weight", type=float, default=.01,
                   help="physical delay-manifold Jacobian weight")
    p.add_argument("--local-perturbations", type=int, default=32, help="true local trajectories per equilibrium")
    p.add_argument("--local-steps", type=int, default=25, help="supervised samples per local trajectory")
    p.add_argument("--local-radius-min", type=float, default=1e-6, help="minimum standardized perturbation")
    p.add_argument("--local-radius-max", type=float, default=1e-3, help="maximum standardized perturbation")
    p.add_argument("--jac-weight", type=float, default=0.0,
                   help="legacy repeated-delay Jacobian loss; local trajectories are preferred")
    p.add_argument("--l2", type=float, default=1e-6)
    p.add_argument("--train-seconds", type=float, default=200.0)
    p.add_argument("--test-seconds", type=float, default=25.0)
    p.add_argument("--no-lbfgs", action="store_true")
    p.add_argument("--no-figs", action="store_true")
    p.add_argument("--quick", action="store_true")
    p.add_argument("--require-gpu", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    print(f"JAX {jax.__version__}; devices: {jax.devices()}; backend: {jax.default_backend()}")
    if args.require_gpu and jax.default_backend() != "gpu":
        raise RuntimeError("GPU required but CUDA was not detected")
    if args.quick:
        args.train_seconds, args.test_seconds = 10.0, 2.0
        args.lift, args.hidden = min(args.lift, 64), min(args.hidden, 32)
        args.adam_epochs, args.lbfgs_iterations = min(args.adam_epochs, 3), min(args.lbfgs_iterations, 2)
    dt = .01; transient = 20.0
    ntr, nte, n0 = round(args.train_seconds/dt)+1, round(args.test_seconds/dt), round(transient/dt)
    trajectory = generate(n0+ntr+nte)
    train = trajectory[n0-1:n0-1+ntr]; test = trajectory[n0-1+ntr:]
    mu, sd = train.mean(0), train.std(0, ddof=1)
    train_z, test_z = (train-mu)/sd, (test-mu)/sd
    delay_x, offset = delay_vectors(train_z, args.delays, args.delay_stride)
    # Central-difference target in standardized units per second.
    delay_x = delay_x[1:-1]
    deriv_y = (train_z[offset+2:] - train_z[offset:-2])/(2*dt)
    fixed = true_fixed_points(mu, sd)
    target_jac = standardized_jacobians(fixed, mu, sd)
    tangent_advance, tangent_embedding = physical_tangent_maps(
        target_jac, args.delays, args.delay_stride, dt)
    local_x, local_y = local_equilibrium_data(
        jax.random.PRNGKey(args.seed+2), fixed, mu, sd, args.delays, args.delay_stride,
        args.local_perturbations, args.local_steps, args.local_radius_min,
        args.local_radius_max, dt)
    projection, params = init_model(jax.random.PRNGKey(args.seed), delay_x.shape[1], args.lift, args.hidden)
    print(f"Delay embedding: {args.delays} x 3, stride={args.delay_stride} ({args.delay_stride*dt:g} s), window={(args.delays-1)*args.delay_stride*dt:g} s")
    print(f"Frozen nonlinear lift: {delay_x.shape[1]} -> {args.lift}; readout hidden={args.hidden}")
    print(f"Equilibrium neighborhoods: {args.local_perturbations} trajectories/fixed point, "
          f"{args.local_steps} samples each, radii=[{args.local_radius_min:g}, {args.local_radius_max:g}]")
    print(f"Physical tangent regularizer: weight={args.tangent_weight:g}, exact exp(dt*J) over "
          f"{args.delays} delay blocks")
    params = adam_train(jax.random.PRNGKey(args.seed+1), params, projection,
                        delay_x.astype(jnp.float32), deriv_y.astype(jnp.float32),
                        local_x, local_y, fixed, target_jac,
                        tangent_advance, tangent_embedding, args)

    objective = lambda p: losses(
        p, projection, delay_x.astype(jnp.float32), deriv_y.astype(jnp.float32),
        local_x, local_y, fixed, target_jac, tangent_advance, tangent_embedding,
        args.fp_weight, args.local_weight, args.tangent_weight, args.jac_weight, args.l2)[0]
    if not args.no_lbfgs and args.lbfgs_iterations:
        solver = LBFGS(objective, maxiter=args.lbfgs_iterations, history_size=10,
                       tol=1e-6, linesearch="zoom", jit=True)
        result = solver.run(params); params = result.params
        print(f"L-BFGS: iterations={int(result.state.iter_num)}, objective={float(result.state.value):.6g}, error={float(result.state.error):.3g}")

    total, parts = losses(
        params, projection, delay_x, deriv_y, local_x, local_y, fixed, target_jac,
        tangent_advance, tangent_embedding, args.fp_weight, args.local_weight,
        args.tangent_weight, args.jac_weight, args.l2)
    print(f"Final losses: total={float(total):.6g} data={float(parts[0]):.6g} "
          f"fixed_point={float(parts[1]):.6g} local={float(parts[2]):.6g} "
          f"tangent={float(parts[3]):.6g} legacy_jacobian={float(parts[4]):.6g}")

    history0 = train_z[-((args.delays-1)*args.delay_stride+1):]
    @functools.partial(jax.jit, static_argnums=(2,))
    def rollout(history, p, steps):
        def step(hist, _):
            ids = jnp.arange(args.delays)*args.delay_stride
            d = hist[-1-ids].reshape(-1)
            xn = jnp.clip(hist[-1] + dt*predict(p, projection, d), -5, 5)
            hist = jnp.concatenate((hist[1:], xn[None]), axis=0)
            return hist, xn
        return jax.lax.scan(step, history, None, steps)[1]
    pred_z = rollout(history0.astype(jnp.float32), params, len(test_z))
    pred = pred_z*sd+mu
    seconds, lyap = valid_time(test, pred, dt)
    print(f"Closed-loop valid prediction time: {seconds:.2f} s ({lyap:.3f} Lyapunov times)")
    print(f"RESULT seed={args.seed} delays={args.delays} stride={args.delay_stride} lift={args.lift} hidden={args.hidden} "
          f"fp_weight={args.fp_weight:g} local_weight={args.local_weight:g} "
          f"tangent_weight={args.tangent_weight:g} jac_weight={args.jac_weight:g} "
          f"t_valid_s={seconds:.3f} t_valid_lyap={lyap:.3f} fp_loss={float(parts[1]):.6g} "
          f"local_loss={float(parts[2]):.6g} tangent_loss={float(parts[3]):.6g} "
          f"jac_loss={float(parts[4]):.6g}")

    if not args.no_figs:
        import matplotlib.pyplot as plt
        out = Path(__file__).resolve().parents[2]/"figures"; out.mkdir(exist_ok=True)
        t = np.arange(1, len(test)+1)*dt
        fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
        for k, label in enumerate("xyz"):
            axes[k].plot(t, np.asarray(test[:, k]), "k", label="true")
            axes[k].plot(t, np.asarray(pred[:, k]), "r", label="delay model")
            axes[k].axvline(seconds, color="gray", ls=":"); axes[k].set_ylabel(label)
        axes[0].legend(); axes[-1].set_xlabel("time (s)"); fig.tight_layout()
        path = out/f"lorenz_delay_fp_seed{args.seed}.png"; fig.savefig(path, dpi=160); plt.close(fig)
        print(f"Figure written to {path}")


if __name__ == "__main__":
    main()
