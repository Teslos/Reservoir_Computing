# FHN reservoir + MULTI-STEP ROLLOUT TRAINING for Lorenz-63.
#
# Motivation. Every readout tried so far (observer, next-step, derivative, ridge)
# is fitted on reservoir states visited under TEACHER FORCING, and none of them
# constrains the autonomous dynamics that the forecast actually consists of. The
# symptoms were consistent: nine hyperparameters flat, four readout formulations
# worth at most ~2x, a near-perfect (corr 0.9999) lobe detector that changed
# nothing, and closed-loop return maps that never formed a tent.
#
# This script closes the loop DURING TRAINING: the loss is the error accumulated
# over K autonomous steps, so gradients see the trajectory distribution the model
# will actually be run on.
#
# Implementation note: the closed-loop rollout must be differentiable, so the
# reservoir is stepped with a hand-rolled RK4 on a PURE-FUNCTIONAL right-hand side
# (the production RHS uses mul!/@. in-place, which Zygote cannot handle), and the
# delay buffer is rebuilt by concatenation rather than mutated.
#
# Run:  julia --project=. Reservoir-Computing-in-Julia/RC_FHN_NN_rollout.jl \
#           erdos_renyi 42 nodes=128 K=150 roll_epochs=400 roll_batches=20

include("common.jl")
using .RCCommon
using Graphs, LinearAlgebra, Random, Statistics, OrdinaryDiffEq, Flux, Printf
using Serialization

# --- args ------------------------------------------------------------------------
topology = isempty(ARGS) ? :erdos_renyi : Symbol(ARGS[1])
SEED     = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 42
ival(k, d) = (i = findfirst(a -> startswith(a, k * "="), ARGS);
              i === nothing ? d : parse(Int, ARGS[i][length(k)+2:end]))
fval(k, d) = (i = findfirst(a -> startswith(a, k * "="), ARGS);
              i === nothing ? d : parse(Float64, ARGS[i][length(k)+2:end]))
n_nodes      = ival("nodes", 64)
K_roll       = ival("K", 100)           # final curriculum horizon in steps
roll_epochs  = ival("roll_epochs", 200)
pre_epochs   = ival("pre_epochs", 400)  # one-step pretraining before rollout
batch_roll   = ival("batch_roll", 8)    # rollout start points per gradient step
roll_batches = ival("roll_batches", 10) # gradient steps per rollout epoch
roll_lr      = fval("roll_lr", 3e-4)
n_hidden     = ival("hidden", 256)
final_epochs = ival("final_epochs", 100) # epochs spent at the full horizon
val_steps    = ival("val_steps", 2500)   # held-out tail of teacher-forced train data
n_val_starts = ival("val_starts", 16)
eval_steps   = ival("eval_steps", 500)   # 5 s multi-start evaluation horizon
n_eval_starts = ival("eval_starts", 20)

dim_system = 3; dt = 0.01; washout = 500; delay_steps = 10
sigma_in = 1.5; eps_fhn = 0.05; a_lo, a_hi = 0.95, 1.1
coupling = 0.3; R0 = 0.5; speed = 20.0
noise_level = 0.02f0

rng = MersenneTwister(SEED); Random.seed!(SEED)

# --- reservoir -------------------------------------------------------------------
function build_graph(kind, n, rng)
    g = kind === :erdos_renyi ? erdos_renyi(n, 0.1; rng = rng) :
        kind === :complete ? complete_graph(n) :
        kind === :grid ? Graphs.grid([isqrt(n), isqrt(n)]) :
        kind === :watts_strogatz ? watts_strogatz(n, 8, 0.25; rng = rng) :
        kind === :barabasi_albert ? barabasi_albert(n, 4; rng = rng) :
        error("unknown topology $kind")
    return SimpleDiGraph(g)
end
g = build_graph(topology, n_nodes, rng); N = nv(g)
println("Topology: $topology, $N nodes, $(ne(g)) directed edges")
flush(stdout)

Wc = zeros(N, N)
for e in edges(g); Wc[dst(e), src(e)] = 0.1 + 0.9rand(rng); end
for i in 1:N
    s = sum(@view Wc[i, :]); s > 0 && (Wc[i, :] .*= coupling / s)
end
in_strength = vec(sum(Wc, dims = 2))
a_fhn = a_lo .+ (a_hi - a_lo) .* rand(rng, N)

train_data, t_train, test_data, t_test =
    generate_lorenz_split(t_train = 200.0, t_test = 25.0, dt = dt)
scaler = Standardizer(train_data)
u_train = transform(scaler, train_data)
W_in = 2 * sigma_in .* (rand(rng, N, dim_system) .- 0.5)

# One transition map is used everywhere below.  In particular, teacher forcing
# is zero-order held rather than linearly interpolated through the future target.
# This also removes the previous RK4/Tsit5 train/evaluation mismatch.
function fhn_deriv(u, w, gin)
    du = speed .* (Wc * u .- in_strength .* u .+ u .- u .^ 3 ./ 3 .- w .+ gin)
    dw = speed .* eps_fhn .* (R0 .* gin .+ u .- a_fhn)
    return du, dw
end
function rk4(u, w, gin)
    k1u, k1w = fhn_deriv(u, w, gin)
    k2u, k2w = fhn_deriv(u .+ 0.5dt .* k1u, w .+ 0.5dt .* k1w, gin)
    k3u, k3w = fhn_deriv(u .+ 0.5dt .* k2u, w .+ 0.5dt .* k2w, gin)
    k4u, k4w = fhn_deriv(u .+ dt .* k3u, w .+ dt .* k3w, gin)
    (u .+ (dt / 6) .* (k1u .+ 2k2u .+ 2k3u .+ k4u),
     w .+ (dt / 6) .* (k1w .+ 2k2w .+ 2k3w .+ k4w))
end

n_tr = size(u_train, 1)
train_end = n_tr - val_steps
train_end > washout + 2delay_steps + K_roll ||
    error("val_steps=$val_steps leaves too little rollout-training data")
R_train = zeros(2N, n_tr)
for i in 1:(n_tr - 1)
    un, wn = rk4(@view(R_train[1:N, i]), @view(R_train[(N+1):2N, i]),
                 W_in * u_train[i, :])
    R_train[:, i + 1] = vcat(un, wn)
end

const d_steps = delay_steps
delay_features(S, idx) = vcat(S[:, idx], S[:, idx .- d_steps], S[:, idx .- 2d_steps])

# Features are standardized component-wise.  Raw FHN fast/slow variables have
# very different scales, making both the input noise and rollout gradients badly
# conditioned.
F_stats = delay_features(R_train, (washout + 1):train_end)
f_mu = vec(mean(F_stats, dims = 2))
f_sd = vec(std(F_stats, dims = 2)) .+ 1e-6
standardize_features(f) = (f .- f_mu) ./ f_sd
n_feat = 6N + dim_system

# The readout predicts a derivative/residual and receives x(t) explicitly:
# x(t+dt) = x(t) + dt*f(phi(r(t+dt)), x(t)).  This supplies the identity path
# instead of asking the MLP to reconstruct it indirectly through the reservoir.
model = Chain(Dense(n_feat => n_hidden, tanh), Dense(n_hidden => dim_system))
train_i = washout:(train_end - 1)
F_next = delay_features(R_train, train_i .+ 1)
X_feat = Float32.(vcat(standardize_features(F_next), permutedims(u_train[train_i, :])))
Y_targ = Float32.(permutedims((u_train[train_i .+ 1, :] .- u_train[train_i, :]) ./ dt))
opt_state = Flux.setup(Adam(1e-3), model)
loader = Flux.DataLoader((X_feat, Y_targ); batchsize = 256, shuffle = true)
for epoch in 1:pre_epochs
    for (xb, yb) in loader
        # Perturb only standardized reservoir features, not the exact state
        # passthrough used by the residual update.
        xn = vcat(xb[1:(6N), :] .+ noise_level .* randn(Float32, 6N, size(xb, 2)),
                  xb[(6N + 1):end, :])
        _, gs = Flux.withgradient(m -> Flux.mse(m(xn), yb), model)
        Flux.update!(opt_state, model, gs[1])
    end
end
@printf("stage 1 derivative training MSE: %.6f\n", Flux.mse(model(X_feat), Y_targ))
flush(stdout)

# --- stage 2: multi-step rollout ------------------------------------------------
# Use the identical safety bound in training and evaluation.  A tanh bound was
# considered here, but repeatedly applying 5*tanh(x/5) contracts even ordinary
# |x|~1 states at every 0.01 s step and therefore changes the learned dynamics.
feedback_bound(x) = clamp.(x, -5.0, 5.0)

"Autonomous rollout of K future steps from training index t0; returns mean MSE."
function rollout_loss(m, t0, K)
    hist = [R_train[:, t0 - 2d_steps + j] for j in 0:(2d_steps)]   # 21 states
    u = R_train[1:N, t0]; w = R_train[(N+1):2N, t0]
    x = u_train[t0, :]
    loss = 0.0f0
    for k in 1:K
        # Advance with x(t), then predict x(t+dt) from the resulting state.
        gin = W_in * x
        u, w = rk4(u, w, gin)
        hist = vcat(hist[2:end], [vcat(u, w)])
        feats = vcat(hist[end], hist[end - d_steps], hist[end - 2d_steps])
        deriv = m(Float32.(vcat(standardize_features(feats), x)))
        x = feedback_bound(x .+ dt .* Float64.(deriv))
        loss += Flux.mse(Float32.(x), Float32.(u_train[t0 + k, :]))
    end
    return loss / K
end

opt_roll = Flux.setup(Adam(roll_lr), model)
val_lo, val_hi = train_end, n_tr - K_roll
val_starts = unique(round.(Int, range(val_lo, val_hi; length = n_val_starts)))
checkpoint_path = joinpath("data", "fhn_rollout_best_seed$(SEED)_N$(N)_K$(K_roll).jls")
mkpath(dirname(checkpoint_path))
best_val = Inf
best_epoch = 0
best_state = deepcopy(Flux.state(model))

@printf("stage 2: smooth curriculum to K=%d (%.2f s), %d epochs (%d at full K), %d batches x %d, lr %.0e\n",
        K_roll, K_roll * dt, roll_epochs, final_epochs,
        roll_batches, batch_roll, roll_lr)
@printf("validation: %d fixed starts in held-out final %.1f s; checkpoint %s\n",
        length(val_starts), val_steps * dt, checkpoint_path)
flush(stdout)
for epoch in 1:roll_epochs
    tot = 0.0f0
    # Increase by small 10-step increments, then leave `final_epochs` entirely
    # at K_roll.  This avoids the previous abrupt 25->50->100 optimization shocks.
    ramp_epochs = max(1, roll_epochs - final_epochs)
    ramp_fraction = min(1.0, epoch / ramp_epochs)
    K_now = min(K_roll, max(10, 10 * round(Int, (K_roll * ramp_fraction) / 10)))
    starts = (washout + 2d_steps):(train_end - K_now)
    for _ in 1:roll_batches
        t0s = rand(rng, starts, batch_roll)
        l, gs = Flux.withgradient(m ->
            sum(rollout_loss(m, t0, K_now) for t0 in t0s) / batch_roll, model)
        Flux.update!(opt_roll, model, gs[1])
        tot += l
    end
    if epoch % 10 == 0 || epoch == 1
        train_loss = tot / roll_batches
        if K_now == K_roll
            val_loss = mean(rollout_loss(model, t0, K_roll) for t0 in val_starts)
            if val_loss < best_val
                global best_val = val_loss
                global best_epoch = epoch
                global best_state = deepcopy(Flux.state(model))
                serialize(checkpoint_path, best_state)
            end
            @printf("  rollout epoch %3d  K=%3d  train %.5f  val %.5f  best %.5f@%d\n",
                    epoch, K_now, train_loss, val_loss, best_val, best_epoch)
        else
            @printf("  rollout epoch %3d  K=%3d  train %.5f\n",
                    epoch, K_now, train_loss)
        end
        flush(stdout)
    end
end

best_epoch > 0 || error("curriculum never reached K_roll; increase roll_epochs or reduce final_epochs")
Flux.loadmodel!(model, best_state)
@printf("restored best checkpoint: epoch %d, fixed validation loss %.5f\n",
        best_epoch, best_val)
flush(stdout)

# --- evaluation: same closed loop as the production script -----------------------
function closed_loop_at(m, t0, n_steps)
    pred = zeros(n_steps, dim_system)
    hist = [R_train[:, t0 - 2d_steps + j] for j in 0:(2d_steps)]
    u = copy(R_train[1:N, t0]); w = copy(R_train[(N+1):2N, t0])
    x = copy(u_train[t0, :])
    for i in 1:n_steps
        u, w = rk4(u, w, W_in * x)
        hist = vcat(hist[2:end], [vcat(u, w)])
        feats = vcat(hist[end], hist[end - d_steps], hist[end - 2d_steps])
        deriv = m(Float32.(vcat(standardize_features(feats), x)))
        x = feedback_bound(x .+ dt .* Float64.(deriv))
        pred[i, :] = x
    end
    return pred
end
X_pred = inverse_transform(scaler, closed_loop_at(model, n_tr, length(t_test)))
t_valid, t_valid_lyap = valid_prediction_time(test_data, X_pred, t_test)
@printf("Closed-loop valid prediction time: %.2f s (%.3f Lyapunov times)\n",
        t_valid, t_valid_lyap)

# Report robustness across many unseen forecast boundaries, rather than relying
# on the single train/test boundary.  These starts lie only in the held-out tail.
eval_hi = n_tr - eval_steps
eval_starts = unique(round.(Int, range(train_end, eval_hi; length = n_eval_starts)))
multi_lyap = Float64[]
eval_t = collect(dt:dt:(eval_steps * dt))
for t0 in eval_starts
    pred_n = closed_loop_at(model, t0, eval_steps)
    pred = inverse_transform(scaler, pred_n)
    truth = train_data[(t0 + 1):(t0 + eval_steps), :]
    _, lyap = valid_prediction_time(truth, pred, eval_t)
    push!(multi_lyap, lyap)
end
q25, med, q75 = quantile(multi_lyap, [0.25, 0.5, 0.75])
@printf("Held-out multi-start Lyapunov times: mean %.3f, median %.3f, IQR [%.3f, %.3f], min %.3f, max %.3f (n=%d)\n",
        mean(multi_lyap), med, q25, q75, minimum(multi_lyap),
        maximum(multi_lyap), length(multi_lyap))
@printf("RESULT topology=%s seed=%d nodes=%d K=%d best_epoch=%d val_loss=%.5f t_valid_lyap=%.3f multi_median_lyap=%.3f multi_mean_lyap=%.3f\n",
        topology, SEED, N, K_roll, best_epoch, best_val,
        t_valid_lyap, med, mean(multi_lyap))
flush(stdout)
