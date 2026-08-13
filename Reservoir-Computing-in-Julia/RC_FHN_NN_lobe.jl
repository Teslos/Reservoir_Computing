# A network of FitzHugh-Nagumo oscillators used as a *physical* reservoir to
# forecast the chaotic Lorenz-63 system, with a neural-network readout.
#
# Differences from the previous version of this script:
#   * The network ODE is written directly with a weighted adjacency matrix
#     instead of NetworkDynamics v0.8 (whose vertex callback here silently
#     used only 2 of each node's incident edges via `e_s, e_d = edges`).
#   * The forecast is genuinely closed-loop: the readout's prediction is fed
#     back as the reservoir input. Previously the reservoir was driven by
#     splines of the *true test data* during "prediction" (teacher forcing),
#     which made the results look far better than a real forecast.
#   * A teacher-forced (open-loop) one-step prediction is still computed, but
#     it is plotted dashed and labeled as such.
#   * Seeded RNG, standardized data, washout, no dead code, CPU-only Flux.
#
# Run from the repo root:  julia +1.11 --project=. Reservoir-Computing-in-Julia/RC_FHN_NN.jl

include("common.jl")
using .RCCommon
using Graphs
using LinearAlgebra
using Random
using Statistics
using OrdinaryDiffEq
using Flux
using CairoMakie
using Printf

# --- hyperparameters --------------------------------------------------------
SEED = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 42
# :erdos_renyi | :complete | :grid | :watts_strogatz | :barabasi_albert
# Command line: julia ... RC_FHN_NN.jl [topology] [seed] [nofigs] [ridge] [quad]
# `nofigs` skips figure generation and the 100 s climate run (for seed sweeps).
# `ridge` replaces the NN readout with ridge regression on [r; r^2], to
# isolate whether the readout or the reservoir limits closed-loop skill.
# `quad` augments the NN input with elementwise squares [f; f^2] (Pathak 2017),
# giving the linear first layer direct access to the bilinear Lorenz terms (xy, xz).
topology = isempty(ARGS) ? :erdos_renyi : Symbol(ARGS[1])
save_figures = !("nofigs" in ARGS)
readout_kind = "ridge" in ARGS ? :ridge : :nn
use_quad = "quad" in ARGS   # augment NN features with [f; f^2]
partial  = "partial" in ARGS  # observe only x(t); y and z reconstructed from history
hidden_arg = findfirst(a -> startswith(a, "hidden="), ARGS)
n_hidden = hidden_arg === nothing ? 256 : parse(Int, ARGS[hidden_arg][8:end])
# ridge regularization, settable as e.g. `beta=1e-2` on the command line
beta_arg = findfirst(a -> startswith(a, "beta="), ARGS)
ridge_beta = beta_arg === nothing ? 1e-4 : parse(Float64, ARGS[beta_arg][6:end])
# reservoir coupling, settable as e.g. `coupling=0.5` on the command line
coupling_arg = findfirst(a -> startswith(a, "coupling="), ARGS)
# sigma_in scaling, settable as e.g. `sigma_in=2.0` on the command line
sigma_in_arg = findfirst(a -> startswith(a, "sigma_in="), ARGS)
# reservoir size, settable as e.g. `nodes=64` on the command line (the June
# topology comparison in figures/ was run at 64; the paper's Lorenz section at 256)
nodes_arg = findfirst(a -> startswith(a, "nodes="), ARGS)
n_nodes = nodes_arg === nothing ? 256 : parse(Int, ARGS[nodes_arg][7:end])
dim_system = 3
sigma_in = sigma_in_arg === nothing ? 1.5 : parse(Float64, ARGS[sigma_in_arg][10:end])
washout = 500             # discard the first 5 s of reservoir transients
dt = 0.01
delay_steps = 10          # time-delay embedding: the readout sees
                          # [r(t); r(t - tau); r(t - 2 tau)], tau = delay_steps*dt.
                          # Triples the linear feature space without extra nodes.

# FitzHugh-Nagumo parameters
eps_fhn = 0.05            # time-scale separation
# Settable as `a_lo=1.05 a_hi=1.30` to move the population across the Hopf
# bifurcation: a > 1 is excitable (quiescent when undriven), a < 1 self-oscillates.
a_lo_arg = findfirst(a -> startswith(a, "a_lo="), ARGS)
a_hi_arg = findfirst(a -> startswith(a, "a_hi="), ARGS)
a_lo = a_lo_arg === nothing ? 0.95 : parse(Float64, ARGS[a_lo_arg][6:end])
a_hi = a_hi_arg === nothing ? 1.10 : parse(Float64, ARGS[a_hi_arg][6:end])
                          # Defaults U(0.95, 1.1) straddle the Hopf bifurcation at
                          # |a| = 1: a mixed population of weakly self-oscillating
                          # (a < 1) and barely excitable (a > 1) nodes. The stated
                          # rationale was that some intrinsic drive lets the
                          # autonomous closed loop sustain a Lorenz-like climate,
                          # since purely excitable nodes decay to quiescence once
                          # the forecast diverges. The sweep over a_lo/a_hi tests
                          # that rationale directly.
coupling = 0.3            # total in-coupling per node (degree-normalized)
R0 = 0.5                  # input coupling into the slow variable
speed = 20.0              # global time-scale factor: matches the oscillator
                          # response time to the ~1 s Lorenz oscillations,
                          # without it the nodes only low-pass the input

# readout training
epochs = 400
batchsize = 256
learning_rate = 1e-3
noise_level = 0.02f0    # noise injected into reservoir states during training;
                        # stabilizes the closed loop against its own feedback errors

rng = MersenneTwister(SEED)
Random.seed!(SEED)
mkpath("figures")

# --- graph topology -----------------------------------------------------------
function build_graph(kind::Symbol, n::Int, rng::AbstractRNG)
    g = kind === :erdos_renyi    ? erdos_renyi(n, 0.1; rng = rng) :
        kind === :complete       ? complete_graph(n) :
        kind === :grid           ? Graphs.grid([isqrt(n), isqrt(n)]) :
        kind === :watts_strogatz ? watts_strogatz(n, 8, 0.25; rng = rng) :
        kind === :barabasi_albert ? barabasi_albert(n, 4; rng = rng) :
        error("unknown topology $kind")
    return SimpleDiGraph(g)   # undirected edges become directed pairs
end

g = build_graph(topology, n_nodes, rng)
N = nv(g)
println("Topology: $topology, $(N) nodes, $(ne(g)) directed edges, ",
        "density $(round(Graphs.density(g), digits = 3))")

# Weighted coupling matrix with positive random weights ("resistivities").
# Each node's total in-coupling is normalized to `coupling`, so topology
# comparisons are not confounded by degree differences (an unnormalized
# dense graph over-couples and synchronizes the nodes).
Wc = zeros(N, N)
for e in edges(g)
    Wc[dst(e), src(e)] = 0.1 + 0.9 * rand(rng)
end
for i in 1:N
    s = sum(@view Wc[i, :])
    s > 0 && (Wc[i, :] .*= coupling / s)
end
in_strength = vec(sum(Wc, dims = 2))   # for the diffusive coupling term

# --- FHN network reservoir ------------------------------------------------------
# State r = [u; w].  For node i (a_i heterogeneous, `speed` rescales time):
#   du_i = speed * (u_i - u_i^3/3 - w_i + g_i(t) + sum_j Wc[i,j] (u_j - u_i))
#   dw_i = speed * eps * (R0 g_i(t) + u_i - a_i)
# `input` is a closure returning the N-vector g(t).
function make_fhn_rhs(Wc, in_strength, eps_fhn, a_fhn, R0, speed, input)
    n = length(in_strength)
    return function fhn!(dr, r, p, t)
        u = @view r[1:n]
        w = @view r[(n + 1):end]
        du = @view dr[1:n]
        dw = @view dr[(n + 1):end]
        gin = input(t)
        mul!(du, Wc, u)
        @. du = speed * (du - in_strength * u + u - u^3 / 3 - w + gin)
        @. dw = speed * eps_fhn * (R0 * gin + u - a_fhn)
        return nothing
    end
end

# --- data ------------------------------------------------------------------------
train_data, t_train, test_data, t_test =
    generate_lorenz_split(t_train = 200.0, t_test = 25.0, dt = dt)
scaler = Standardizer(train_data)
u_train = transform(scaler, train_data)
u_test = transform(scaler, test_data)

# --- LOBE VARIABLE ------------------------------------------------------------------
# A slow bistable state s carrying WHICH LOBE of the Lorenz attractor we are on:
#     ds/dt = gamma (s - s^3) + kappa * x_hat
# Wells at s = +-1; x_hat (the predicted/true standardized x) changes sign between
# lobes, so s follows lobe identity with hysteresis rather than tracking x.
# It is fed back as an extra reservoir input channel and added to the readout
# features, so the fast FHN network handles intra-lobe motion while s carries the
# discrete lobe decision and the switching.
#
# NOTE the Lorenz lobe centres C+- = (+-8.49, +-8.49, 27) are UNSTABLE saddle-foci,
# not attractors, so s's stable wells encode lobe IDENTITY, not the lobe centres
# themselves -- putting stable attractors at C+- would freeze the forecast in a lobe.
use_lobe = "lobe" in ARGS
gl_arg = findfirst(a -> startswith(a, "gamma_lobe="), ARGS)
kl_arg = findfirst(a -> startswith(a, "kappa_lobe="), ARGS)
gamma_lobe = gl_arg === nothing ? 2.0 : parse(Float64, ARGS[gl_arg][12:end])
kappa_lobe = kl_arg === nothing ? 1.0 : parse(Float64, ARGS[kl_arg][12:end])

"Integrate the lobe variable over a known driving series `xs` (RK2, step dt)."
function lobe_series(xs::AbstractVector, dt, gamma, kappa; s0 = 0.0)
    s = s0
    out = similar(xs, Float64)
    @inbounds for i in eachindex(xs)
        f1 = gamma * (s - s^3) + kappa * xs[i]
        sm = s + 0.5dt * f1
        f2 = gamma * (sm - sm^3) + kappa * xs[i]
        s = clamp(s + dt * f2, -3.0, 3.0)
        out[i] = s
    end
    return out
end
lobe_step(s, xhat, dt, gamma, kappa) = begin
    f1 = gamma * (s - s^3) + kappa * xhat
    sm = s + 0.5dt * f1
    f2 = gamma * (sm - sm^3) + kappa * xhat
    clamp(s + dt * f2, -3.0, 3.0)
end

dim_in = (partial ? 1 : dim_system) + (use_lobe ? 1 : 0)
W_in = 2 * sigma_in * (rand(rng, N, dim_in) .- 0.5)
a_fhn = a_lo .+ (a_hi - a_lo) .* rand(rng, N)   # heterogeneous node thresholds

# --- drive the reservoir with the training signal (teacher forcing) ---------------
s_train = use_lobe ? lobe_series(u_train[:, 1], dt, gamma_lobe, kappa_lobe) :
                     zeros(size(u_train, 1))
if use_lobe
    @printf("Lobe variable: gamma=%.2f kappa=%.2f  mean|s|=%.2f  sign flips=%d\n",
            gamma_lobe, kappa_lobe, mean(abs, s_train),
            count(i -> sign(s_train[i]) != sign(s_train[i - 1]), 2:length(s_train)))
end
U_drive = partial ? u_train[:, 1:1]' : u_train'
use_lobe && (U_drive = vcat(U_drive, s_train'))
G_train = W_in * U_drive                                   # N x n_train input currents
input_train = make_lerp(t_train, G_train)
fhn_train! = make_fhn_rhs(Wc, in_strength, eps_fhn, a_fhn, R0, speed, input_train)

r0 = zeros(2N)
prob_train = ODEProblem(fhn_train!, r0, (t_train[1], t_train[end]))
sol_train = solve(prob_train, Tsit5(); saveat = t_train,
                  abstol = 1e-6, reltol = 1e-6)
R_train = Array(sol_train)                      # 2N x n_train, [u; w] per column

# plot a few oscillator traces to inspect the reservoir dynamics
save_figures && let fig = Figure(size = (1000, 500))
    ax = Axis(fig[1, 1], xlabel = "Time (s)", ylabel = "u",
              title = "FitzHugh-Nagumo reservoir, first 8 oscillators (first 20 s)")
    n_show = round(Int, 20.0 / dt)
    for i in 1:8
        lines!(ax, t_train[1:n_show], R_train[i, 1:n_show], label = "node $i")
    end
    axislegend(ax, position = :rt, framevisible = false)
    save("figures/RC_FHN_reservoir_states_$(topology).png", fig)
end

# --- delayed feature space -----------------------------------------------------------
# Time-multiplexing: the readout operates on [r(t); r(t-tau); r(t-2tau)]
# (6N features) rather than the instantaneous state (2N).
const d_steps = delay_steps
"Delayed feature columns for indices `idx` of state matrix `S` (needs idx .- 2d >= 1)."
delay_features(S, idx) = vcat(S[:, idx], S[:, idx .- d_steps], S[:, idx .- 2d_steps])

n_tr = size(R_train, 2)
F_train = delay_features(R_train, washout:n_tr)   # washout > 2*delay_steps

# --- lobe DETECTOR: read lobe identity off the RESERVOIR, not off the output -------
# Driving s with the model's own predicted x is circular: once the forecast
# degrades, s switches at the wrong times and inherits the failure instead of
# correcting it. `lobe_res` instead fits a dedicated linear detector from the
# reservoir features to the lobe state, so s has a handle on which lobe the
# SUBSTRATE is in, independent of the main readout's error.
lobe_from_res = "lobe_res" in ARGS
lobe_detect = nothing
if use_lobe && lobe_from_res
    fs_mu = vec(mean(F_train, dims = 2))
    fs_sd = vec(std(F_train, dims = 2)) .+ 1e-8
    Fz = (F_train .- fs_mu) ./ fs_sd
    s_tgt = s_train[washout:n_tr]
    W_s = ((Fz * Fz' + 1e-2 * I) \ (Fz * s_tgt))'          # 1 x 6N
    ŝ_fit = vec(W_s * Fz)
    @printf("Lobe detector: train corr(s_hat, s) = %.4f, sign agreement = %.3f\n",
            cor(ŝ_fit, s_tgt), mean(sign.(ŝ_fit) .== sign.(s_tgt)))
    global lobe_detect = f -> (W_s * ((f .- fs_mu) ./ fs_sd))[1]
end

# the lobe state is part of the readout's input, so it can use lobe identity directly
use_lobe && (F_train = vcat(F_train, s_train[washout:n_tr]'))

# --- readout: delayed reservoir features at t_i -> Lorenz state at t_i -----------------
# `readout` accepts a feature vector or a matrix of feature columns.
if readout_kind === :ridge
    # standardize the features so the ridge penalty acts uniformly; the raw
    # excitable-FHN states have small, very unequal variances, which lets a
    # tiny beta produce huge weights that do not generalize
    f_mu = vec(mean(F_train, dims = 2))
    f_sd = vec(std(F_train, dims = 2)) .+ 1e-8
    Fs = (F_train .- f_mu) ./ f_sd
    Phi = vcat(Fs, Fs .^ 2)
    Y = u_train[washout:n_tr, :]
    W_out = ((Phi * Phi' + ridge_beta * I) \ (Phi * Y))'
    println("Ridge readout (beta = $ridge_beta) training MSE ",
            "(standardized units): ", mean(abs2, W_out * Phi .- Y'))
    readout(f) = (fs = (f .- f_mu) ./ f_sd; W_out * vcat(fs, fs .^ 2))
else
    aug(f::AbstractMatrix) = use_quad ? vcat(f, f .^ 2) : f
    aug(f::AbstractVector) = use_quad ? vcat(f, f .^ 2) : f
    n_feat = (6N + (use_lobe ? 1 : 0)) * (use_quad ? 2 : 1)

    X_feat = Float32.(aug(F_train))
    Y_targ = Float32.(u_train[washout:n_tr, :]')

    model = Chain(Dense(n_feat => n_hidden, tanh), Dense(n_hidden => dim_system))
    opt_state = Flux.setup(Adam(learning_rate), model)
    loader = Flux.DataLoader((X_feat, Y_targ); batchsize = batchsize, shuffle = true)

    for epoch in 1:epochs
        epoch == round(Int, 0.75 * epochs) && Flux.adjust!(opt_state, 1e-4)
        epoch_loss = 0.0
        for (x, y) in loader
            xn = x .+ noise_level .* randn(Float32, size(x))
            loss, grads = Flux.withgradient(model) do m
                Flux.mse(m(xn), y)
            end
            Flux.update!(opt_state, model, grads[1])
            epoch_loss += loss / length(loader)
        end
        epoch % 50 == 0 && println("Epoch $epoch, loss: $epoch_loss")
    end
    println("Final readout training MSE (standardized units): ",
            Flux.mse(model(X_feat), Y_targ))

    readout(r) = Float64.(model(Float32.(aug(r))))
end

# --- closed-loop (autonomous) forecast -----------------------------------------------
# The predicted Lorenz state is fed back as the input current, held constant
# over each dt step (zero-order hold). The truth is never used. Predictions
# are clamped to +-5 standardized units so a diverging forecast cannot blow
# up the cubic FHN nonlinearity.
function fhn_closed_loop_forecast(r_start, x_start, n_steps)
    g_ref = Ref(zeros(N))
    fhn_cl! = make_fhn_rhs(Wc, in_strength, eps_fhn, a_fhn, R0, speed, t -> g_ref[])
    prob = ODEProblem(fhn_cl!, copy(r_start), (0.0, n_steps * dt))
    integ = init(prob, Tsit5(); abstol = 1e-6, reltol = 1e-6,
                 save_everystep = false)
    pred = zeros(n_steps, dim_system)
    # rolling buffer of the last 2*delay_steps+1 states (dt-spaced) for the
    # delayed features; seeded from the end of the training run
    hist = R_train[:, (end - 2d_steps):end]
    x = copy(x_start)
    s = use_lobe ? s_train[end] : 0.0     # continue the lobe state from training
    for i in 1:n_steps
        drive = partial ? x[1:1] : x
        use_lobe && (drive = vcat(drive, s))
        g_ref[] = W_in * drive
        step!(integ, dt, true)
        hist = hcat(hist[:, 2:end], integ.u)
        feats_res = vcat(hist[:, end], hist[:, end - d_steps], hist[:, end - 2d_steps])
        feats = use_lobe ? vcat(feats_res, s) : feats_res
        x = clamp.(readout(feats), -5.0, 5.0)
        if use_lobe
            # drive s either from the reservoir (lobe_res, independent of the
            # readout's error) or from the model's own predicted x (circular)
            drive_s = lobe_from_res ? lobe_detect(feats_res) : x[1]
            s = lobe_step(s, drive_s, dt, gamma_lobe, kappa_lobe)
        end
        pred[i, :] = x
    end
    return pred
end

r_end = R_train[:, end]                  # synchronized state at the end of training
x_end = u_train[end, :]                  # last known true state (standardized)
pred_closed_n = fhn_closed_loop_forecast(r_end, x_end, length(t_test))
X_pred_closed = inverse_transform(scaler, pred_closed_n)

# --- open-loop (teacher-forced) one-step prediction, for comparison only --------------
s_test = use_lobe ? lobe_series(u_test[:, 1], dt, gamma_lobe, kappa_lobe;
                                s0 = s_train[end]) : zeros(size(u_test, 1))
U_test_drive = partial ? u_test[:, 1:1]' : u_test'
use_lobe && (U_test_drive = vcat(U_test_drive, s_test'))
G_test = W_in * U_test_drive
input_test = make_lerp(t_test, G_test)
fhn_test! = make_fhn_rhs(Wc, in_strength, eps_fhn, a_fhn, R0, speed, input_test)
prob_test = ODEProblem(fhn_test!, r_end, (t_test[1], t_test[end]))
sol_test = solve(prob_test, Tsit5(); saveat = t_test,
                 abstol = 1e-6, reltol = 1e-6)
# prepend the training tail so the delayed features are defined from the
# first test sample onward
R_ext = hcat(R_train[:, (end - 2d_steps + 1):end], Array(sol_test))
F_test = delay_features(R_ext, (2d_steps + 1):size(R_ext, 2))
use_lobe && (F_test = vcat(F_test, s_test'))
pred_open_n = readout(F_test)'
X_pred_open = inverse_transform(scaler, Matrix(pred_open_n))

# --- evaluation -------------------------------------------------------------------------
t_valid, t_valid_lyap = valid_prediction_time(test_data, X_pred_closed, t_test)
n_short = round(Int, 1 / (LORENZ_LYAPUNOV * dt))
mse_1lyap = mean(abs2, test_data[1:n_short, :] .- X_pred_closed[1:n_short, :])
mse_open = mean(abs2, test_data .- X_pred_open)
println("Closed-loop valid prediction time: $(round(t_valid, digits = 2)) s ",
        "($(round(t_valid_lyap, digits = 2)) Lyapunov times)")
println("Closed-loop MSE over the first Lyapunov time: ", mse_1lyap)
println("Open-loop (teacher-forced) MSE over the whole test set: ", mse_open)

# machine-readable summary line for seed sweeps
println("RESULT topology=$topology seed=$SEED readout=$readout_kind ",
        readout_kind === :ridge ? "beta=$ridge_beta " : "",
        readout_kind === :nn && use_quad ? "quad=true " : "",
        readout_kind === :nn ? "hidden=$n_hidden " : "",
        partial ? "partial=true " : "",
        "t_valid_s=$(round(t_valid, digits = 3)) ",
        "t_valid_lyap=$(round(t_valid_lyap, digits = 3)) mse_open=$(round(mse_open, digits = 4))")

if save_figures
    # --- long autonomous run for the attractor climate ------------------------------------
    pred_climate_n = fhn_closed_loop_forecast(r_end, x_end, round(Int, 100.0 / dt))
    X_climate = inverse_transform(scaler, pred_climate_n)

    # --- plots -----------------------------------------------------------------------------
    pfx = partial ? "_partial" : ""
    suffix = readout_kind === :ridge ? "$(topology)_ridge$(pfx)" :
             use_quad ? "$(topology)_nn_quad$(pfx)" : "$(topology)$(pfx)"
    plot_forecast(t_test, test_data, X_pred_closed,
                  "figures/lorenz_FHN_NN_$(suffix).png";
                  pred_open_loop = X_pred_open, t_valid = t_valid,
                  title = "FHN reservoir ($suffix) + $(readout_kind) readout " *
                          "(closed-loop valid for $(round(t_valid_lyap, digits = 1)) Lyapunov times)")
    plot_forecast_3d(test_data, X_pred_closed,
                     "figures/lorenz3d_FHN_NN_$(suffix).png";
                     title = "FHN reservoir ($suffix): closed-loop forecast")
    plot_lorenz_map(train_data, X_climate, "figures/lorenz_map_FHN_NN_$(suffix).png")
    println("Figures written to figures/lorenz_FHN_NN_$(suffix).png, ",
            "figures/lorenz3d_FHN_NN_$(suffix).png, ",
            "figures/lorenz_map_FHN_NN_$(suffix).png, ",
            "figures/RC_FHN_reservoir_states_$(suffix).png")
end
