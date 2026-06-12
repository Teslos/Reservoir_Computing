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

# --- hyperparameters --------------------------------------------------------
SEED = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 42
# :erdos_renyi | :complete | :grid | :watts_strogatz | :barabasi_albert
# Command line: julia ... RC_FHN_NN.jl [topology] [seed] [nofigs] [ridge]
# `nofigs` skips figure generation and the 100 s climate run (for seed sweeps).
# `ridge` replaces the NN readout with ridge regression on [r; r^2], to
# isolate whether the readout or the reservoir limits closed-loop skill.
topology = isempty(ARGS) ? :erdos_renyi : Symbol(ARGS[1])
save_figures = !("nofigs" in ARGS)
readout_kind = "ridge" in ARGS ? :ridge : :nn
# ridge regularization, settable as e.g. `beta=1e-2` on the command line
beta_arg = findfirst(a -> startswith(a, "beta="), ARGS)
ridge_beta = beta_arg === nothing ? 1e-4 : parse(Float64, ARGS[beta_arg][6:end])
n_nodes = 256
dim_system = 3
sigma_in = 1.5            # input scaling (data is standardized)
washout = 500             # discard the first 5 s of reservoir transients
dt = 0.01
delay_steps = 10          # time-delay embedding: the readout sees
                          # [r(t); r(t - tau); r(t - 2 tau)], tau = delay_steps*dt.
                          # Triples the linear feature space without extra nodes.

# FitzHugh-Nagumo parameters
eps_fhn = 0.05            # time-scale separation
a_lo, a_hi = 1.05, 1.3    # per-node threshold a_i ~ U(a_lo, a_hi). |a| > 1 puts
                          # the nodes in the *excitable* (non-self-oscillating)
                          # regime: quiescent without input, near-linear
                          # sub-threshold response, and fading memory --
                          # which makes the states far more linearly decodable
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

W_in = 2 * sigma_in * (rand(rng, N, dim_system) .- 0.5)
a_fhn = a_lo .+ (a_hi - a_lo) .* rand(rng, N)   # heterogeneous node thresholds

# --- drive the reservoir with the training signal (teacher forcing) ---------------
G_train = W_in * u_train'                       # N x n_train input currents
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
    X_feat = Float32.(F_train)
    Y_targ = Float32.(u_train[washout:n_tr, :]')

    model = Chain(Dense(6N => 256, tanh), Dense(256 => dim_system))
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

    readout(r) = Float64.(model(Float32.(r)))
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
    for i in 1:n_steps
        g_ref[] = W_in * x
        step!(integ, dt, true)
        hist = hcat(hist[:, 2:end], integ.u)
        feats = vcat(hist[:, end], hist[:, end - d_steps], hist[:, end - 2d_steps])
        x = clamp.(readout(feats), -5.0, 5.0)
        pred[i, :] = x
    end
    return pred
end

r_end = R_train[:, end]                  # synchronized state at the end of training
x_end = u_train[end, :]                  # last known true state (standardized)
pred_closed_n = fhn_closed_loop_forecast(r_end, x_end, length(t_test))
X_pred_closed = inverse_transform(scaler, pred_closed_n)

# --- open-loop (teacher-forced) one-step prediction, for comparison only --------------
G_test = W_in * u_test'
input_test = make_lerp(t_test, G_test)
fhn_test! = make_fhn_rhs(Wc, in_strength, eps_fhn, a_fhn, R0, speed, input_test)
prob_test = ODEProblem(fhn_test!, r_end, (t_test[1], t_test[end]))
sol_test = solve(prob_test, Tsit5(); saveat = t_test,
                 abstol = 1e-6, reltol = 1e-6)
# prepend the training tail so the delayed features are defined from the
# first test sample onward
R_ext = hcat(R_train[:, (end - 2d_steps + 1):end], Array(sol_test))
F_test = delay_features(R_ext, (2d_steps + 1):size(R_ext, 2))
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
        "t_valid_s=$(round(t_valid, digits = 3)) ",
        "t_valid_lyap=$(round(t_valid_lyap, digits = 3)) mse_open=$(round(mse_open, digits = 4))")

if save_figures
    # --- long autonomous run for the attractor climate ------------------------------------
    pred_climate_n = fhn_closed_loop_forecast(r_end, x_end, round(Int, 100.0 / dt))
    X_climate = inverse_transform(scaler, pred_climate_n)

    # --- plots -----------------------------------------------------------------------------
    suffix = readout_kind === :ridge ? "$(topology)_ridge" : String(topology)
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
