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
const SEED = 42
# :erdos_renyi | :complete | :grid | :watts_strogatz | :barabasi_albert
# can be passed on the command line: julia ... RC_FHN_NN.jl watts_strogatz
topology = isempty(ARGS) ? :erdos_renyi : Symbol(ARGS[1])
n_nodes = 64
dim_system = 3
sigma_in = 1.5            # input scaling (data is standardized)
washout = 500             # discard the first 5 s of reservoir transients
dt = 0.01

# FitzHugh-Nagumo parameters
eps_fhn = 0.05            # time-scale separation
a_lo, a_hi = 0.3, 0.7     # per-node threshold a_i ~ U(a_lo, a_hi); heterogeneity
                          # breaks synchronization so nodes respond diversely
coupling = 0.3            # global coupling strength
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

# Weighted coupling matrix: Wc[dst, src] = coupling * w_edge, with positive
# random weights ("resistivities"), instead of the old deterministic
# bell-shape-over-edge-index weights.
Wc = zeros(N, N)
for e in edges(g)
    Wc[dst(e), src(e)] = coupling * (0.1 + 0.9 * rand(rng))
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
let fig = Figure(size = (1000, 500))
    ax = Axis(fig[1, 1], xlabel = "Time (s)", ylabel = "u",
              title = "FitzHugh-Nagumo reservoir, first 8 oscillators (first 20 s)")
    n_show = round(Int, 20.0 / dt)
    for i in 1:8
        lines!(ax, t_train[1:n_show], R_train[i, 1:n_show], label = "node $i")
    end
    axislegend(ax, position = :rt, framevisible = false)
    save("figures/RC_FHN_reservoir_states_$(topology).png", fig)
end

# --- NN readout: reservoir state at t_i -> Lorenz state at t_i ----------------------
X_feat = Float32.(R_train[:, washout:end])
Y_targ = Float32.(u_train[washout:end, :]')

model = Chain(Dense(2N => 256, tanh), Dense(256 => dim_system))
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
    x = copy(x_start)
    for i in 1:n_steps
        g_ref[] = W_in * x
        step!(integ, dt, true)
        x = clamp.(readout(integ.u), -5.0, 5.0)
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
pred_open_n = Float64.(model(Float32.(Array(sol_test))))'
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

# --- long autonomous run for the attractor climate ----------------------------------------
pred_climate_n = fhn_closed_loop_forecast(r_end, x_end, round(Int, 100.0 / dt))
X_climate = inverse_transform(scaler, pred_climate_n)

# --- plots ----------------------------------------------------------------------------------
suffix = String(topology)
plot_forecast(t_test, test_data, X_pred_closed,
              "figures/lorenz_FHN_NN_$(suffix).png";
              pred_open_loop = X_pred_open, t_valid = t_valid,
              title = "FHN reservoir ($suffix) + NN readout " *
                      "(closed-loop valid for $(round(t_valid_lyap, digits = 1)) Lyapunov times)")
plot_forecast_3d(test_data, X_pred_closed,
                 "figures/lorenz3d_FHN_NN_$(suffix).png";
                 title = "FHN reservoir ($suffix): closed-loop forecast")
plot_lorenz_map(train_data, X_climate, "figures/lorenz_map_FHN_NN_$(suffix).png")
println("Figures written to figures/lorenz_FHN_NN_$(suffix).png, ",
        "figures/lorenz3d_FHN_NN_$(suffix).png, ",
        "figures/lorenz_map_FHN_NN_$(suffix).png, ",
        "figures/RC_FHN_reservoir_states_$(suffix).png")
