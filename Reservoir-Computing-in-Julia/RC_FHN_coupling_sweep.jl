# Coupling-strength sweep for the FHN reservoir on the Lorenz-63 task.
#
# The graph topology, W_in, and node thresholds are fixed (seeded) once.
# Only the degree-normalised coupling `c` changes between runs, so any
# differences in validation skill are attributable to coupling alone. The final
# test continuation is evaluated once after the validation winner is fixed.
#
# Ridge regression is used as the readout (faster than NN, isolates the
# reservoir's contribution).  Pass `nn` on the command line to use the NN
# readout instead (much slower).
#
# Run:
#   julia --project=. Reservoir-Computing-in-Julia/RC_FHN_coupling_sweep.jl \
#         [topology] [seed] [nn] [quick] [nofigs]
#
#   topology : erdos_renyi (default) | complete | grid | watts_strogatz | barabasi_albert
#   seed     : integer (default 42)
#   nn       : use neural-network readout instead of ridge
#   quick    : small coupling grid (3 points) for a fast sanity check
#   nofigs   : skip figure generation

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

# ── command-line arguments ────────────────────────────────────────────────────
seed_arg = findfirst(a -> tryparse(Int, a) !== nothing, ARGS)
SEED = seed_arg === nothing ? 42 : parse(Int, ARGS[seed_arg])
topology = let t = findfirst(a -> a ∈ ("erdos_renyi","complete","grid",
                                        "watts_strogatz","barabasi_albert"), ARGS)
    t === nothing ? :erdos_renyi : Symbol(ARGS[t])
end
readout_kind = "nn" in ARGS ? :nn : :ridge
quick        = "quick"  in ARGS
save_figures = !("nofigs" in ARGS)

rng = MersenneTwister(SEED)
Random.seed!(SEED)
mkpath("figures")

println("FHN coupling sweep | topology=$topology | seed=$SEED | readout=$readout_kind")

# ── swept grid ────────────────────────────────────────────────────────────────
coupling_grid = quick ?
    [0.1, 0.3, 0.6] :
    [0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0]

# ── fixed hyperparameters (match RC_FHN_NN.jl defaults) ──────────────────────
n_nodes     = quick ? 64 : 256
dim_system  = 3
sigma_in    = 1.5
washout     = quick ? 100 : 500
dt          = 0.01
delay_steps = 10
eps_fhn     = 0.05
a_lo, a_hi  = 0.95, 1.1
R0          = 0.5
speed       = 20.0
ridge_beta  = 1e-4
noise_level = 0.02f0
epochs      = 400
batchsize   = 256
learning_rate = 1e-3

# ── graph (built once) ────────────────────────────────────────────────────────
function build_graph(kind::Symbol, n::Int, rng::AbstractRNG)
    g = kind === :erdos_renyi     ? erdos_renyi(n, 0.1; rng = rng) :
        kind === :complete        ? complete_graph(n) :
        kind === :grid            ? (isqrt(n)^2 == n ? Graphs.grid([isqrt(n), isqrt(n)]) :
                                    error("grid topology requires nodes to be a perfect square")) :
        kind === :watts_strogatz  ? watts_strogatz(n, 8, 0.25; rng = rng) :
        kind === :barabasi_albert ? barabasi_albert(n, 4; rng = rng) :
        error("unknown topology $kind")
    return SimpleDiGraph(g)
end

g = build_graph(topology, n_nodes, rng)
N = nv(g)
println("Topology: $topology, $N nodes, $(ne(g)) directed edges")

# raw (unnormalised) weight seeds; coupling is applied per-run below
raw_weights = Dict{Tuple{Int,Int},Float64}()
for e in edges(g)
    raw_weights[(dst(e), src(e))] = 0.1 + 0.9 * rand(rng)
end

function build_Wc(coupling::Float64)
    W = zeros(N, N)
    for ((i, j), w) in raw_weights
        W[i, j] = w
    end
    for i in 1:N
        s = sum(@view W[i, :])
        s > 0 && (W[i, :] .*= coupling / s)
    end
    return W, vec(sum(W, dims = 2))
end

# ── fixed random matrices ─────────────────────────────────────────────────────
W_in  = 2 * sigma_in * (rand(rng, N, dim_system) .- 0.5)
a_fhn = a_lo .+ (a_hi - a_lo) .* rand(rng, N)

# ── data (generated once) ─────────────────────────────────────────────────────
train_seconds = quick ? 10.0 : 200.0
test_seconds = quick ? 2.0 : 25.0
validation_seconds = quick ? 2.0 : 25.0
train_data, t_train, test_data, t_test =
    generate_lorenz_split(t_train = train_seconds, t_test = test_seconds, dt = dt)
validation_steps = round(Int, validation_seconds / dt)
fit_end = size(train_data, 1) - validation_steps
fit_end > washout + 2delay_steps || error("validation tail leaves too little fitting data")
validation_data = train_data[(fit_end + 1):end, :]
t_validation = collect(dt:dt:(validation_steps * dt))
scaler  = Standardizer(train_data[1:fit_end, :])
u_train = transform(scaler, train_data)

const d_steps = delay_steps
delay_features(S, idx) = vcat(S[:, idx], S[:, idx .- d_steps], S[:, idx .- 2d_steps])

# ── FHN RHS ───────────────────────────────────────────────────────────────────
function make_fhn_rhs(Wc, in_strength, eps_fhn, a_fhn, R0, speed, input)
    n = length(in_strength)
    return function fhn!(dr, r, p, t)
        u  = @view r[1:n];       w  = @view r[(n + 1):end]
        du = @view dr[1:n];      dw = @view dr[(n + 1):end]
        gin = input(t)
        mul!(du, Wc, u)
        @. du = speed * (du - in_strength * u + u - u^3 / 3 - w + gin)
        @. dw = speed * eps_fhn * (R0 * gin + u - a_fhn)
        nothing
    end
end

# ── closed-loop forecast ─────────────────────────────────────────────────────
function closed_loop(Wc, in_strength, R_train_ref, readout, r_end, x_end, n_steps)
    g_ref = Ref(zeros(N))
    fhn_cl! = make_fhn_rhs(Wc, in_strength, eps_fhn, a_fhn, R0, speed, t -> g_ref[])
    prob = ODEProblem(fhn_cl!, copy(r_end), (0.0, n_steps * dt))
    integ = init(prob, Tsit5(); abstol = 1e-6, reltol = 1e-6, save_everystep = false)
    pred = zeros(n_steps, dim_system)
    hist = R_train_ref[:, (end - 2d_steps):end]
    x = copy(x_end)
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

"Fit a readout with deterministic initialization/noise for fair coupling comparisons."
function fit_readout(F_train, Y; nn_seed=SEED + 10_000)
    if readout_kind === :ridge
        f_mu = vec(mean(F_train, dims = 2))
        f_sd = vec(std(F_train, dims = 2)) .+ 1e-8
        Fs = (F_train .- f_mu) ./ f_sd
        Phi = vcat(Fs, Fs .^ 2)
        W_out = ((Phi * Phi' + ridge_beta * I) \ (Phi * Y))'
        return f -> (fs = (f .- f_mu) ./ f_sd; W_out * vcat(fs, fs .^ 2))
    end

    Random.seed!(nn_seed)
    X_feat = Float32.(F_train)
    Y_targ = Float32.(Y')
    model = Chain(Dense(6N => 256, tanh), Dense(256 => dim_system))
    opt_st = Flux.setup(Adam(learning_rate), model)
    loader = Flux.DataLoader((X_feat, Y_targ); batchsize = batchsize, shuffle = true)
    for epoch in 1:epochs
        epoch == round(Int, 0.75 * epochs) && Flux.adjust!(opt_st, 1e-4)
        for (x, y) in loader
            xn = x .+ noise_level .* randn(Float32, size(x))
            loss, grads = Flux.withgradient(m -> Flux.mse(m(xn), y), model)
            Flux.update!(opt_st, model, grads[1])
        end
    end
    return f -> Float64.(model(Float32.(f)))
end

# ── sweep ────────────────────────────────────────────────────────────────────
G_train = W_in * u_train'
input_train_lerp = make_lerp(t_train, G_train)   # same signal every run

results = Tuple{Float64, Float64, Float64}[]   # validation results

println("\nValidation sweep ($(round(validation_steps * dt, digits=1)) s held-out training tail)")
println("  coupling │ t_valid (s) │ Lyapunov times")
println("  ─────────┼─────────────┼───────────────")

for coupling in coupling_grid
    Wc, in_strength = build_Wc(coupling)

    # --- drive training trajectory ---
    fhn_train! = make_fhn_rhs(Wc, in_strength, eps_fhn, a_fhn, R0, speed, input_train_lerp)
    prob_train  = ODEProblem(fhn_train!, zeros(2N), (t_train[1], t_train[end]))
    sol_train   = solve(prob_train, Tsit5(); saveat = t_train, abstol = 1e-6, reltol = 1e-6)
    R_train     = Array(sol_train)

    R_fit = R_train[:, 1:fit_end]
    F_fit = delay_features(R_fit, washout:fit_end)
    readout = fit_readout(F_fit, u_train[washout:fit_end, :])

    # --- closed-loop forecast ---
    r_end = R_fit[:, end]
    x_end = u_train[fit_end, :]
    pred_n = closed_loop(Wc, in_strength, R_fit, readout, r_end, x_end, validation_steps)
    X_pred = inverse_transform(scaler, pred_n)

    t_valid, t_valid_lyap = valid_prediction_time(validation_data, X_pred, t_validation)
    push!(results, (coupling, t_valid, t_valid_lyap))

    @printf("  %8.3f │ %11.3f │ %14.3f\n", coupling, t_valid, t_valid_lyap)
end

# ── summary ───────────────────────────────────────────────────────────────────
best = results[argmax(last.(results))]
println("\nBest validation coupling: $(best[1]) → $(round(best[3], digits=3)) Lyapunov times ",
        "($(round(best[2], digits=3)) s)")

# Refit the selected configuration on all training data, then evaluate the test
# continuation exactly once.
best_Wc, best_strength = build_Wc(best[1])
best_rhs = make_fhn_rhs(best_Wc, best_strength, eps_fhn, a_fhn, R0, speed, input_train_lerp)
best_sol = solve(ODEProblem(best_rhs, zeros(2N), (t_train[1], t_train[end])),
                 Tsit5(); saveat = t_train, abstol = 1e-6, reltol = 1e-6)
best_R = Array(best_sol)
best_F = delay_features(best_R, washout:size(best_R, 2))
best_readout = fit_readout(best_F, u_train[washout:end, :])
test_pred_n = closed_loop(best_Wc, best_strength, best_R, best_readout,
                          best_R[:, end], u_train[end, :], length(t_test))
test_pred = inverse_transform(scaler, test_pred_n)
test_valid, test_valid_lyap = valid_prediction_time(test_data, test_pred, t_test)
@printf("Final held-out test: %.3f s (%.3f Lyapunov times)\n", test_valid, test_valid_lyap)
@printf("RESULT topology=%s seed=%d readout=%s coupling=%.3f val_lyap=%.3f test_lyap=%.3f\n",
        topology, SEED, readout_kind, best[1], best[3], test_valid_lyap)

outfile = joinpath(@__DIR__, "..",
                   "fhn_coupling_sweep_results_$(topology)_$(readout_kind)_seed$(SEED).md")
open(outfile, "w") do io
    println(io, "# FHN coupling validation sweep — Lorenz-63 valid-prediction time\n")
    println(io, "topology=$topology, seed=$SEED, readout=$readout_kind, ",
                "N=$N, sigma_in=$sigma_in, speed=$speed, eps=$eps_fhn\n")
    println(io, "| coupling | validation t_valid (s) | validation Lyapunov times |")
    println(io, "|---|---|---|")
    for (c, tv, tl) in results
        println(io, @sprintf("| %.3f | %.3f | %.3f |", c, tv, tl))
    end
    println(io, "\n**Best validation coupling = $(best[1])** ",
                "($(round(best[3], digits=3)) Lyapunov times)\n")
    println(io, "Final held-out test: $(round(test_valid, digits=3)) s " *
                "($(round(test_valid_lyap, digits=3)) Lyapunov times).")
end
println("Results written to $outfile")

# ── plot ──────────────────────────────────────────────────────────────────────
if save_figures
    fig = Figure(size = (700, 420))
    ax  = Axis(fig[1, 1],
               xlabel = "Coupling c",
               ylabel = "Validation time (Lyapunov times)",
               title  = "FHN coupling validation sweep | $topology | readout=$readout_kind | seed=$SEED")
    cs  = [r[1] for r in results]
    tls = [r[3] for r in results]
    scatterlines!(ax, cs, tls, color = :steelblue, markersize = 10)
    vlines!(ax, [0.3], color = :gray, linestyle = :dash, label = "default (0.3)")
    vlines!(ax, [best[1]], color = :crimson, linestyle = :dot,
            label = "best ($(best[1]))")
    axislegend(ax, position = :rb, framevisible = false)
    savepath = "figures/fhn_coupling_sweep_$(topology)_$(readout_kind)_seed$(SEED).png"
    save(savepath, fig)
    println("Figure: $savepath")
end
