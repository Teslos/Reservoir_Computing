# NARMA-10 benchmark -- the standard reservoir-computing test of *memory* (the
# target depends on inputs up to 10 steps back) and *nonlinearity* (products of
# past values). Because the readout is a plain linear ridge, all the memory and
# nonlinearity must be provided by the reservoir itself -- which is exactly what
# makes NARMA-10 a fair characterization of a reservoir.
#
# Compares three reservoirs on the same task:
#   * discrete tanh ESN              (RC.jl-style)
#   * continuous-time tanh ESN       (CTESN-style, RC_LPCTESN.jl-style)
#   * FitzHugh-Nagumo oscillator net (our physical-reservoir approach)
#
#   target:  y(t+1) = 0.3 y(t) + 0.05 y(t) sum_{i=0}^{9} y(t-i)
#                     + 1.5 u(t-9) u(t) + 0.1,   u(t) ~ U[0, 0.5]
#   metric:  NRMSE = sqrt(<(yhat-y)^2> / var(y))  (lower is better)
#
# Run from the repo root:
#   julia --project=. Reservoir-Computing-in-Julia/RC_NARMA10.jl [seed] [quick] [nofigs]

include("common.jl")
using .RCCommon
using LinearAlgebra
using Random
using Statistics
using OrdinaryDiffEq
using CairoMakie
using Printf

# --- args / hyperparameters -------------------------------------------------
seed_arg = findfirst(a -> tryparse(Int, a) !== nothing, ARGS)
SEED0 = seed_arg === nothing ? 1 : parse(Int, ARGS[seed_arg])
quick = "quick" in ARGS
save_figures = !("nofigs" in ARGS)

n_seeds   = quick ? 3 : 8
NR        = quick ? 120 : 200      # reservoir size
n_train   = quick ? 1500 : 3000
n_test    = quick ? 600 : 1000
washout   = 200
density   = 0.1
ridge_beta = 1e-6

# discrete / continuous ESN knobs (tuned: leak=1 + sigma_in=0.2 is canonical
# for NARMA-10, ~0.3 NRMSE; the continuous reservoir prefers a modest speed)
sr_esn    = 0.9                    # spectral radius
leak_esn  = 1.0                    # leaking rate (1 = no leak; best for NARMA-10)
sigma_in  = 0.2                    # input scaling
dt        = 1.0                    # one NARMA step per unit time
speed_ct  = 3.0                    # continuous-reservoir time-scale factor

# FHN oscillator network knobs (best found; still weak on NARMA-10, see below)
eps_fhn   = 0.05
a_lo, a_hi = 0.95, 1.1            # heterogeneous thresholds straddling the Hopf
R0        = 0.5
speed_fhn = 8.0
sigma_in_fhn = 1.0

# --- NARMA-10 sequence ------------------------------------------------------
function narma10(n; rng)
    u = 0.5 .* rand(rng, n)
    y = zeros(n)
    for t in 10:(n - 1)
        y[t + 1] = 0.3 * y[t] + 0.05 * y[t] * sum(@view y[t - 9:t]) +
                   1.5 * u[t - 9] * u[t] + 0.1
    end
    return u, y
end

nrmse(yhat, y) = sqrt(mean(abs2, yhat .- y) / var(y))

# linear ridge readout on feature columns F (feat x n) -> scalar y
function eval_readout(F_tr, y_tr, F_te, y_te; beta)
    Φtr = vcat(F_tr, ones(1, size(F_tr, 2)))
    Φte = vcat(F_te, ones(1, size(F_te, 2)))
    Wout = (y_tr' * Φtr') / (Φtr * Φtr' + beta * I)   # 1 x (feat+1)
    yhat = vec(Wout * Φte)
    return nrmse(yhat, y_te), yhat
end

# --- reservoir state generators (each returns features F, size feat x n) ----
function states_discrete_esn(u, rng)
    A = generate_reservoir(rng, NR, density; spectral_radius = sr_esn)
    Win = sigma_in .* (2 .* rand(rng, NR, 1) .- 1)
    return drive_reservoir(A, Win, reshape(u, :, 1); f = tanh, leak = leak_esn)
end

function states_continuous_esn(u, rng)
    A = generate_reservoir(rng, NR, density; spectral_radius = sr_esn)
    Win = sigma_in .* (2 .* rand(rng, NR, 1) .- 1)
    t = collect(0:length(u)-1) .* dt
    drive = make_lerp(t, Win * u')
    rhs(dr, r, p, τ) = (dr .= speed_ct .* (-leak_esn .* r .+ tanh.(A * r .+ drive(τ))); nothing)
    sol = solve(ODEProblem(rhs, zeros(NR), (t[1], t[end])), Tsit5();
                saveat = t, abstol = 1e-6, reltol = 1e-6)
    return Array(sol)
end

function states_fhn(u, rng)
    A = generate_reservoir(rng, NR, density; spectral_radius = sr_esn)
    Win = sigma_in_fhn .* (2 .* rand(rng, NR, 1) .- 1)
    a = a_lo .+ (a_hi - a_lo) .* rand(rng, NR)
    t = collect(0:length(u)-1) .* dt
    drive = make_lerp(t, Win * u')
    function rhs(dr, r, p, τ)
        uu = @view r[1:NR]; w = @view r[NR+1:end]
        du = @view dr[1:NR]; dw = @view dr[NR+1:end]
        g = drive(τ)
        mul!(du, A, uu)
        @. du = speed_fhn * (du + uu - uu^3 / 3 - w + g)
        @. dw = speed_fhn * eps_fhn * (uu - a + R0 * g)
        return nothing
    end
    sol = solve(ODEProblem(rhs, zeros(2NR), (t[1], t[end])), Tsit5();
                saveat = t, abstol = 1e-6, reltol = 1e-6)
    S = Array(sol)
    return vcat(S[1:NR, :], S[NR+1:end, :])      # [u; w] features
end

const RESERVOIRS = ["discrete ESN" => states_discrete_esn,
                    "continuous ESN" => states_continuous_esn,
                    "FHN oscillators" => states_fhn]

# --- run --------------------------------------------------------------------
println("NARMA-10: NR=$NR, train=$n_train, test=$n_test, $(n_seeds) seeds")
results = Dict(name => Float64[] for (name, _) in RESERVOIRS)
example = Dict{String,Vector{Float64}}()
local y_te_example
for s in 1:n_seeds
    rng = MersenneTwister(SEED0 + s)
    u, y = narma10(n_train + n_test + 1; rng = rng)
    tr = (washout + 1):n_train
    te = (n_train + 1):(n_train + n_test)
    for (name, gen) in RESERVOIRS
        F = gen(u, rng)
        err, yhat = eval_readout(F[:, tr], y[tr], F[:, te], y[te]; beta = ridge_beta)
        push!(results[name], err)
        if s == 1
            example[name] = yhat
            global y_te_example = y[te]
        end
    end
    @printf("  seed %d: %s\n", s,
            join([@sprintf("%s=%.3f", name, results[name][end]) for (name, _) in RESERVOIRS], "  "))
end

# --- report -----------------------------------------------------------------
println("\nNARMA-10 NRMSE (mean ± std over $(n_seeds) seeds, lower is better):")
open(joinpath(@__DIR__, "..", "narma10_results.md"), "w") do io
    println(io, "# NARMA-10 reservoir benchmark\n")
    println(io, "NRMSE (mean ± std over $(n_seeds) seeds), NR=$NR, ",
                "train=$n_train, test=$n_test, linear ridge readout.\n")
    println(io, "| Reservoir | NRMSE |")
    println(io, "|---|---|")
    for (name, _) in RESERVOIRS
        v = results[name]
        m = sum(v) / length(v); sd = length(v) > 1 ? std(v) : 0.0
        @printf("  %-16s %.3f ± %.3f\n", name, m, sd)
        println(io, @sprintf("| %s | %.3f ± %.3f |", name, m, sd))
    end
    println(io, "\nReference: a well-tuned ESN reaches NRMSE ~0.2–0.4 on NARMA-10 ",
                "(NRMSE ≥ 1 means no better than predicting the mean).")
    println(io, "\nNARMA-10 rewards precise discrete-step *linear* memory (10 lags) with ",
                "only mild nonlinearity, which favors the discrete ESN. The continuous ",
                "reservoirs low-pass the input and lose exact lag memory; the excitable ",
                "FHN network in particular is poorly suited to this task — its strengths ",
                "(rich nonlinear transients, spiking/event-driven inputs) lie elsewhere.")
end

# --- figure -----------------------------------------------------------------
if save_figures
    fig = Figure(size = (900, 600))
    ax1 = Axis(fig[1, 1], xlabel = "step", ylabel = "y",
               title = "NARMA-10: prediction vs target (seed $(SEED0+1), first 200 test steps)")
    npl = min(200, length(y_te_example))
    lines!(ax1, 1:npl, y_te_example[1:npl], color = :black, label = "target")
    cols = [:crimson, :seagreen, :royalblue]
    for (i, (name, _)) in enumerate(RESERVOIRS)
        lines!(ax1, 1:npl, example[name][1:npl], color = (cols[i], 0.8), label = name)
    end
    axislegend(ax1, position = :rt, framevisible = false)

    ax2 = Axis(fig[2, 1], ylabel = "NRMSE", title = "NARMA-10 NRMSE by reservoir",
               xticks = (1:length(RESERVOIRS), [name for (name, _) in RESERVOIRS]))
    means = [sum(results[name]) / length(results[name]) for (name, _) in RESERVOIRS]
    sds = [length(results[name]) > 1 ? std(results[name]) : 0.0 for (name, _) in RESERVOIRS]
    barplot!(ax2, 1:length(RESERVOIRS), means, color = cols[1:length(RESERVOIRS)])
    errorbars!(ax2, 1:length(RESERVOIRS), means, sds, color = :black, whiskerwidth = 10)
    save(joinpath(@__DIR__, "..", "figures", "narma10.png"), fig)
    println("\nFigure: figures/narma10.png")
end
println("Wrote narma10_results.md")
