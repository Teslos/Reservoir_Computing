# Finer hyperparameter sweep for the SHD reservoir: vary ESN `leak` and FHN
# `speed` (the two "memory depth" levers identified as dominant for SHD) while
# holding everything else fixed at the tuned values from RC_SHD.jl.
#
# The reservoir matrices (A, Win, a) and the common train/test subsample are
# built ONCE and reused across every sweep point, so the only thing changing is
# the knob under study -- the accuracy differences are attributable to it alone.
#
# A moderate subsample is used (the FHN ODE is the bottleneck); this finds the
# relative optimum reliably. Re-confirm the winner at full size in RC_SHD.jl.
#
# Run:
#   julia --project=. Reservoir-Computing-in-Julia/RC_SHD_sweep.jl [seed] [quick] [nofhn] [nofigs]

include("common.jl")
using .RCCommon
using HDF5
using LinearAlgebra
using Random
using Statistics
using OrdinaryDiffEq
using CairoMakie
using Printf

seed_arg = findfirst(a -> tryparse(Int, a) !== nothing, ARGS)
SEED = seed_arg === nothing ? 1 : parse(Int, ARGS[seed_arg])
quick = "quick" in ARGS
run_fhn = !("nofhn" in ARGS)
save_figures = !("nofigs" in ARGS)
rng = MersenneTwister(SEED)

# fixed at tuned values (RC_SHD.jl); only leak/speed vary below
NR = quick ? 150 : 500
TRAIN_PER = quick ? 20 : 60        # smaller than the headline run to keep the
TEST_PER  = quick ? 15 : 40        # FHN ODE tractable across many sweep points
ridge_beta = 1000.0
sr_esn = 1.1
sigma_in = 0.3
eps_fhn = 0.05; a_lo, a_hi = 0.95, 1.1; R0 = 0.5; sigma_in_fhn = 0.3
dt = 1.0

# --- swept ranges (finer, bracketing the current optima leak=0.05, speed=0.5) -
leak_grid  = quick ? [0.03, 0.05, 0.08] : [0.01, 0.02, 0.03, 0.05, 0.08, 0.12, 0.20]
speed_grid = quick ? [0.4, 0.5, 0.7]   : [0.20, 0.30, 0.40, 0.50, 0.70, 1.00]

# --- load pre-binned SHD ----------------------------------------------------
path = joinpath(@__DIR__, "..", "data", "shd", "shd_binned.h5")
isfile(path) || error("missing $path -- run prepare_shd.py first")
Xtr_r, ytr, Xte_r, yte = h5open(path, "r") do f
    permutedims(read(f["X_train"]), (3, 2, 1)),
    Int.(read(f["y_train"])),
    permutedims(read(f["X_test"]), (3, 2, 1)),
    Int.(read(f["y_test"]))
end
classes = sort(unique(ytr)); nclass = length(classes)
C = size(Xtr_r, 2); T = size(Xtr_r, 3)

function subsample(y, per; rng)
    idx = Int[]
    for c in unique(y); ci = shuffle(rng, findall(==(c), y)); append!(idx, ci[1:min(per, length(ci))]); end
    shuffle(rng, idx)
end
let tri = subsample(ytr, TRAIN_PER; rng=rng), tei = subsample(yte, TEST_PER; rng=rng)
    global Xtr_r, ytr, Xte_r, yte = Xtr_r[tri,:,:], ytr[tri], Xte_r[tei,:,:], yte[tei]
end
ntr, nte = length(ytr), length(yte)
println("SHD sweep: $ntr train / $nte test, $C ch x $T bins, NR=$NR, seed=$SEED")

sample(X, n) = @view X[n, :, :]

function onehot(y)
    Y = zeros(nclass, length(y))
    for (j, c) in enumerate(y); Y[findfirst(==(c), classes), j] = 1; end
    return Y
end

function classify(Ftr, Fte)
    mu = vec(mean(Ftr, dims=2)); sd = vec(std(Ftr, dims=2)) .+ 1e-8
    Ftr = (Ftr .- mu) ./ sd; Fte = (Fte .- mu) ./ sd
    Φtr = vcat(Ftr, ones(1, size(Ftr, 2))); Φte = vcat(Fte, ones(1, size(Fte, 2)))
    Wout = (onehot(ytr) * Φtr') / (Φtr * Φtr' + ridge_beta * I)
    pred = [classes[argmax(col)] for col in eachcol(Wout * Φte)]
    return mean(pred .== yte)
end

const K_SNAP = 8
const FEATDIM = (K_SNAP + 1) * NR
time_summary(R) = (idx = round.(Int, range(1, size(R, 2), length=K_SNAP));
                   vcat(vec(mean(R, dims=2)), vec(R[:, idx])))

# ESN features at a given leak
function esn_features(X, A, Win, leak)
    F = zeros(FEATDIM, size(X, 1))
    for n in 1:size(X, 1)
        R = drive_reservoir(A, Win, permutedims(sample(X, n)); f=tanh, leak=leak)
        F[:, n] = time_summary(R)
    end
    return F
end

# FHN features at a given speed
function fhn_features(X, A, Win, a, speed)
    F = zeros(FEATDIM, size(X, 1))
    t = collect(0:T-1) .* dt
    for n in 1:size(X, 1)
        drive = make_lerp(t, Win * sample(X, n))
        function rhs(dr, r, p, τ)
            u = @view r[1:NR]; w = @view r[NR+1:end]; du = @view dr[1:NR]; dw = @view dr[NR+1:end]
            g = drive(τ); mul!(du, A, u)
            @. du = speed * (du + u - u^3 / 3 - w + g)
            @. dw = speed * eps_fhn * (u - a + R0 * g)
            nothing
        end
        S = Array(solve(ODEProblem(rhs, zeros(2NR), (t[1], t[end])), Tsit5();
                        saveat=t, abstol=1e-5, reltol=1e-5))
        F[:, n] = time_summary(S[1:NR, :])
    end
    return F
end

# --- ESN leak sweep ---------------------------------------------------------
A = generate_reservoir(rng, NR, 0.1; spectral_radius=sr_esn)
Win = sigma_in .* (2 .* rand(rng, NR, C) .- 1)
esn_results = Pair{Float64,Float64}[]
println("\nESN leak sweep:")
for leak in leak_grid
    acc = classify(esn_features(Xtr_r, A, Win, leak), esn_features(Xte_r, A, Win, leak))
    push!(esn_results, leak => acc)
    @printf("  leak=%.3f  acc=%.3f\n", leak, acc)
end

# --- FHN speed sweep --------------------------------------------------------
fhn_results = Pair{Float64,Float64}[]
if run_fhn
    Af = generate_reservoir(rng, NR, 0.1; spectral_radius=sr_esn)
    Winf = sigma_in_fhn .* (2 .* rand(rng, NR, C) .- 1)
    a = a_lo .+ (a_hi - a_lo) .* rand(rng, NR)
    println("\nFHN speed sweep:")
    for speed in speed_grid
        acc = classify(fhn_features(Xtr_r, Af, Winf, a, speed),
                       fhn_features(Xte_r, Af, Winf, a, speed))
        push!(fhn_results, speed => acc)
        @printf("  speed=%.3f  acc=%.3f\n", speed, acc)
    end
end

# --- report -----------------------------------------------------------------
best_leak  = esn_results[argmax([a for (_, a) in esn_results])]
best_speed = isempty(fhn_results) ? nothing : fhn_results[argmax([a for (_, a) in fhn_results])]
println("\nBest ESN leak:  $(best_leak[1]) -> $(round(best_leak[2], digits=3))")
best_speed !== nothing && println("Best FHN speed: $(best_speed[1]) -> $(round(best_speed[2], digits=3))")

open(joinpath(@__DIR__, "..", "shd_sweep_results.md"), "w") do io
    println(io, "# SHD leak/speed sweep\n")
    println(io, "$ntr train / $nte test, $C channels × $T bins, NR=$NR, seed=$SEED, ",
                "ridge β=$ridge_beta, sr=$sr_esn. Everything else fixed at RC_SHD.jl values.\n")
    println(io, "## ESN leak (memory depth)\n")
    println(io, "| leak | test acc |\n|---|---|")
    for (l, acc) in esn_results; println(io, @sprintf("| %.3f | %.3f |", l, acc)); end
    println(io, "\n**Best leak = $(best_leak[1])** (acc $(round(best_leak[2], digits=3)))\n")
    if best_speed !== nothing
        println(io, "## FHN speed (slow dynamics = memory depth)\n")
        println(io, "| speed | test acc |\n|---|---|")
        for (s, acc) in fhn_results; println(io, @sprintf("| %.3f | %.3f |", s, acc)); end
        println(io, "\n**Best speed = $(best_speed[1])** (acc $(round(best_speed[2], digits=3)))")
    end
end

if save_figures
    fig = Figure(size=(820, 360))
    ax1 = Axis(fig[1, 1], xlabel="ESN leak", ylabel="test accuracy",
               title="ESN leak sweep", xscale=log10)
    scatterlines!(ax1, [l for (l, _) in esn_results], [a for (_, a) in esn_results],
                  color=:seagreen)
    vlines!(ax1, [0.05], color=:gray, linestyle=:dash)   # previous value
    if best_speed !== nothing
        ax2 = Axis(fig[1, 2], xlabel="FHN speed", ylabel="test accuracy",
                   title="FHN speed sweep")
        scatterlines!(ax2, [s for (s, _) in fhn_results], [a for (_, a) in fhn_results],
                      color=:crimson)
        vlines!(ax2, [0.5], color=:gray, linestyle=:dash)
    end
    save(joinpath(@__DIR__, "..", "figures", "shd_sweep.png"), fig)
    println("Figure: figures/shd_sweep.png")
end
println("Wrote shd_sweep_results.md")
