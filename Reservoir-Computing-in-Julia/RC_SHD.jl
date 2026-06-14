# Spiking Heidelberg Digits (SHD) classification with reservoir computing.
# SHD is a spiking-audio benchmark: spoken digits 0-9 (English + German, 20
# classes) encoded as spikes over 700 cochlear channels -- a temporal, spiking
# task, the kind an oscillator reservoir is actually built for (unlike static
# images or exact-lag memory like NARMA).
#
# Pipeline: spikes are pre-binned to a dense (channels x time) count matrix by
# prepare_shd.py (-> data/shd/shd_binned.h5). Each sample is fed through a
# reservoir over time; a linear ridge readout on the reservoir's time-summary
# features classifies the 20 digits. We compare:
#   * raw binned input (no reservoir)   -- control
#   * discrete tanh ESN
#   * FHN oscillator network            -- our approach
#
# Prereqs (run once, from the repo root):
#   python Reservoir-Computing-in-Julia/prepare_shd.py --pool 10 --tbins 100 --per-class 150
# Run:
#   julia --project=. Reservoir-Computing-in-Julia/RC_SHD.jl [seed] [quick] [nofigs] [nofhn]

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
save_figures = !("nofigs" in ARGS)
run_fhn = !("nofhn" in ARGS)
rng = MersenneTwister(SEED)

NR = quick ? 120 : 250
ridge_beta = 1.0
sr_esn = 0.9
leak_esn = 0.6
sigma_in = 0.2
# FHN
eps_fhn = 0.05; a_lo, a_hi = 0.95, 1.1; R0 = 0.5; speed_fhn = 6.0; sigma_in_fhn = 0.5
dt = 1.0

# --- load pre-binned SHD ----------------------------------------------------
path = joinpath(@__DIR__, "..", "data", "shd", "shd_binned.h5")
isfile(path) || error("missing $path -- run prepare_shd.py first")
# HDF5.jl reverses dims vs the (N,C,T) written by h5py -> read as (T,C,N)
Xtr_r, ytr, Xte_r, yte = h5open(path, "r") do f
    permutedims(read(f["X_train"]), (3, 2, 1)),  # -> (N, C, T)
    Int.(read(f["y_train"])),
    permutedims(read(f["X_test"]), (3, 2, 1)),
    Int.(read(f["y_test"]))
end
classes = sort(unique(ytr)); nclass = length(classes)
C = size(Xtr_r, 2); T = size(Xtr_r, 3)

# optional subsample for speed (esp. the FHN ODE)
function subsample(y, per; rng)
    idx = Int[]
    for c in unique(y); ci = shuffle(rng, findall(==(c), y)); append!(idx, ci[1:min(per, length(ci))]); end
    shuffle(rng, idx)
end
if quick
    tri = subsample(ytr, 20; rng=rng); tei = subsample(yte, 10; rng=rng)
    Xtr_r, ytr, Xte_r, yte = Xtr_r[tri,:,:], ytr[tri], Xte_r[tei,:,:], yte[tei]
end
ntr, nte = length(ytr), length(yte)
println("SHD: $ntr train / $nte test, $C channels x $T bins, $nclass classes, NR=$NR")

sample(X, n) = @view X[n, :, :]    # C x T spike-count matrix for sample n

# one-hot for labels in `classes`
function onehot(y)
    Y = zeros(nclass, length(y))
    for (j, c) in enumerate(y); Y[findfirst(==(c), classes), j] = 1; end
    return Y
end

# ridge one-hot readout; returns test accuracy
function classify(Ftr, Fte)
    mu = vec(mean(Ftr, dims=2)); sd = vec(std(Ftr, dims=2)) .+ 1e-8
    Ftr = (Ftr .- mu) ./ sd; Fte = (Fte .- mu) ./ sd
    Φtr = vcat(Ftr, ones(1, size(Ftr, 2))); Φte = vcat(Fte, ones(1, size(Fte, 2)))
    Wout = (onehot(ytr) * Φtr') / (Φtr * Φtr' + ridge_beta * I)
    pred = [classes[argmax(col)] for col in eachcol(Wout * Φte)]
    return mean(pred .== yte)
end

# --- feature extractors (return feat x nsample) -----------------------------
# raw: flattened binned input
raw_features(X) = reduce(hcat, [vec(sample(X, n)) for n in 1:size(X, 1)])

# temporal summary of a reservoir state matrix R (NR x T): the time-mean plus
# K evenly-spaced snapshots, so the readout sees the state's time evolution
# (essential for a temporal task) -> (K+1)*NR features.
const K_SNAP = 5
const FEATDIM = (K_SNAP + 1) * NR
time_summary(R) = (idx = round.(Int, range(1, size(R, 2), length=K_SNAP));
                   vcat(vec(mean(R, dims=2)), vec(R[:, idx])))

# discrete ESN: drive over time, temporal-summary features
function esn_features(X, A, Win)
    F = zeros(FEATDIM, size(X, 1))
    for n in 1:size(X, 1)
        R = drive_reservoir(A, Win, permutedims(sample(X, n)); f=tanh, leak=leak_esn)  # NR x T
        F[:, n] = time_summary(R)
    end
    return F
end

# FHN oscillator network: continuous reservoir driven by the spike-count input
function fhn_features(X, A, Win, a)
    F = zeros(FEATDIM, size(X, 1))
    t = collect(0:T-1) .* dt
    for n in 1:size(X, 1)
        drive = make_lerp(t, Win * sample(X, n))     # NR-vector current over time
        function rhs(dr, r, p, τ)
            u = @view r[1:NR]; w = @view r[NR+1:end]; du = @view dr[1:NR]; dw = @view dr[NR+1:end]
            g = drive(τ); mul!(du, A, u)
            @. du = speed_fhn * (du + u - u^3 / 3 - w + g)
            @. dw = speed_fhn * eps_fhn * (u - a + R0 * g)
            nothing
        end
        S = Array(solve(ODEProblem(rhs, zeros(2NR), (t[1], t[end])), Tsit5();
                        saveat=t, abstol=1e-5, reltol=1e-5))
        F[:, n] = time_summary(S[1:NR, :])
    end
    return F
end

# --- run --------------------------------------------------------------------
results = Pair{String,Float64}[]

println("raw baseline...");
push!(results, "raw input" => classify(raw_features(Xtr_r), raw_features(Xte_r)))

println("discrete ESN...")
A = generate_reservoir(rng, NR, 0.1; spectral_radius=sr_esn)
Win = sigma_in .* (2 .* rand(rng, NR, C) .- 1)
push!(results, "discrete ESN" => classify(esn_features(Xtr_r, A, Win), esn_features(Xte_r, A, Win)))

if run_fhn
    println("FHN oscillators...")
    Af = generate_reservoir(rng, NR, 0.1; spectral_radius=sr_esn)
    Winf = sigma_in_fhn .* (2 .* rand(rng, NR, C) .- 1)
    a = a_lo .+ (a_hi - a_lo) .* rand(rng, NR)
    push!(results, "FHN oscillators" => classify(fhn_features(Xtr_r, Af, Winf, a), fhn_features(Xte_r, Af, Winf, a)))
end

# --- report -----------------------------------------------------------------
println("\nSHD test accuracy (20 classes, chance = 0.05):")
open(joinpath(@__DIR__, "..", "shd_results.md"), "w") do io
    println(io, "# SHD (Spiking Heidelberg Digits) reservoir classification\n")
    println(io, "$ntr train / $nte test, $C channels × $T bins, 20 classes, NR=$NR, ",
                "linear ridge readout. Chance = 0.05.\n")
    println(io, "| Method | Test accuracy |")
    println(io, "|---|---|")
    for (name, acc) in results
        @printf("  %-16s %.3f\n", name, acc)
        println(io, @sprintf("| %s | %.3f |", name, acc))
    end
end

if save_figures
    fig = Figure(size=(640, 420))
    ax = Axis(fig[1, 1], ylabel="test accuracy", title="SHD classification (20 classes)",
              xticks=(1:length(results), [n for (n, _) in results]))
    barplot!(ax, 1:length(results), [a for (_, a) in results],
             color=[:gray70, :seagreen, :crimson][1:length(results)])
    hlines!(ax, [1/nclass], color=:black, linestyle=:dash)   # chance
    save(joinpath(@__DIR__, "..", "figures", "shd.png"), fig)
    println("\nFigure: figures/shd.png")
end
println("Wrote shd_results.md")
