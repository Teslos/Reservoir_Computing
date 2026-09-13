# Dry Bean classification with a fixed random reservoir and ridge readout.
#
# The original script accidentally constructed its test loader from the training
# tensors, evolved one reservoir state across unrelated samples, and ended in a
# stale Lorenz-forecast block containing undefined variables. This version uses a
# stratified split, fits normalization on training data only, resets the reservoir
# for every bean, and evaluates the test split exactly once.

include("drybean.jl")
using .drybean
using DataFrames
using LinearAlgebra
using NNlib: swish
using Random
using Statistics
using Printf

seed_arg = findfirst(a -> tryparse(Int, a) !== nothing, ARGS)
SEED = seed_arg === nothing ? 1234 : parse(Int, ARGS[seed_arg])
quick = "quick" in ARGS
save_figures = !("nofigs" in ARGS)

NR = quick ? 64 : 200
density = 0.05
spectral_radius = 0.9
sigma_in = 0.2
leak = 0.9
settle_steps = quick ? 3 : 5
ridge_beta = 1e-2
rng = MersenneTwister(SEED)

function generate_reservoir(rng, dim, density; spectral_radius)
    A = (rand(rng, dim, dim) .< density) .* (2 .* rand(rng, dim, dim) .- 1)
    rho = maximum(abs, eigvals(A))
    rho > eps(Float64) || error("zero-radius reservoir; increase NR or density")
    return A .* (spectral_radius / rho)
end

data_path = joinpath(@__DIR__, "data", "DryBeanDataset.csv")
isfile(data_path) || error("missing Dry Bean dataset: $data_path")
db = read_drybean(data_path)
X = Matrix{Float64}(db[:, 1:16])'
labels = String.(db[:, 17])
classes = sort(unique(labels))

"Return stratified train/test indices without inspecting feature values."
function stratified_split(labels; train_fraction=0.8, rng)
    train_idx = Int[]; test_idx = Int[]
    for class in unique(labels)
        ids = shuffle(rng, findall(==(class), labels))
        ntrain = clamp(round(Int, train_fraction * length(ids)), 1, length(ids) - 1)
        append!(train_idx, ids[1:ntrain])
        append!(test_idx, ids[(ntrain + 1):end])
    end
    return shuffle(rng, train_idx), shuffle(rng, test_idx)
end

train_idx, test_idx = stratified_split(labels; rng)
X_train, X_test = X[:, train_idx], X[:, test_idx]
y_train, y_test = labels[train_idx], labels[test_idx]

# Train-only min/max scaling. Constant columns remain finite.
xmin = minimum(X_train, dims = 2)
xrange = maximum(X_train, dims = 2) .- xmin
xrange[xrange .<= eps(Float64)] .= 1.0
X_train = (X_train .- xmin) ./ xrange
X_test = (X_test .- xmin) ./ xrange

function onehot(labels)
    Y = zeros(length(classes), length(labels))
    for (j, label) in enumerate(labels)
        Y[findfirst(==(label), classes), j] = 1.0
    end
    return Y
end

"Drive every static sample independently with a constant input."
function reservoir_features(X, A, W_in; steps=settle_steps)
    R = zeros(size(A, 1), size(X, 2))
    for _ in 1:steps
        R = (1 - leak) .* R .+ leak .* swish.(A * R .+ W_in * X)
    end
    return R
end

function ridge_accuracy(F_train, y_train, F_test, y_test)
    mu = mean(F_train, dims = 2)
    sd = std(F_train, dims = 2)
    sd[sd .<= eps(Float64)] .= 1.0
    Z_train = (F_train .- mu) ./ sd
    Z_test = (F_test .- mu) ./ sd
    Phi_train = vcat(Z_train, ones(1, size(Z_train, 2)))
    Phi_test = vcat(Z_test, ones(1, size(Z_test, 2)))
    W_out = (onehot(y_train) * Phi_train') /
            (Phi_train * Phi_train' + ridge_beta * I)
    pred = [classes[argmax(col)] for col in eachcol(W_out * Phi_test)]
    return mean(pred .== y_test)
end

raw_accuracy = ridge_accuracy(X_train, y_train, X_test, y_test)
A = generate_reservoir(rng, NR, density; spectral_radius)
W_in = sigma_in .* (2 .* rand(rng, NR, size(X_train, 1)) .- 1)
R_train = reservoir_features(X_train, A, W_in)
R_test = reservoir_features(X_test, A, W_in)
reservoir_accuracy = ridge_accuracy(R_train, y_train, R_test, y_test)

println("Dry Bean: $(length(y_train)) train / $(length(y_test)) test, " *
        "$(length(classes)) classes, NR=$NR, seed=$SEED")
@printf("Raw-feature ridge accuracy: %.4f\n", raw_accuracy)
@printf("Reservoir ridge accuracy:   %.4f\n", reservoir_accuracy)
@printf("RESULT seed=%d NR=%d raw_acc=%.4f reservoir_acc=%.4f\n",
        SEED, NR, raw_accuracy, reservoir_accuracy)

if save_figures
    @eval using CairoMakie
    mkpath(joinpath(@__DIR__, "..", "figures"))
    fig = Figure(size = (620, 420))
    ax = Axis(fig[1, 1], ylabel = "test accuracy",
              title = "Dry Bean classification (seed $SEED)",
              xticks = (1:2, ["raw ridge", "reservoir ridge"]))
    barplot!(ax, 1:2, [raw_accuracy, reservoir_accuracy],
             color = [:gray65, :seagreen])
    ylims!(ax, 0, 1)
    path = joinpath(@__DIR__, "..", "figures", "drybean_seed$(SEED).png")
    save(path, fig)
    println("Figure: $path")
end
