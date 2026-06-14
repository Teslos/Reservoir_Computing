# Fixed-point quantization study for the FHN-reservoir closed-loop Lorenz
# forecast. Answers: what word length does an FPGA implementation need?
#
# Method:
#   1. Build the best config (grid, 256 near-bifurcation nodes, NN readout,
#      seed 42) exactly as RC_FHN_NN.jl, train the readout once in Float64.
#   2. Replace the adaptive Tsit5 solver with fixed-step Euler (K substeps per
#      dt) -- what an FPGA actually runs -- to get a hardware-realistic
#      Float64 baseline, separating the integration-scheme penalty from the
#      quantization penalty.
#   3. Quantize the values held in FPGA registers -- static weights, reservoir
#      state, hidden activations, feedback -- to B-bit signed fixed-point.
#      Internal multiply/accumulate is assumed wider (standard FPGA practice),
#      so quantization is applied at register/signal boundaries, not to every
#      partial product. Ranges are calibrated per signal group from the
#      Float64 Euler run (i.e. best-case per-signal Q-format scaling).
#   4. Sweep B and measure valid prediction time (Lyapunov times).
#
# CORRECTION (see integrator_matched.jl): the comparison this script prints
# between the fixed-step Float64 baseline (~0.45 LT) and the "adaptive Tsit5
# reference 1.57 LT" is NOT a clean integrator effect. Those two numbers came
# from different readouts. A controlled paired experiment shows that for a
# *fixed* readout, Tsit5 and fixed-step deployment give the SAME valid time --
# the 0.45-vs-1.57 spread is dominated by NN-initialization variance (closed-
# loop valid time ranges ~0.45-1.57 LT across readout seeds on this config).
# The *bit-width* conclusion below is unaffected (it sweeps one fixed model):
# ~12-bit knee, ~16 bits sufficient, collapse below 10 bits. But disregard any
# reading of the integrator as "the dominant penalty" -- it is not.
#
# Run from the repo root:
#   julia +1.11 --project=. Reservoir-Computing-in-Julia/fixedpoint_study.jl

include("common.jl")
using .RCCommon
using Graphs
using LinearAlgebra
using Random
using Statistics
using OrdinaryDiffEq
using Flux
using CairoMakie

# --- config (matches RC_FHN_NN.jl best result) -------------------------------
const SEED = 42
topology = :grid
n_nodes = 256
dim_system = 3
sigma_in = 1.5
washout = 500
dt = 0.01
delay_steps = 10
eps_fhn = 0.05
a_lo, a_hi = 0.95, 1.1
coupling = 0.3
R0 = 0.5
speed = 20.0
epochs = 400
batchsize = 256
learning_rate = 1e-3
noise_level = 0.02f0
K_sub = 50                  # substeps per dt; the closed-loop valid time is
                            # flat from K_sub=20..200 (see convergence check),
                            # so 50 is well converged and 4x faster than 200
method = :imex              # :rk2  -> explicit midpoint (previous study)
                            # :imex -> linear-implicit: the stiff linear part
                            #   (coupling + recovery) is solved implicitly with a
                            #   constant, pre-inverted operator Minv (just another
                            #   weight matrix on an FPGA); the per-node cubic is
                            #   the only term left explicit. A-stable.
bit_widths = [4, 6, 8, 10, 12, 14, 16, 24]

rng = MersenneTwister(SEED)
Random.seed!(SEED)
mkpath("figures")

# --- reservoir (identical to RC_FHN_NN.jl) -----------------------------------
g = SimpleDiGraph(Graphs.grid([isqrt(n_nodes), isqrt(n_nodes)]))
N = nv(g)
println("Topology: $topology, $N nodes, $(ne(g)) directed edges")

Wc = zeros(N, N)
for e in edges(g)
    Wc[dst(e), src(e)] = 0.1 + 0.9 * rand(rng)
end
for i in 1:N
    s = sum(@view Wc[i, :])
    s > 0 && (Wc[i, :] .*= coupling / s)
end
in_strength = vec(sum(Wc, dims = 2))

function make_fhn_rhs(Wc, in_strength, eps_fhn, a_fhn, R0, speed, input)
    n = length(in_strength)
    return function fhn!(dr, r, p, t)
        u = @view r[1:n]; w = @view r[(n + 1):end]
        du = @view dr[1:n]; dw = @view dr[(n + 1):end]
        gin = input(t)
        mul!(du, Wc, u)
        @. du = speed * (du - in_strength * u + u - u^3 / 3 - w + gin)
        @. dw = speed * eps_fhn * (R0 * gin + u - a_fhn)
        return nothing
    end
end

# --- data --------------------------------------------------------------------
train_data, t_train, test_data, t_test =
    generate_lorenz_split(t_train = 200.0, t_test = 25.0, dt = dt)
scaler = Standardizer(train_data)
u_train = transform(scaler, train_data)

W_in = 2 * sigma_in * (rand(rng, N, dim_system) .- 0.5)
a_fhn = a_lo .+ (a_hi - a_lo) .* rand(rng, N)

G_train = W_in * u_train'
input_train = make_lerp(t_train, G_train)
fhn_train! = make_fhn_rhs(Wc, in_strength, eps_fhn, a_fhn, R0, speed, input_train)
sol_train = solve(ODEProblem(fhn_train!, zeros(2N), (t_train[1], t_train[end])),
                  Tsit5(); saveat = t_train, abstol = 1e-6, reltol = 1e-6)
R_train = Array(sol_train)
n_tr = size(R_train, 2)

const d_steps = delay_steps
delay_features(S, idx) = vcat(S[:, idx], S[:, idx .- d_steps], S[:, idx .- 2d_steps])
F_train = delay_features(R_train, washout:n_tr)

# --- train the NN readout once in Float64 ------------------------------------
X_feat = Float32.(F_train)
Y_targ = Float32.(u_train[washout:n_tr, :]')
model = Chain(Dense(6N => 256, tanh), Dense(256 => dim_system))
opt_state = Flux.setup(Adam(learning_rate), model)
loader = Flux.DataLoader((X_feat, Y_targ); batchsize = batchsize, shuffle = true)
for epoch in 1:epochs
    epoch == round(Int, 0.75 * epochs) && Flux.adjust!(opt_state, 1e-4)
    for (x, y) in loader
        xn = x .+ noise_level .* randn(Float32, size(x))
        _, grads = Flux.withgradient(m -> Flux.mse(m(xn), y), model)
        Flux.update!(opt_state, model, grads[1])
    end
end
println("Readout training MSE: ", Flux.mse(model(X_feat), Y_targ))

# extract weights as plain Float64 matrices for the manual fixed-point loop
W1 = Float64.(model[1].weight); b1 = Float64.(model[1].bias)
W2 = Float64.(model[2].weight); b2 = Float64.(model[2].bias)

# --- fixed-point quantizer ---------------------------------------------------
# B-bit signed symmetric uniform quantization to range [-M, M].
# bits <= 0 means "no quantization" (Float64 reference).
function fxq(x::Real, bits::Int, M::Float64)
    (bits <= 0 || M <= 0) && return float(x)
    Δ = M / 2.0^(bits - 1)
    return clamp(round(x / Δ) * Δ, -M, M)
end
fxq(x::AbstractArray, bits::Int, M::Float64) = fxq.(x, bits, M)

# --- closed-loop forecast with fixed-step Euler + optional quantization -------
# When bits <= 0 this is the Float64 Euler baseline. `ranges` is a NamedTuple of
# per-group max-abs magnitudes; pass bits<=0 first to calibrate (returns the
# observed ranges), then sweep bit widths reusing those ranges.
function closed_loop(bits::Int, ranges; record = false, ksub::Int = K_sub)
    dt_sub = dt / ksub
    # static weights quantized once
    Wc_q  = fxq(Wc, bits, ranges.wc)
    Win_q = fxq(W_in, bits, ranges.win)
    W1_q  = fxq(W1, bits, ranges.w1); b1_q = fxq(b1, bits, ranges.w1)
    W2_q  = fxq(W2, bits, ranges.w2); b2_q = fxq(b2, bits, ranges.w2)

    # IMEX operator: the linear part L acts on s = [u; w] as
    #   du_lin = speed*((Wc - diag(in_strength) + I)*u - w),  dw_lin = speed*eps*u
    # (I - dt_sub*L) s_new = s + dt_sub*(forcing(gin) + cubic(u_old)).
    # M is constant, so Minv is precomputed in full precision and then stored
    # quantized -- on an FPGA it is just another weight matrix for the matvec.
    Minv_q = zeros(2N, 2N)
    if method === :imex
        L = zeros(2N, 2N)
        L[1:N, 1:N] = speed * (Wc - Diagonal(in_strength) + I)
        L[1:N, (N + 1):2N] = -speed * Matrix(I, N, N)
        L[(N + 1):2N, 1:N] = speed * eps_fhn * Matrix(I, N, N)
        Minv_q = fxq(inv(Matrix(I, 2N, 2N) - dt_sub * L), bits, ranges.minv)
    end

    n_steps = length(t_test)
    pred = zeros(n_steps, dim_system)
    obs = Dict(:u => 0.0, :w => 0.0, :g => 0.0, :h => 0.0, :fb => 0.0)
    hist = fxq(R_train[:, (end - 2d_steps):end], bits, ranges.u)
    state = fxq(R_train[:, end], bits, ranges.u)
    x = fxq(u_train[end, :], bits, ranges.fb)

    # explicit FHN derivative (used by :rk2); accumulators full precision,
    # results quantized at the register write.
    deriv(u, w, gin) = (speed .* (Wc_q * u .- in_strength .* u .+ u .- u .^ 3 ./ 3 .- w .+ gin),
                        speed * eps_fhn .* (R0 .* gin .+ u .- a_fhn))

    for i in 1:n_steps
        gin = fxq(Win_q * x, bits, ranges.g)
        # forcing constant over the substeps (zero-order-hold input)
        forcing = vcat(speed .* gin, speed * eps_fhn .* (R0 .* gin .- a_fhn))
        for _ in 1:ksub
            u = @view state[1:N]; w = @view state[(N + 1):2N]
            if method === :imex
                # linear-implicit: only the cubic is explicit
                rhs = state .+ dt_sub .* (forcing .+ vcat(-speed .* u .^ 3 ./ 3, zeros(N)))
                state = fxq(Minv_q * rhs, bits, ranges.u)
            else
                # RK2 (midpoint): 2nd-order explicit
                du1, dw1 = deriv(u, w, gin)
                um = u .+ (dt_sub / 2) .* du1
                wm = w .+ (dt_sub / 2) .* dw1
                du2, dw2 = deriv(um, wm, gin)
                unew = fxq(u .+ dt_sub .* du2, bits, ranges.u)
                wnew = fxq(w .+ dt_sub .* dw2, bits, ranges.u)
                state = vcat(unew, wnew)
            end
        end
        hist = hcat(hist[:, 2:end], state)
        feats = vcat(hist[:, end], hist[:, end - d_steps], hist[:, end - 2d_steps])
        h = fxq(tanh.(fxq(W1_q * feats .+ b1_q, bits, ranges.g)), bits, ranges.h)
        y = fxq(W2_q * h .+ b2_q, bits, ranges.fb)
        x = clamp.(y, -5.0, 5.0)
        pred[i, :] = x
        if record
            obs[:g] = max(obs[:g], maximum(abs, gin))
            obs[:u] = max(obs[:u], maximum(abs, state))
            obs[:h] = max(obs[:h], maximum(abs, h))
            obs[:fb] = max(obs[:fb], maximum(abs, x))
        end
    end
    return pred, obs
end

# --- check that the fixed-step Float64 loop converges to the Tsit5 1.57 ------
# (separates the integration-scheme penalty from the quantization penalty)
seed_ranges = (u = 12.0, w = 12.0, g = 30.0, h = 1.0, fb = 5.0, minv = 5.0,
               wc = maximum(abs, Wc), win = maximum(abs, W_in),
               w1 = max(maximum(abs, W1), maximum(abs, b1)),
               w2 = max(maximum(abs, W2), maximum(abs, b2)))
println("K_sub convergence (Float64, no quantization):")
for k in (20, 50, 200)
    p, o = closed_loop(0, seed_ranges; record = true, ksub = k)
    _, tvl = valid_prediction_time(test_data, inverse_transform(scaler, p), t_test)
    println("  K_sub=$k: $(round(tvl, digits = 2)) Lyapunov times, ",
            "state range ±$(round(o[:u], digits = 2))")
end

# --- calibrate ranges from the *operating* dynamics --------------------------
# Size the fixed-point registers to the teacher-forced reservoir's range (the
# range the device actually uses while tracking), not the post-divergence
# closed-loop excursions, which an FPGA would simply saturate. Same principle
# for the input current: calibrate to the teacher-forced drive.
pred_float, obs = closed_loop(0, seed_ranges; record = true)
state_range = 1.2 * maximum(abs, R_train)
input_range = 1.2 * maximum(abs, G_train)
# calibrate the implicit-operator range from the actual Minv
L0 = zeros(2N, 2N)
L0[1:N, 1:N] = speed * (Wc - Diagonal(in_strength) + I)
L0[1:N, (N + 1):2N] = -speed * Matrix(I, N, N)
L0[(N + 1):2N, 1:N] = speed * eps_fhn * Matrix(I, N, N)
minv_range = maximum(abs, inv(Matrix(I, 2N, 2N) - (dt / K_sub) * L0))
ranges = (u = state_range, w = state_range, g = input_range, h = 1.0, fb = 5.0,
          minv = minv_range,
          wc = seed_ranges.wc, win = seed_ranges.win,
          w1 = seed_ranges.w1, w2 = seed_ranges.w2)
println("Operating ranges: state=±$(round(ranges.u, digits=2)) ",
        "input=±$(round(ranges.g, digits=2)) feedback=±5  ",
        "(closed-loop excursions reached ±$(round(obs[:u], digits=1)), saturated)")

X_float = inverse_transform(scaler, pred_float)
tv_float, tvl_float = valid_prediction_time(test_data, X_float, t_test)
println("Float64 $(method) baseline: $(round(tvl_float, digits = 2)) Lyapunov times ",
        "(explicit RK2 fixed-step was 0.45, adaptive Tsit5 reference 1.57)")

# --- sweep bit widths --------------------------------------------------------
results = Tuple{Int, Float64}[]
preds = Dict{Int, Matrix{Float64}}()
for B in bit_widths
    pred_q, _ = closed_loop(B, ranges)
    Xq = inverse_transform(scaler, pred_q)
    _, tvl = valid_prediction_time(test_data, Xq, t_test)
    push!(results, (B, tvl))
    preds[B] = Xq
    println("  $(B)-bit: $(round(tvl, digits = 2)) Lyapunov times")
end

# --- plots -------------------------------------------------------------------
fig = Figure(size = (760, 520))
ax = Axis(fig[1, 1], xlabel = "Word length (bits)",
          ylabel = "Valid prediction time (Lyapunov times)",
          title = "FHN reservoir FPGA word-length study ($(method), grid, near-bifurcation)")
hlines!(ax, [tvl_float], color = :gray, linestyle = :dash,
        label = "Float64 $(method) baseline")
scatterlines!(ax, first.(results), last.(results), color = :firebrick,
              markersize = 12, label = "fixed-point")
axislegend(ax, position = :rb, framevisible = false)
save("figures/fixedpoint_validtime_vs_bits_$(method).png", fig)

# forecast at a low and a high bit width, vs truth
let
    fig2 = Figure(size = (1000, 600))
    for (row, B) in enumerate((8, 12))
        for k in 1:3
            ax = Axis(fig2[k, row], ylabel = ["x", "y", "z"][k],
                      xlabel = k == 3 ? "Time (s)" : "",
                      title = k == 1 ? "$(B)-bit fixed-point" : "")
            lines!(ax, t_test, test_data[:, k], color = :black)
            lines!(ax, t_test, preds[B][:, k], color = :red)
        end
    end
    save("figures/fixedpoint_forecast_8_vs_12bit_$(method).png", fig2)
end

println("Figures: figures/fixedpoint_validtime_vs_bits_$(method).png, ",
        "figures/fixedpoint_forecast_8_vs_12bit_$(method).png")
