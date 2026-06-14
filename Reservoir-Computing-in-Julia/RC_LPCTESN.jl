# Linear-projection continuous-time echo state network (LPCTESN) as a reservoir
# to forecast the chaotic Lorenz-63 system. Continuous-time counterpart of the
# discrete ridge ESN in RC.jl and the FHN physical reservoir in RC_FHN_NN.jl.
#
# Method (Anantharaman et al. 2021, "Accelerating Simulation of Stiff Nonlinear
# Systems using Continuous-Time Echo State Networks", arXiv:2010.04004; see
# docs/2010.04004v6.pdf):
#   reservoir ODE   r'(t) = speed * (-leak r + tanh(A r + W_in x(t)))
#   linear readout  x(t)  = W_out r(t)            (the "LP" / identity output)
# Only W_out is trained, by ridge least squares. For forecasting we teacher-force
# the reservoir on the training trajectory, fit W_out, then close the loop
# (drive the reservoir with its own prediction) and integrate forward. Because
# the readout is linear, the closed loop is itself a smooth ODE.
#
# Run from the repo root:
#   julia --project=. Reservoir-Computing-in-Julia/RC_LPCTESN.jl [seed] [quick] [nofigs]

include("common.jl")
using .RCCommon
using LinearAlgebra
using Random
using Statistics
using OrdinaryDiffEq
using CairoMakie

# --- hyperparameters --------------------------------------------------------
seed_arg = findfirst(a -> tryparse(Int, a) !== nothing, ARGS)
SEED = seed_arg === nothing ? 42 : parse(Int, ARGS[seed_arg])
quick = "quick" in ARGS
save_figures = !("nofigs" in ARGS)

NR = quick ? 150 : 300          # reservoir size
dim_system = 3
spectral_radius = 1.1           # >1 is fine: the leaky continuous reservoir is
                                # still contracting for the echo-state property
density = 0.1
sigma_in = 0.5                  # input scaling (data is standardized)
leak = 1.0
speed = 10.0                    # reservoir time-scale factor; matches the
                                # reservoir response to the ~1 s Lorenz dynamics
ridge_beta = 0.1                # Tikhonov regularization (stabilizes the loop)
noise_level = 1e-3              # state noise during fitting (loop robustness)
washout = 500                   # discard initial reservoir transients
dt = 0.01
t_train = quick ? 60.0 : 200.0
t_test  = quick ? 15.0 : 25.0
t_climate = quick ? 40.0 : 100.0

rng = MersenneTwister(SEED)
Random.seed!(SEED)
mkpath("figures")

# --- data -------------------------------------------------------------------
train_data, t_tr, test_data, t_te = generate_lorenz_split(; t_train, t_test, dt)
scaler = Standardizer(train_data)
u_train = transform(scaler, train_data)

A = generate_reservoir(rng, NR, density; spectral_radius)
W_in = 2 * sigma_in .* (rand(rng, NR, dim_system) .- 0.5)

# --- teacher-forced continuous reservoir ------------------------------------
reservoir_rhs(input) = (dr, r, p, t) ->
    (dr .= speed .* (-leak .* r .+ tanh.(A * r .+ input(t))); nothing)

input_train = make_lerp(t_tr, W_in * u_train')        # NR-vector current g(t)
sol_tr = solve(ODEProblem(reservoir_rhs(input_train), zeros(NR), (t_tr[1], t_tr[end])),
               Tsit5(); saveat = t_tr, abstol = 1e-6, reltol = 1e-6)
R_train = Array(sol_tr)                                # NR x n_train

# --- linear-projection ridge readout ----------------------------------------
n_tr = size(R_train, 2)
F = R_train[:, washout:n_tr]
f_mu = vec(mean(F, dims = 2)); f_sd = vec(std(F, dims = 2)) .+ 1e-8
Fs = (F .- f_mu) ./ f_sd .+ noise_level .* randn(rng, size(F))
Y = u_train[washout:n_tr, :]
W_out = ((Fs * Fs' + ridge_beta * I) \ (Fs * Y))'      # 3 x NR
readout_std(r) = W_out * ((r .- f_mu) ./ f_sd)
println("Readout training MSE (standardized): ",
        round(mean(abs2, W_out * ((F .- f_mu) ./ f_sd) .- Y'), digits = 5))

# --- closed-loop (autonomous) forecast: a pure ODE, since the readout is linear
closed_rhs(dr, r, p, t) =
    (dr .= speed .* (-leak .* r .+ tanh.(A * r .+ W_in * readout_std(r))); nothing)

r_end = R_train[:, end]
predict_closed(tgrid) = begin
    sol = solve(ODEProblem(closed_rhs, copy(r_end), (tgrid[1], tgrid[end])),
                Tsit5(); saveat = tgrid, abstol = 1e-6, reltol = 1e-6)
    inverse_transform(scaler, (W_out * ((Array(sol) .- f_mu) ./ f_sd))')
end

X_pred = predict_closed(t_te)

# --- evaluation -------------------------------------------------------------
t_valid, t_valid_lyap = valid_prediction_time(test_data, X_pred, t_te)
println("LPCTESN closed-loop valid prediction time: $(round(t_valid, digits = 2)) s ",
        "($(round(t_valid_lyap, digits = 2)) Lyapunov times)")
println("RESULT method=LPCTESN seed=$SEED NR=$NR ",
        "t_valid_s=$(round(t_valid, digits = 3)) t_valid_lyap=$(round(t_valid_lyap, digits = 3))")

# --- figures ----------------------------------------------------------------
if save_figures
    X_climate = predict_closed(range(0.0, t_climate; step = dt))   # long climate run
    plot_forecast(t_te, test_data, X_pred, "figures/lorenz_LPCTESN.png";
                  t_valid = t_valid,
                  title = "LPCTESN reservoir (closed-loop valid for " *
                          "$(round(t_valid_lyap, digits = 1)) Lyapunov times)")
    plot_forecast_3d(test_data, X_pred, "figures/lorenz3d_LPCTESN.png";
                     title = "LPCTESN: closed-loop Lorenz forecast")
    plot_lorenz_map(train_data, X_climate, "figures/lorenz_map_LPCTESN.png")
    println("Figures: figures/lorenz_LPCTESN.png, lorenz3d_LPCTESN.png, lorenz_map_LPCTESN.png")
end
