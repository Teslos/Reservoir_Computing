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
# `quad` augments the readout features with elementwise squares, [r; r^2]
# (Pathak et al. 2017). Lorenz-63's only nonlinearities are the products xy,
# xz; with the state ~affine in r, the squares span exactly those bilinear
# terms, so a *linear* readout on [r; r^2] can represent the dynamics nearly
# exactly. The closed loop stays a smooth ODE (r^2 is smooth), so the CTESN
# character is preserved -- this isolates "linear vs quadratic readout" from
# "discrete vs continuous reservoir".
quad = "quad" in ARGS
phi(r) = quad ? vcat(r, r .^ 2) : r           # readout feature map
# `partial` feeds only x(t) to the reservoir; y and z must be reconstructed
# from x-history via the fading-memory property (Takens embedding theorem).
partial = "partial" in ARGS
observe(x) = partial ? x[1:1] : x             # which components drive the reservoir

NR = quick ? 150 : 300          # reservoir size
dim_system = 3
spectral_radius = 1.1           # >1 is fine: the leaky continuous reservoir is
                                # still contracting for the echo-state property
density = 0.1
sigma_in = 0.5                  # input scaling (data is standardized)
leak = 1.0
speed = 10.0                    # reservoir time-scale factor; matches the
                                # reservoir response to the ~1 s Lorenz dynamics
beta_arg = findfirst(a -> startswith(a, "beta="), ARGS)
ridge_beta = beta_arg === nothing ? 0.1 : parse(Float64, ARGS[beta_arg][6:end])
                                # Tikhonov regularization (stabilizes the loop)
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
W_in = 2 * sigma_in .* (rand(rng, NR, partial ? 1 : dim_system) .- 0.5)

# --- teacher-forced continuous reservoir ------------------------------------
reservoir_rhs(input) = (dr, r, p, t) ->
    (dr .= speed .* (-leak .* r .+ tanh.(A * r .+ input(t))); nothing)

u_obs_train = partial ? u_train[:, 1:1]' : u_train'   # observed components
input_train = make_lerp(t_tr, W_in * u_obs_train)     # NR-vector current g(t)
sol_tr = solve(ODEProblem(reservoir_rhs(input_train), zeros(NR), (t_tr[1], t_tr[end])),
               Tsit5(); saveat = t_tr, abstol = 1e-6, reltol = 1e-6)
R_train = Array(sol_tr)                                # NR x n_train

# --- ridge readout on phi(r) = r  (or [r; r^2] when quad) --------------------
n_tr = size(R_train, 2)
F = quad ? vcat(R_train, R_train .^ 2)[:, washout:n_tr] : R_train[:, washout:n_tr]
f_mu = vec(mean(F, dims = 2)); f_sd = vec(std(F, dims = 2)) .+ 1e-8
Fs = (F .- f_mu) ./ f_sd .+ noise_level .* randn(rng, size(F))
Y = u_train[washout:n_tr, :]
W_out = ((Fs * Fs' + ridge_beta * I) \ (Fs * Y))'      # 3 x (NR or 2NR)
readout_std(r) = W_out * ((phi(r) .- f_mu) ./ f_sd)
println("Readout ($(quad ? "quadratic" : "linear")) training MSE (standardized): ",
        round(mean(abs2, W_out * ((F .- f_mu) ./ f_sd) .- Y'), digits = 5))

# --- closed-loop (autonomous) forecast: still a smooth ODE in r --------------
closed_rhs(dr, r, p, t) =
    (dr .= speed .* (-leak .* r .+ tanh.(A * r .+ W_in * observe(readout_std(r)))); nothing)

reconstruct(sol) = quad ?
    (W_out * ((vcat(Array(sol), Array(sol) .^ 2) .- f_mu) ./ f_sd))' :
    (W_out * ((Array(sol) .- f_mu) ./ f_sd))'

r_end = R_train[:, end]
predict_closed(tgrid) = begin
    sol = solve(ODEProblem(closed_rhs, copy(r_end), (tgrid[1], tgrid[end])),
                Tsit5(); saveat = tgrid, abstol = 1e-6, reltol = 1e-6)
    inverse_transform(scaler, reconstruct(sol))
end

X_pred = predict_closed(t_te)

# --- evaluation -------------------------------------------------------------
t_valid, t_valid_lyap = valid_prediction_time(test_data, X_pred, t_te)
println("LPCTESN closed-loop valid prediction time: $(round(t_valid, digits = 2)) s ",
        "($(round(t_valid_lyap, digits = 2)) Lyapunov times)")
println("RESULT method=LPCTESN$(quad ? "_quad" : "")$(partial ? "_partial" : "") seed=$SEED NR=$NR ",
        "t_valid_s=$(round(t_valid, digits = 3)) t_valid_lyap=$(round(t_valid_lyap, digits = 3))")

# --- figures ----------------------------------------------------------------
if save_figures
    sfx = (quad ? "_quad" : "") * (partial ? "_partial" : "")
    X_climate = predict_closed(range(0.0, t_climate; step = dt))   # long climate run
    plot_forecast(t_te, test_data, X_pred, "figures/lorenz_LPCTESN$(sfx).png";
                  t_valid = t_valid,
                  title = "LPCTESN$(quad ? " (quadratic readout)" : "") reservoir " *
                          "(closed-loop valid for $(round(t_valid_lyap, digits = 1)) Lyapunov times)")
    plot_forecast_3d(test_data, X_pred, "figures/lorenz3d_LPCTESN$(sfx).png";
                     title = "LPCTESN$(quad ? " (quadratic)" : ""): closed-loop Lorenz forecast")
    # model/panel: this is panel a) of the return-map figure in the manuscript,
    # paired with the FHN panel from RC_FHN_NN.jl
    plot_lorenz_map(train_data, X_climate, "figures/lorenz_map_LPCTESN$(sfx).png";
                    model = "LPCTESN, $(quad ? "quadratic" : "linear") readout ($NR nodes)",
                    panel = "a)")
    println("Figures: figures/lorenz_LPCTESN$(sfx).png, lorenz3d_LPCTESN$(sfx).png, ",
            "lorenz_map_LPCTESN$(sfx).png")
end
