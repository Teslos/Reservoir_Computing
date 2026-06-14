# Echo-state network with ridge-regression readout forecasting the chaotic
# Lorenz-63 system. The forecast is closed-loop (autonomous): after training,
# the model's own output is fed back as input, with no access to the truth.
#
# Run from the repo root:
#   julia +1.11 --project=. Reservoir-Computing-in-Julia/RC.jl [seed] [nofigs]

include("common.jl")
using .RCCommon
using LinearAlgebra
using Random
using Statistics
using CairoMakie

# --- hyperparameters --------------------------------------------------------
seed_arg = findfirst(a -> tryparse(Int, a) !== nothing, ARGS)
SEED = seed_arg === nothing ? 42 : parse(Int, ARGS[seed_arg])
save_figures = !("nofigs" in ARGS)
dim_system = 3
dim_reservoir = 500
density = 0.02          # average in-degree 10
spectral_radius = 0.9
sigma_in = 0.3          # input scaling (data is standardized)
ridge_beta = 1e-6       # regularization parameter
washout = 500           # discard the first 5 s of reservoir transients
dt = 0.01

rng = MersenneTwister(SEED)
mkpath("figures")

# --- data: one continuous trajectory, test continues training ---------------
train_data, t_train, test_data, t_test =
    generate_lorenz_split(t_train = 200.0, t_test = 25.0, dt = dt)
scaler = Standardizer(train_data)
u_train = transform(scaler, train_data)
u_test = transform(scaler, test_data)

# --- reservoir --------------------------------------------------------------
A = generate_reservoir(rng, dim_reservoir, density;
                       spectral_radius = spectral_radius)
W_in = 2 * sigma_in * (rand(rng, dim_reservoir, dim_system) .- 0.5)

# Drive the reservoir through the training data (teacher forcing).
# R[:, i] is the state after consuming u_train[i, :].
R = drive_reservoir(A, W_in, u_train)

# --- ridge regression readout ------------------------------------------------
# Quadratic feature augmentation [r; r.^2] (Pathak et al. 2017) breaks the
# r -> -r symmetry of tanh reservoirs and markedly improves Lorenz forecasts.
features(r) = vcat(r, r .^ 2)

Phi = vcat(R, R .^ 2)[:, washout:(end - 1)]   # predicts the *next* sample
Y = u_train[(washout + 1):end, :]
# W_out = Y' Phi' (Phi Phi' + beta I)^-1, solved with \ instead of inv()
W_out = ((Phi * Phi' + ridge_beta * I) \ (Phi * Y))'

train_mse = mean(abs2, W_out * Phi .- Y')
println("Readout training MSE (standardized units): ", train_mse)

# --- closed-loop forecast ----------------------------------------------------
# Start from the synchronized state at the end of training and run fully
# autonomously: each prediction is fed back as the next input.
readout(r) = W_out * features(r)
pred_n, _ = closed_loop_forecast(A, W_in, readout, R[:, end], length(t_test))
X_predicted = inverse_transform(scaler, pred_n)

# --- evaluation ---------------------------------------------------------------
t_valid, t_valid_lyap = valid_prediction_time(test_data, X_predicted, t_test)
n_short = round(Int, 1 / (LORENZ_LYAPUNOV * dt))   # one Lyapunov time
mse_1lyap = mean(abs2, test_data[1:n_short, :] .- X_predicted[1:n_short, :])
println("Valid prediction time: $(round(t_valid, digits = 2)) s ",
        "($(round(t_valid_lyap, digits = 2)) Lyapunov times)")
println("MSE over the first Lyapunov time: ", mse_1lyap)
println("RESULT method=ESN_ridge seed=$SEED NR=$dim_reservoir ",
        "t_valid_s=$(round(t_valid, digits = 3)) t_valid_lyap=$(round(t_valid_lyap, digits = 3))")

# --- plots ---------------------------------------------------------------------
if save_figures
    # long autonomous run for the attractor climate
    climate_n, _ = closed_loop_forecast(A, W_in, readout, R[:, end],
                                        round(Int, 100.0 / dt))
    X_climate = inverse_transform(scaler, climate_n)
    plot_forecast(t_test, test_data, X_predicted, "figures/lorenz_RC.png";
                  t_valid = t_valid,
                  title = "ESN + ridge: closed-loop Lorenz forecast " *
                          "(valid for $(round(t_valid_lyap, digits = 1)) Lyapunov times)")
    plot_forecast_3d(test_data, X_predicted, "figures/lorenz3d_RC.png")
    plot_lorenz_map(train_data, X_climate, "figures/lorenz_map_RC.png")
    println("Figures written to figures/lorenz_RC.png, figures/lorenz3d_RC.png, ",
            "figures/lorenz_map_RC.png")
end
