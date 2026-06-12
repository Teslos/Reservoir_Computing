# Echo-state network with a neural-network readout forecasting the chaotic
# Lorenz-63 system. Identical reservoir to RC.jl; only the readout differs:
# a small MLP trained with Adam (explicit-style Flux API) instead of ridge
# regression. The forecast is closed-loop: the model output is fed back as
# the next reservoir input, with no access to the truth.
#
# Run from the repo root:  julia +1.11 --project=. Reservoir-Computing-in-Julia/RC_NN.jl

include("common.jl")
using .RCCommon
using LinearAlgebra
using Random
using Statistics
using Flux
using CairoMakie

# --- hyperparameters --------------------------------------------------------
const SEED = 42
dim_system = 3
dim_reservoir = 500
density = 0.02
spectral_radius = 0.9
sigma_in = 0.3
washout = 500
dt = 0.01
epochs = 400
batchsize = 256
learning_rate = 1e-3
noise_level = 0.02f0    # noise injected into reservoir states during training;
                        # stabilizes the closed loop against its own feedback errors

rng = MersenneTwister(SEED)
Random.seed!(SEED)      # Flux layer initialization
mkpath("figures")

# --- data ---------------------------------------------------------------------
train_data, t_train, test_data, t_test =
    generate_lorenz_split(t_train = 200.0, t_test = 25.0, dt = dt)
scaler = Standardizer(train_data)
u_train = transform(scaler, train_data)

# --- reservoir (teacher-forced; the A*r memory term stays in the loop) --------
A = generate_reservoir(rng, dim_reservoir, density;
                       spectral_radius = spectral_radius)
W_in = 2 * sigma_in * (rand(rng, dim_reservoir, dim_system) .- 0.5)
R = drive_reservoir(A, W_in, u_train)

# --- NN readout: state after sample i -> sample i+1 ----------------------------
X_feat = Float32.(R[:, washout:(end - 1)])
Y_targ = Float32.(u_train[(washout + 1):end, :]')

model = Chain(Dense(dim_reservoir => 128, tanh), Dense(128 => dim_system))
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
println("Final training MSE (standardized units): ",
        Flux.mse(model(X_feat), Y_targ))

# --- closed-loop forecast -------------------------------------------------------
readout(r) = Float64.(model(Float32.(r)))
pred_n, _ = closed_loop_forecast(A, W_in, readout, R[:, end], length(t_test))
X_predicted = inverse_transform(scaler, pred_n)

# --- evaluation -------------------------------------------------------------------
t_valid, t_valid_lyap = valid_prediction_time(test_data, X_predicted, t_test)
n_short = round(Int, 1 / (LORENZ_LYAPUNOV * dt))
mse_1lyap = mean(abs2, test_data[1:n_short, :] .- X_predicted[1:n_short, :])
println("Valid prediction time: $(round(t_valid, digits = 2)) s ",
        "($(round(t_valid_lyap, digits = 2)) Lyapunov times)")
println("MSE over the first Lyapunov time: ", mse_1lyap)

# --- long autonomous run for the attractor climate ---------------------------------
climate_n, _ = closed_loop_forecast(A, W_in, readout, R[:, end],
                                    round(Int, 100.0 / dt))
X_climate = inverse_transform(scaler, climate_n)

# --- plots ---------------------------------------------------------------------------
plot_forecast(t_test, test_data, X_predicted, "figures/lorenz_NN.png";
              t_valid = t_valid,
              title = "ESN + NN readout: closed-loop Lorenz forecast " *
                      "(valid for $(round(t_valid_lyap, digits = 1)) Lyapunov times)")
plot_forecast_3d(test_data, X_predicted, "figures/lorenz3d_NN.png")
plot_lorenz_map(train_data, X_climate, "figures/lorenz_map_NN.png")
println("Figures written to figures/lorenz_NN.png, figures/lorenz3d_NN.png, ",
        "figures/lorenz_map_NN.png")
