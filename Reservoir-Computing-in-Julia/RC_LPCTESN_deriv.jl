# STRUCTURAL VARIANT of RC_LPCTESN.jl: derivative readout ("velocity form").
#
# Why: RC_LPCTESN.jl trains W_out as an OBSERVER -- it reconstructs u(t) from the
# reservoir state r(t) at the SAME time index -- and then closes the loop as an
# autonomous ODE in r alone. Nothing in the training constrains those autonomous
# dynamics, so closed-loop skill is whatever the r-ODE happens to do. Its valid
# prediction time sits at ~1.1 Lyapunov times and is insensitive to every
# hyperparameter (ridge 1e-6..0.1, noise 0..0.1, rho 0.5..1.5, speed 0.5..100),
# i.e. it is structural, not a tuning failure.
#
# The discrete ridge ESN it is compared against (NetworkDynamics
# src/baselines/baseline_models.jl `esn_lorenz`) is trained on a different
# problem: features [x_t; u_t; 1] -> target u_{t+1}. Because the current state
# u_t is passed through to the readout, that model only has to learn the
# INCREMENT u_{t+1} - u_t, and iterating it is exactly what it was trained for.
# It reaches ~6-7 Lyapunov times.
#
# This variant gives the continuous-time model the same advantage in the way that
# is well-defined for an ODE: learn the VECTOR FIELD, not the state, and let the
# closed loop integrate it. The forecast variable u becomes an explicit state, so
# the identity path is built in by integration rather than by a passthrough term.
#
#   training :  W_out : phi(r(t))  ->  d/dt u(t)      (standardized units)
#   closed   :  r' = speed*(-leak*r + tanh(A*r + W_in*u))
#               u' = W_out * phi_std(r)
#
# Everything else (reservoir, data split, metric, seed) is identical to
# RC_LPCTESN.jl, so the comparison isolates the readout target.
#
# Run from the repo root:
#   julia --project=. Reservoir-Computing-in-Julia/RC_LPCTESN_deriv.jl quad nofigs

include("common.jl")
using .RCCommon
using LinearAlgebra, Random, Statistics, OrdinaryDiffEq

seed_arg = findfirst(a -> tryparse(Int, a) !== nothing, ARGS)
SEED = seed_arg === nothing ? 42 : parse(Int, ARGS[seed_arg])
quick = "quick" in ARGS
quad  = "quad" in ARGS
phi(r) = quad ? vcat(r, r .^ 2) : r

argval(key, default) = begin
    i = findfirst(a -> startswith(a, key * "="), ARGS)
    i === nothing ? default : parse(Float64, ARGS[i][length(key)+2:end])
end

NR = quick ? 150 : Int(argval("NR", 300))
dim_system = 3
spectral_radius = argval("rho", 1.1)
density = 0.1
sigma_in = argval("sigma_in", 0.5)
leak = 1.0
speed = argval("speed", 10.0)
ridge_beta = argval("beta", 0.1)
noise_level = argval("noise", 1e-3)
washout = 500
dt = 0.01
t_train = quick ? 60.0 : 200.0
t_test  = quick ? 15.0 : 25.0

rng = MersenneTwister(SEED)
Random.seed!(SEED)

# --- data -------------------------------------------------------------------
train_data, t_tr, test_data, t_te = generate_lorenz_split(; t_train, t_test, dt)
scaler = Standardizer(train_data)
u_train = transform(scaler, train_data)          # n_train x 3, standardized

A = generate_reservoir(rng, NR, density; spectral_radius)
W_in = 2 * sigma_in .* (rand(rng, NR, dim_system) .- 0.5)

# --- teacher-forced continuous reservoir (identical to RC_LPCTESN.jl) --------
reservoir_rhs(input) = (dr, r, p, t) ->
    (dr .= speed .* (-leak .* r .+ tanh.(A * r .+ input(t))); nothing)

input_train = make_lerp(t_tr, W_in * u_train')
sol_tr = solve(ODEProblem(reservoir_rhs(input_train), zeros(NR), (t_tr[1], t_tr[end])),
               Tsit5(); saveat = t_tr, abstol = 1e-6, reltol = 1e-6)
R_train = Array(sol_tr)

# --- derivative targets: central differences of the standardized trajectory --
n_tr = size(R_train, 2)
Udot = similar(u_train)
Udot[2:end-1, :] = (u_train[3:end, :] .- u_train[1:end-2, :]) ./ (2dt)
Udot[1, :]   = (u_train[2, :]   .- u_train[1, :])     ./ dt
Udot[end, :] = (u_train[end, :] .- u_train[end-1, :]) ./ dt

F = quad ? vcat(R_train, R_train .^ 2)[:, washout:n_tr] : R_train[:, washout:n_tr]
f_mu = vec(mean(F, dims = 2)); f_sd = vec(std(F, dims = 2)) .+ 1e-8
Fs = (F .- f_mu) ./ f_sd .+ noise_level .* randn(rng, size(F))
Y = Udot[washout:n_tr, :]                        # <-- derivative, not state
W_out = ((Fs * Fs' + ridge_beta * I) \ (Fs * Y))'
phi_std(r) = (phi(r) .- f_mu) ./ f_sd
println("Derivative-readout training MSE (standardized units/s): ",
        round(mean(abs2, W_out * ((F .- f_mu) ./ f_sd) .- Y'), digits = 5))

# --- closed loop: augmented state z = [r; u], u integrated from the field ----
function closed_rhs(dz, z, p, t)
    r = @view z[1:NR]; u = @view z[NR+1:NR+3]
    dz[1:NR]        .= speed .* (-leak .* r .+ tanh.(A * r .+ W_in * u))
    dz[NR+1:NR+3]   .= W_out * phi_std(r)
    nothing
end

z0 = vcat(R_train[:, end], u_train[end, :])
sol = solve(ODEProblem(closed_rhs, z0, (t_te[1], t_te[end])), Tsit5();
            saveat = t_te, abstol = 1e-6, reltol = 1e-6)
U_pred_std = Array(sol)[NR+1:NR+3, :]'
X_pred = inverse_transform(scaler, U_pred_std)

t_valid, t_valid_lyap = valid_prediction_time(test_data, X_pred, t_te)
println("Derivative-readout LPCTESN valid prediction time: ",
        "$(round(t_valid, digits = 2)) s ($(round(t_valid_lyap, digits = 2)) Lyapunov times)")
println("RESULT method=LPCTESN_deriv$(quad ? "_quad" : "") seed=$SEED NR=$NR ",
        "speed=$speed beta=$ridge_beta ",
        "t_valid_s=$(round(t_valid, digits = 3)) t_valid_lyap=$(round(t_valid_lyap, digits = 3))")
