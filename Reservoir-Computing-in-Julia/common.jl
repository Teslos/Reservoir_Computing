# Shared utilities for the reservoir-computing Lorenz experiments.
# Used by RC.jl (ridge readout), RC_NN.jl (NN readout) and RC_FHN_NN.jl
# (FitzHugh-Nagumo oscillator network as a physical reservoir).
module RCCommon

using OrdinaryDiffEq
using LinearAlgebra
using Random
using Statistics
using CairoMakie   # also re-exports @L_str, used for the Lorenz-map axis labels

export lorenz!, generate_lorenz_split, generate_reservoir, drive_reservoir,
       closed_loop_forecast, Standardizer, transform, inverse_transform,
       find_maxima_in_z, valid_prediction_time, LORENZ_LYAPUNOV,
       make_lerp, plot_forecast, plot_forecast_3d, plot_lorenz_map

# Largest Lyapunov exponent of the Lorenz system at sigma=10, rho=28, beta=8/3.
# 1/LORENZ_LYAPUNOV ~ 1.1 s is one "Lyapunov time".
const LORENZ_LYAPUNOV = 0.9056

function lorenz!(dx, x, p, t)
    sigma, rho, beta = p
    dx[1] = sigma * (x[2] - x[1])
    dx[2] = x[1] * (rho - x[3]) - x[2]
    dx[3] = x[1] * x[2] - beta * x[3]
end

"""
    generate_lorenz_split(; t_train, t_test, dt, u0, p, t_transient)

Generate one continuous Lorenz trajectory sampled at `dt` and split it into a
training segment and a test segment that directly continues it. Because the
test segment is the continuation of the training trajectory, a reservoir that
was driven through the training data is already synchronized when the
closed-loop forecast starts -- no separate warm-up pass is needed.
The initial `t_transient` seconds are discarded so all data lies on the
attractor. Returns `(train_data, t_train, test_data, t_test)`, data as N x 3.
"""
function generate_lorenz_split(; t_train = 200.0, t_test = 25.0, dt = 0.01,
                               u0 = [10.0, 10.0, 10.0],
                               p = (10.0, 28.0, 8 / 3),
                               t_transient = 20.0)
    tspan = (0.0, t_transient + t_train + t_test)
    prob = ODEProblem(lorenz!, u0, tspan, p)
    sol = solve(prob, Tsit5(); saveat = dt, abstol = 1e-9, reltol = 1e-9)
    data = Array(sol)'                     # N x 3, avoids splatting hcat(sol.u...)
    t = sol.t
    i0 = findfirst(>=(t_transient), t)
    i1 = findfirst(>=(t_transient + t_train), t)
    train = data[i0:i1, :]
    t_tr = t[i0:i1] .- t[i0]
    test = data[(i1 + 1):end, :]
    t_te = t[(i1 + 1):end] .- t[i1]
    return train, t_tr, test, t_te
end

"""
    generate_reservoir(rng, dim, density; spectral_radius)

Random sparse reservoir matrix with entries uniform in [-1, 1], rescaled to
the requested spectral radius. Takes an explicit RNG for reproducibility.
"""
function generate_reservoir(rng::AbstractRNG, dim::Int, density::Float64;
                            spectral_radius::Float64 = 0.9)
    A = (rand(rng, dim, dim) .< density) .* (2 .* rand(rng, dim, dim) .- 1)
    rho = maximum(abs.(eigvals(A)))
    return A .* (spectral_radius / rho)
end

"""
    drive_reservoir(A, W_in, data; f, leak, r0)

Run the discrete reservoir `r <- (1-leak) r + leak f(A r + W_in x)` over the
rows of `data` (teacher forcing). Returns `R` where column `i` is the state
*after* consuming `data[i, :]`, so column `i` is the feature vector for
predicting `data[i+1, :]`.
"""
function drive_reservoir(A::AbstractMatrix, W_in::AbstractMatrix,
                         data::AbstractMatrix;
                         f = tanh, leak::Float64 = 1.0,
                         r0::AbstractVector = zeros(size(A, 1)))
    n = size(data, 1)
    R = zeros(size(A, 1), n)
    r = copy(r0)
    for i in 1:n
        r = (1 - leak) .* r .+ leak .* f.(A * r .+ W_in * data[i, :])
        R[:, i] = r
    end
    return R
end

"""
    closed_loop_forecast(A, W_in, readout, r0, n_steps; f, leak)

Run the discrete reservoir autonomously for `n_steps`: at every step the
`readout` function maps the reservoir state to a system-state prediction,
which is fed back as the next input. The truth is never used. Returns the
n_steps x dim predictions and the final reservoir state.
"""
function closed_loop_forecast(A::AbstractMatrix, W_in::AbstractMatrix,
                              readout, r0::AbstractVector, n_steps::Int;
                              f = tanh, leak::Float64 = 1.0)
    r = copy(r0)
    x = readout(r)
    pred = zeros(n_steps, length(x))
    for i in 1:n_steps
        pred[i, :] = x
        r = (1 - leak) .* r .+ leak .* f.(A * r .+ W_in * x)
        x = readout(r)
    end
    return pred, r
end

# --- data normalization -----------------------------------------------------

struct Standardizer
    mu::Vector{Float64}
    sigma::Vector{Float64}
end

Standardizer(X::AbstractMatrix) =
    Standardizer(vec(mean(X, dims = 1)), vec(std(X, dims = 1)))

transform(s::Standardizer, X::AbstractMatrix) = (X .- s.mu') ./ s.sigma'
inverse_transform(s::Standardizer, X::AbstractMatrix) = X .* s.sigma' .+ s.mu'

# --- evaluation -------------------------------------------------------------

"Successive local maxima of the z-coordinate of an N x 3 trajectory."
function find_maxima_in_z(trj::AbstractMatrix{<:Real})
    z = trj[:, 3]
    is_max = (z[2:(end - 1)] .> z[1:(end - 2)]) .& (z[2:(end - 1)] .> z[3:end])
    return z[2:(end - 1)][is_max]
end

"""
    valid_prediction_time(truth, pred, t; threshold)

Time until the normalized forecast error
`||x_true - x_pred|| / sqrt(<||x_true - <x_true>||^2>)` first exceeds
`threshold`. Returns `(t_valid_seconds, t_valid_lyapunov_times)`. This is the
standard metric for chaotic forecasts; long-horizon MSE is meaningless because
trajectory divergence is guaranteed.
"""
function valid_prediction_time(truth::AbstractMatrix, pred::AbstractMatrix,
                               t::AbstractVector; threshold = 0.4)
    scale = sqrt(mean(sum(abs2, truth .- mean(truth, dims = 1), dims = 2)))
    err = sqrt.(vec(sum(abs2, truth .- pred, dims = 2))) ./ scale
    idx = findfirst(>(threshold), err)
    t_valid = idx === nothing ? t[end] - t[1] : t[idx] - t[1]
    return t_valid, t_valid * LORENZ_LYAPUNOV
end

"""
    make_lerp(t, G)

Linear interpolation closure for the columns of `G` (size N x length(t)) over
the uniform time grid `t`. Used to drive a continuous-time reservoir with
discretely sampled input.
"""
function make_lerp(t::AbstractVector, G::AbstractMatrix)
    dt = t[2] - t[1]
    n = length(t)
    return tau -> begin
        s = clamp((tau - t[1]) / dt + 1, 1.0, Float64(n))
        i = min(floor(Int, s), n - 1)
        fr = s - i
        @views G[:, i] .* (1 - fr) .+ G[:, i + 1] .* fr
    end
end

# --- plotting ---------------------------------------------------------------

"""
    save_vector(path, fig)

Save `fig` at `path` and, when `path` is a raster format, additionally emit a
vector PDF alongside it. Figures that end up in the manuscript are line art, so
the PDF is what should be included there; the PNG is kept for quick viewing.
"""
function save_vector(path::AbstractString, fig)
    save(path, fig)
    endswith(lowercase(path), ".png") && save(replace(path, r"(?i)\.png$" => ".pdf"), fig)
    return fig
end

"""
    plot_forecast(t, truth, pred, path; pred_open_loop, t_valid, title)

Three stacked panels (x, y, z) comparing truth and the closed-loop forecast.
An optional teacher-forced (open-loop) prediction is drawn dashed, and the
valid-prediction-time is marked with a vertical line.
"""
function plot_forecast(t, truth, pred, path::AbstractString;
                       pred_open_loop = nothing, t_valid = nothing,
                       title = "Closed-loop Lorenz forecast")
    fig = Figure(size = (1000, 700))
    labels = ["x", "y", "z"]
    for k in 1:3
        ax = Axis(fig[k, 1], ylabel = labels[k],
                  xlabel = k == 3 ? "Time (s)" : "",
                  title = k == 1 ? title : "")
        lines!(ax, t, truth[:, k], color = :black, label = "True")
        lines!(ax, t, pred[:, k], color = :red, label = "Closed-loop forecast")
        if pred_open_loop !== nothing
            lines!(ax, t, pred_open_loop[:, k], color = (:blue, 0.6),
                   linestyle = :dash, label = "Open-loop (teacher-forced)")
        end
        if t_valid !== nothing
            vlines!(ax, [t[1] + t_valid], color = :gray, linestyle = :dot)
        end
        k == 1 && axislegend(ax, position = :rt, framevisible = false)
    end
    save_vector(path, fig)
    return fig
end

function plot_forecast_3d(truth, pred, path::AbstractString;
                          title = "Predicting Lorenz 63")
    fig = Figure(size = (800, 600))
    ax = Axis3(fig[1, 1], title = title, xlabel = "x", ylabel = "y", zlabel = "z")
    lines!(ax, truth[:, 1], truth[:, 2], truth[:, 3], color = (:blue, 0.7),
           label = "True")
    lines!(ax, pred[:, 1], pred[:, 2], pred[:, 3], color = (:red, 0.7),
           label = "Predicted")
    axislegend(ax)
    save_vector(path, fig)
    return fig
end

"""
    plot_lorenz_map(truth_traj, pred_traj, path)

Return map of successive z-maxima (the Lorenz map). A forecast that diverges
pointwise can still reproduce this "climate" of the attractor.
"""
function plot_lorenz_map(truth_traj, pred_traj, path::AbstractString)
    fig = Figure(size = (700, 600))
    # LaTeX strings: as plain strings the axis labels rendered the underscore
    # literally ("z_n", "z_n+1") instead of subscripting it.
    ax = Axis(fig[1, 1], title = "Lorenz map (successive z-maxima)",
              xlabel = L"z_n", ylabel = L"z_{n+1}")
    mt = find_maxima_in_z(truth_traj)
    mp = find_maxima_in_z(pred_traj)
    scatter!(ax, mt[1:(end - 1)], mt[2:end], color = (:black, 0.6),
             markersize = 7, label = "True")
    scatter!(ax, mp[1:(end - 1)], mp[2:end], color = (:red, 0.6),
             markersize = 7, label = "Predicted")
    axislegend(ax, position = :lt)
    save_vector(path, fig)
    return fig
end

end # module
