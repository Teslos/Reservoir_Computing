using OrdinaryDiffEq
using LinearAlgebra
using NNlib: swish

function generate_reservoir(dim_reservoir, density)
    A = rand(dim_reservoir, dim_reservoir)
    A = A .< density
    ran = 2*(rand(dim_reservoir, dim_reservoir) .- 0.5)
    A = A.*ran
    # get eigenvalues of A
    eigA = eigvals(A)
    # set spectral radius of A to 1
    A = A./maximum(abs.(eigA))
    return A
end

# generate the training data
function lorenz!(dx, x, p, t)
    sigma, rho, beta = p
    dx[1] = sigma*(x[2] - x[1])
    dx[2] = x[1]*(rho - x[3]) - x[2]
    dx[3] = x[1]*x[2] - beta*x[3]
end

u0 = [10;10;10]
tspan = (0.0, 300.0)
dt = 0.01
p = (10.0, 28.0, 8/3) # sigma, rho, beta values
prob = ODEProblem(lorenz!, u0, tspan, p)
sol  = solve(prob, Tsit5(), saveat = dt, progress = true)
train_data = hcat(sol.u...)'
t = sol.t

# generate the test data
IC_validate = [10.1; 10.0; 10.0]
tspan2 = (0.0, 50.0)
prob2 = ODEProblem(lorenz!, IC_validate, tspan2, p)
sol2 = solve(prob2, Tsit5(), saveat = dt, progress = true)
test_data = hcat(sol2.u...)'
t2 = sol2.t

dim_system = 3
dim_reservoir = 500

sigma = 0.1
density = 0.05 # density of the reservoir
beta = 0.01 # regularization parameter

r_state = zeros(dim_reservoir)
A = generate_reservoir(dim_reservoir, density)
W_in = 2*sigma*(rand(dim_reservoir, dim_system) .- 0.5)

W_out = zeros(dim_system, dim_reservoir)
R = zeros(dim_reservoir, length(t))

for i in 1:length(t)
    R[:,i] = r_state
    #r_state = 1.0 ./ (1 .+ exp.(-(A*r_state + W_in*train_data[i,:])))
    #r_state = 1.0 .+ tanh.(A*r_state + W_in*train_data[i,:])
    r_state = swish.(A*r_state + W_in*train_data[i,:])
end

# using the ridge regression to fit output weights
W_out = (train_data'*R')*inv( (R*R') + beta * I(dim_reservoir) )

X_predicted = zeros(length(t2), dim_system)
r_state = 1.0 ./ (1 .+ exp.(-(A*r_state + W_in*IC_validate)))

for i in 1:length(t2)
    X_predicted[i,:] = W_out*r_state
    #r_state = 1.0 ./ (1 .+ exp.(-(A*r_state + W_in*X_predicted[i,:])))
    #r_state = 1.0 .+ tanh.(A*r_state + W_in*X_predicted[i,:])
    r_state = swish.(A*r_state + W_in*X_predicted[i,:])
end

# mean squared error
MSE = sum((test_data .- X_predicted).^2)/length(t2)
println("MSE: ", MSE)
# plot the results
using GLMakie

fig = Figure()
ax1 = Axis(fig[1, 1])
ax2 = Axis(fig[2, 1])
ax3 = Axis(fig[3, 1])
lines!(ax1, t2, test_data[:,1], color = :blue)
lines!(ax1, t2, X_predicted[:,1], color = :red)

lines!(ax2, t2, test_data[:,2], color = :green)
lines!(ax2, t2, X_predicted[:,2], color = :orange)

lines!(ax3, t2, test_data[:,3], color = :purple)
lines!(ax3, t2, X_predicted[:,3], color = :yellow)

fig
GLMakie.save("lorenz_RC.png", fig)

# Create a new figure
fig = Figure(resolution = (800, 600))

# 3D plot
ax = Axis3(fig[1, 1], title = "Predicting Lorenz 63", xlabel = "x", ylabel = "y", zlabel = "z")
lines!(ax, test_data[:, 1], test_data[:, 2], test_data[:, 3], color = :blue, label = "True")
lines!(ax, X_predicted[:, 1], X_predicted[:, 2], X_predicted[:, 3], color = :red, label = "Predicted")

# Add grid and legend
axislegend(ax)

# Display the figure
fig
GLMakie.save("lorenz3d_RC.png", fig)