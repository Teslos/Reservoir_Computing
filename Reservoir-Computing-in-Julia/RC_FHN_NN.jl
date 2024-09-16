using OrdinaryDiffEq
using LinearAlgebra
using NNlib: swish
using Graphs
using NetworkDynamics
using Distributions
using Interpolations

# graph creation 
function create_complete_graph(N::Int=5)
    # create all to all graph
    g = Graphs.complete_graph(N)
    edge_weights = ones(length(edges(g)))
    g_weighted = SimpleDiGraph(g)
    g_directed = SimpleDiGraph(g_weighted)
    return g_directed, edge_weights
end

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

# create graph 
g_directed, edge_weights = create_complete_graph(8*8)
println("Number of nodes: ", nv(g_directed))
println("Number of edges: ", size(edge_weights))
println("Number of edges: ", ne(g_directed))

# generate the network
Base.@propagate_inbounds function fhn_electrical_vertex!(dv, v, edges, p, t)
    # add external input j0
    g = p

    #println("g size:",size(g))
    #println("g type:",typeof(g))
    # adding external input current
    if t < 0.5
        dv[1] = (v[1] + v[1]^3/3 - v[2])
    else
        dv[1] = (g[t] + v[1] - v[1]^3/3 - v[2])
    end
    # adding the external input voltage
    if t < 0.5
        dv[2] = (v[1] + a)*ϵ
    else
        dv[2] = R0 .* g[t] + (v[1] + a)*ϵ
    end
    
    for e in edges
        dv[1] += e[1]
        dv[2] += e[2]
    end
    nothing
end

Base.@propagate_inbounds function fhn_electrical_vertex_win!(dv, v, edges, p, t)
    # add external input j0
    w = p

    #println("g size:",size(g))
    #println("g type:",typeof(g))
    # adding external input current
    if t < 0.5
        dv[1] = (v[1] + v[1]^3/3 - v[2])
    else
        dv[1] = (v[1] - v[1]^3/3 - v[2])
    end
    # adding the external input voltage
    if t < 0.5
        dv[2] = (v[1] + a)*ϵ
    else
        dv[2] = R0 .* g[t] + (v[1] + a)*ϵ
    end
    
    for e in edges
        dv[1] += e[1]
        dv[2] += e[2]
    end
    nothing
end
# set the B rotational matrix with an angle ϕ,
# the default value is ϕ = π/2 - 0.1, but the value causes the numerics to be unstable
ϕ = π/2 - 0.1
B = [cos(ϕ) sin(ϕ); -sin(ϕ) cos(ϕ)]
Base.Base.@propagate_inbounds function electrical_edge!(e, v_s, v_d, p, t)
    #println("p type:",typeof(p))
    #println("v_s size:",size(v_s))
    #println("v_d size:",size(v_d))
    e[1] = p*(B[1,1]*(v_s[1] - v_d[1]) + B[1,2]*(v_s[2] - v_d[2])) # *σ  - edge coupling for current
    e[2] = p*(B[2,1]*(v_s[1] - v_d[1]) + B[2,2]*(v_s[2] - v_d[2])) # *σ  - edge coupling for voltage
    nothing
end


odeelevertex = ODEVertex(; f=fhn_electrical_vertex!, dim=2, sym=[:u, :v])
odeeleedge = StaticEdge(; f=electrical_edge!, dim=2, coupling=:directed)

fhn_network! = network_dynamics(odeelevertex, odeeleedge, g_directed)



# generate the training data for lorenz system
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
dim_reservoir = 64*2

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

# Parameter handling
N = nv(g_directed) # Number of nodes in the network
const ϵ = 0.05 # time scale separation parameter, default value is 0.05
const a = 0.5 # threshold parameter abs(a) < 1 is self-sustained limit cycle, abs(a) = 1 is a Hopf bifurcation
const σ = 0.5
const R0 = 0.1
# different weights for edges, because the resitivity of the edges are always positive
w_ij = [pdf(Normal(), x) for x in range(-1, 1, length=ne(g_directed))]
#w_ij = σ/ϵ * ones(ne(g_directed))
#g0 = interpolate((W_in*train_data')', BSpline(Quadratic(Line(OnCell()))))
g0v = [interpolate((W_in*train_data')[i,:], BSpline(Quadratic(Line(OnCell())))) for i in 1:nv(g_directed)] 
# Tuple of parameters for nodes and edges
p = (g0v, w_ij)
#Initial conditions
x0 = randn(2N)

# Solving the ODE
using OrdinaryDiffEq

tspan = (0.0, 300)
datasize = length(t)
tsteps = range(tspan[1], tspan[2], length=datasize)
prob = ODEProblem(fhn_network!, x0, tspan, p)
sol = solve(prob, Tsit5(), saveat=tsteps)

using GLMakie
fig = Figure()
ax = GLMakie.Axis(fig[1, 1], xlabel = "Time", ylabel = "u", title = "FitzHugh-Nagumo network")
t= sol.t
u = hcat(sol.u...)
# use only the u values
diff_data = u
for i in 1:64
    lines!(ax, t, u[i,:], label="Oscillator $i")
    #text!(ax, t[end], u[i,end]+0.1, text=string("Oscillator ", i), align=(:right, :center))
end
#axislegend(ax, position = :rt)
fig
GLMakie.save("RC_FHN_NN.png", fig)

# instead of ridge regression, we can use neural network to fit the output weights
using Flux
using Flux: crossentropy, onecold, onehotbatch, params, mse
using Statistics: mean

# using the neural network to fit output weights
model = Chain(Dense(dim_reservoir, 100, swish), Dense(100, dim_system))
loss(x, y) = mse(model(x), y')
opt = ADAM(0.01)
data = [(u, train_data)]
Flux.train!(loss, params(model), data, opt)
R_test = zeros(dim_reservoir, length(t2))
# get prediction of the model for test data
r_state = 1.0 ./ (1 .+ exp.(-(A*r_state + W_in*IC_validate)))
for t in 1:length(t2)
    R_test[:,t] = r_state
    r_state = swish.(A*r_state + W_in*test_data[t,:])
end



X_predicted = model(R_test)
X_predicted = X_predicted'
# using the ridge regression to fit output weights
#W_out = (train_data'*R')*inv( (R*R') + beta * I(dim_reservoir) )

#=
X_predicted = zeros(length(t2), dim_system)
r_state = 1.0 ./ (1 .+ exp.(-(A*r_state + W_in*IC_validate)))

for i in 1:length(t2)
    predict = model(r_state)
    X_predicted[i,:] = predict'
    #r_state = 1.0 ./ (1 .+ exp.(-(A*r_state + W_in*X_predicted[i,:])))
    #r_state = 1.0 .+ tanh.(A*r_state + W_in*X_predicted[i,:])
    r_state = swish.(A*r_state + W_in*X_predicted[i,:])
end
=#

# plot the results
using GLMakie

fig = Figure()
ax1 = Axis(fig[1, 1])
ax2 = Axis(fig[2, 1])
ax3 = Axis(fig[3, 1])
lines!(ax1, t2, test_data[:,1], color = :blue)
lines!(ax1, t2, 10*X_predicted[1:length(t2),1], color = :red)

lines!(ax2, t2, test_data[:,2], color = :green)
lines!(ax2, t2, 10*X_predicted[1:length(t2),2], color = :orange)

lines!(ax3, t2, test_data[:,3], color = :purple)
lines!(ax3, t2, 10*X_predicted[1:length(t2),3], color = :yellow)

fig
GLMakie.save("lorenz.png", fig)

# Create a new figure
fig = Figure(resolution = (1600, 1200))

# 3D plot
ax = Axis3(fig[1, 1], title = "Predicting Lorenz 63", xlabel = "x", ylabel = "y", zlabel = "z")
lines!(ax, test_data[:, 1], test_data[:, 2], test_data[:, 3], color = :blue, label = "True")
lines!(ax, 10X_predicted[:, 1], 10X_predicted[:, 2], 10X_predicted[:, 3], color = :red, label = "Predicted")

# Add grid and legend
axislegend(ax)

# Display the figure
fig
GLMakie.save("lorenz3d.png", fig)