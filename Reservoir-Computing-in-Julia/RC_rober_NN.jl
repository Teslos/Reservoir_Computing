using OrdinaryDiffEq
using LinearAlgebra
using NNlib: swish
using Graphs
using NetworkDynamics
using Distributions
using Interpolations
using Dierckx


# graph creation 
function create_barabasi_albert_graph(N::Int=5, k::Int=2)
    g = barabasi_albert(N, k, is_directed=true)
    edge_weights = ones(length(edges(g)))
    g_weighted = SimpleDiGraph(g)
    g_directed = SimpleDiGraph(g_weighted)
    return g_directed, edge_weights
end
# create the grid graph
function create_graph(N::Int=8, M::Int=8)
    g = Graphs.grid([N, M])
    edge_weights = ones(length(edges(g)))
    g_weighted = SimpleDiGraph(g)
    g_directed = SimpleDiGraph(g_weighted)
    return g_directed, edge_weights
end

function create_watts_strogatz_graph(N::Int=5, k::Int=2, p::Float64=0.5)
    g = watts_strogatz(N, k, p)
    edge_weights = ones(length(edges(g)))
    g_weighted = SimpleDiGraph(g)
    g_directed = SimpleDiGraph(g_weighted)
    return g_directed, edge_weights
end

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

# define original stiff ODE system
function robertson!(du, u, p, t)
	y₁, y₂, y₃ = u
	k₁, k₂, k₃ = p
	du[1] = -k₁ * y₁ + k₃ * y₂ * y₃
	du[2] =  k₁ * y₁ - k₂ * y₂^2 - k₃ * y₂ * y₃
	du[3] =  k₂ * y₂^2
	return nothing
end

# create graph 
g_directed, edge_weights = create_complete_graph(8*8)
#g_directed, edge_weights = create_barabasi_albert_graph(8*8, 12)
#g_directed, edge_weights = create_graph(8, 8)
#g_directed, edge_weights = create_watts_strogatz_graph(16*16, 32*2, 0.25)

println("Number of nodes: ", nv(g_directed))
println("Number of edges: ", size(edge_weights))
println("Density of graph: ", Graphs.density(g_directed))

# generate the network
@inline Base.@propagate_inbounds function fhn_electrical_vertex_simple!(dv, v, edges, p, t)
    g = p
    e_s, e_d = edges
    dv[1] = g(t) + v[1] - v[1]^3 / 3 - v[2]
    dv[2] = (g(t) .* R0 + v[1] - a) * ϵ
    for e in e_s
        dv[1] -= e[1]
    end
    for e in e_d
        dv[1] += e[1]
    end
    nothing
end

@inline Base.@propagate_inbounds function electrical_edge_simple!(e, v_s, v_d, p, t)
    e[1] =  p * (v_s[1] - v_d[1]) # * σ
    nothing
end

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
    if t < 300
        dv[1] = (v[1] - v[1]^3/3 - v[2]) * 1/ϵ
    else
        dv[1] = (g[t] + v[1] - v[1]^3/3 - v[2]) * 1/ϵ
    end
    # adding the external input voltage
    if t < 300
        dv[2] = v[1] + a
    else
        dv[2] = R0 .* g[t] + v[1] + a
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
#B = [0.25 0.25; -0.25 0.25]
Base.Base.@propagate_inbounds function electrical_edge!(e, v_s, v_d, p, t)
    #println("p type:",typeof(p))
    #println("v_s size:",size(v_s))
    #println("v_d size:",size(v_d))
    e[1] = p*(B[1,1]*(v_s[1] - v_d[1]) + B[1,2]*(v_s[2] - v_d[2])) # *σ  - edge coupling for current
    e[2] = p*(B[2,1]*(v_s[1] - v_d[1]) + B[2,2]*(v_s[2] - v_d[2])) # *σ  - edge coupling for voltage
    nothing
end


odeelevertex = ODEVertex(; f=fhn_electrical_vertex_simple!, dim=2, sym=[:u, :v])
odeeleedge = StaticEdge(; f=electrical_edge_simple!, dim=2, coupling=:directed)

fhn_network! = network_dynamics(odeelevertex, odeeleedge, g_directed)



# generate the training data for Robertson system
p0 = (0.04, 3e7, 1e7)
u0 = [0.7; 0.2; 0.3]; u0 = u0./sum(u0) # to enforce sum(u) == 1.0
tspan = (0.0, 1e7)

tt = 10.0.^collect(range(-5.0, +5.0; length=33))
modelODESize = length(u0)
# generate the training data for the robertson system

# define ODEProblem, optimize it, and solve for original ODE solution at (u0, p0)
prob = ODEProblem(robertson!, u0, tspan, p0)
sol_rob = solve(prob, Rosenbrock23(); abstol=1e-6, reltol=1e-6, saveat=tt)
using Plots
# plot the solution of the robertson system
Plots.plot(sol_rob, xscale=:log, yscale=:log, label=["y1" "y2" "y3"], xlabel="Time", ylabel="Concentration", title="Robertson System Solution")
# generate the training data
trober = sol.t
train_data = hcat(sol.u...)' # train_data is the solution of the Rober system

# generate the testing data for the robertson system
p0_test = (0.04, 3e7, 1.1e7)
u0_test = [0.7; 0.2; 0.3]; u0_test = u0_test./sum(u0_test) # to enforce sum(u) == 1.0
tspan_test = (0.0, 1e5)
tt_test = 10.0.^collect(range(-5.0, +5.0; length=33))
# define ODEProblem, optimize it, and solve for original ODE solution at (u0_test, p0_test)
prob_test = ODEProblem(robertson!, u0_test, tspan_test, p0_test)
sol_test = solve(prob_test, Rosenbrock23(); abstol=1e-6, reltol=1e-6, saveat=tt)

# plot the solution of the robertson system
Plots.plot(sol_test, xscale=:log, yscale=:log, label=["y1" "y2" "y3"], xlabel="Time", ylabel="Concentration", title="Robertson System Test Solution")
# generate the testing data
ttest = sol_test.t
test_data = hcat(sol_test.u...)' # test_data is the solution of the Rober system
dim_system = 3
dim_reservoir = 2*64

sigma = 0.1 # input scaling
density = 0.05 # density of the reservoir
beta = 0.01 # regularization parameter

r_state = zeros(dim_reservoir)
A = generate_reservoir(dim_reservoir, density)
W_in = 2*sigma*(rand(dim_reservoir, dim_system) .- 0.5)

W_out = zeros(dim_system, dim_reservoir)
R = zeros(dim_reservoir, length(tlorenz))

# Parameter handling
N = nv(g_directed) # Number of nodes in the network
const ϵ = 0.05 # time scale separation parameter, default value is 0.05
const a = 0.5 # threshold parameter abs(a) < 1 is self-sustained limit cycle, abs(a) = 1 is a Hopf bifurcation
const σ = 0.06 # coupling strength
const R0 = 0.5
# different weights for edges, because the resitivity of the edges are always positive
w_ij = [pdf(Normal(), x) for x in range(-1, 1, length=ne(g_directed))]
gs = [Spline1D(trober, (W_in*train_data')[i,:], k=2) for i in 1:nv(g_directed)]
# Tuple of parameters for nodes and edges
p = (gs,σ * w_ij)
# Initial conditions
x0 = W_in*u0
# Solving the ODE
using OrdinaryDiffEq

tspan = (0.0, 1e5)

prob = ODEProblem(fhn_network!, x0, tspan, p)
sol = solve(prob, Tsit5(), saveat=tt, progress=true)

using GLMakie
fig = Figure()
ax = GLMakie.Axis(fig[1, 1], xlabel = "Time", ylabel = "u", title = "FitzHugh-Nagumo network")
t= sol.t
u = sol[:,:]
# use only the u values
diff_data = u
for i in 1:64
    lines!(ax, t, u[i,:], label="Oscillator $i")
    #GLMakie.heatmap!(ax, t, i*ones(length(t)), sol[i,:], colormap = :viridis)
    #text!(ax, t[end], u[i,end]+0.1, text=string("Oscillator ", i), align=(:right, :center))
end
#axislegend(ax, position = :rt)
fig
GLMakie.save("RC_rober_NN.png", fig)

# instead of ridge regression, we can use neural network to fit the output weights
using Flux
using Lux
using LuxCUDA
using Flux: crossentropy, onecold, onehotbatch, params, mse
using Flux: DataLoader
using Statistics: mean
using Random
using Optimisers
using MLDataUtils
using Zygote


using FluxOptTools
using Optim 
# using the neural network to fit output weights
model = Flux.Chain(Flux.Dense(128, 256, swish), Flux.Dense(256, dim_system))
loss(model) = mean(abs2, model(u) .- train_data')
opt = Flux.Adam(0.01)

Zygote.refresh()
ps = Flux.params(model)
lossfun, gradfun, fg!, p0 = optfuns(()->loss(model), ps)
res = Optim.optimize(Optim.only_fg!(fg!), p0, BFGS(), Optim.Options(iterations=1000, store_trace=true))
#data = [(u, train_data)]
#Flux.train!(loss, params(model), data, opt)


# final Loss
println("Final Loss: ", loss(model))
u0 = [0.7; 0.2; 0.3]; u0 = u0./sum(u0)
R_test = zeros(dim_reservoir, length(tt))
# get prediction of the model for test data
ut = W_in*u0
gs = [Spline1D(tt, (W_in*train_data')[i,:], k=2) for i in 1:nv(g_directed)]
p = (gs,σ * w_ij)
prob2 = remake(prob, u0=ut, p=p)
sol2 = solve(prob2, Tsit5(), saveat=tt)
r_test = sol2[:,:]
# plot the r_test
fig = Figure()
ax = GLMakie.Axis(fig[1, 1], xlabel = "Time", ylabel = "u", title = "FitzHugh-Nagumo network")
for i in 1:64
    lines!(ax, sol2.t, r_test[i,:], label="Oscillator $i")
    #text!(ax, t[end], u[i,end]+0.1, text=string("Oscillator ", i), align=(:right, :center))
end
fig
r_state = zeros(128)
X_predicted = zeros(dim_system, length(tt))
X_predicted_rec = zeros(dim_system, length(tt))
res_sol = solve(prob2, Tsit5(), u0=W_in*u0, tspan=(tt[1], tt[2]))
r_state_u = swish.(res_sol[:,end])

for t in 1:length(tt)-1
    X_predict = model(r_state_u) # Wout * rstate
    #println("X_predict: ", X_predict)
    #t == 10 ? break : nothing
    X_predicted_rec[:,t] = X_predict
    res_sol = solve(prob2, Tsit5(), u0=r_state, tspan=(tt[t], tt[t+1]), saveat=[tt[t], tt[t+1]]) # reservoir compute the next state
    println("res_sol: ", size(res_sol.u)) 
    r_state_u = swish.(hcat(res_sol.u...)') # update the reservoir state
    r_state = res_sol[:,1]
    #r_state = model(r_state + W_in*test_data[t,:])
end
# predict the Lorenz system from the reservoir
X_predicted = model(r_test)


#X_predicted = dev_cpu(Lux.apply(model, dev(R_test), ps, st)[1])
#X_predicted = model(R_test)
# X_predicted = X_train'
X_predicted = X_predicted'
X_predicted_rec = X_predicted_rec'
# using the ridge regression to fit output weights
#W_out = (train_data'*R')*inv( (R*R') + beta * I(dim_reservoir) )

#=

r_state = 1.0 ./ (1 .+ exp.(-(A*r_state + W_in*IC_validate)))

for i in 1:length(t2)
    predict = model(r_state)
    X_predicted[i,:] = predict'
    #r_state = 1.0 ./ (1 .+ exp.(-(A*r_state + W_in*X_predicted[i,:])))
    #r_state = 1.0 .+ tanh.(A*r_state + W_in*X_predicted[i,:])
    r_state = swish.(A*r_state + W_in*X_predicted[i,:])
end
=#
# mean squared error
MSE = sum((test_data .- X_predicted).^2)/length(tt)
MSE_rec = sum((test_data[1:50,:] .- X_predicted_rec[1:50,:]).^2)/50
# plot the results
using GLMakie

fig = Figure()
ax1 = Axis(fig[1, 1], ylabel = "x")
ax2 = Axis(fig[2, 1], ylabel = "y")
ax3 = Axis(fig[3, 1], ylabel = "z", xlabel = "Time")
xlims!(ax1, tt[1], tt[end])
xlims!(ax2, tt[1], tt[end])
xlims!(ax3, tt[1], tt[end])
lines!(ax1, tt, test_data[:,1], color = :blue)
lines!(ax1, tt, X_predicted[1:length(tt),1], color = :red, linestyle = :dash)

lines!(ax2, tt, test_data[:,2], color = :green)
lines!(ax2, tt, X_predicted[1:length(tt),2], color = :orange, linestyle = :dash)

lines!(ax3, tt, test_data[:,3], color = :purple)
lines!(ax3, tt, X_predicted[1:length(tt),3], color = :cyan, linestyle = :dash)

fig
GLMakie.save("robertson_FHN_NN.png", fig)

using Optim

# Create a new figure
fig = Figure(resolution = (1600, 1200))
index_15_sec = findfirst(x -> x > 15.0, t2)
range15 = 1:index_15_sec
# 3D plot
ax = Axis3(fig[1, 1], title = "Predicting Lorenz 63", xlabel = "x", ylabel = "y", zlabel = "z")
lines!(ax, test_data[range15, 1], test_data[range15, 2], test_data[range15, 3], color = :blue, label = "True")
lines!(ax, X_predicted[range15, 1], X_predicted[range15, 2],X_predicted[range15, 3], color = :red, label = "Predicted")

# Add grid and legend
axislegend(ax)

# Display the figure
fig
GLMakie.save("lorenz3d_FHN_NN.png", fig)