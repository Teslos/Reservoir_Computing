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

# generate the training data
function lorenz!(dx, x, p, t)
    sigma, rho, beta = p
    dx[1] = sigma*(x[2] - x[1])
    dx[2] = x[1]*(rho - x[3]) - x[2]
    dx[3] = x[1]*x[2] - beta*x[3]
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



# generate the training data for lorenz system
u0 = [10;10;10]
tspan = (0.0, 300.0)
dt = 0.01
p = (10.0, 28.0, 8/3) # sigma, rho, beta values
prob = ODEProblem(lorenz!, u0, tspan, p)
sol  = solve(prob, Tsit5(), saveat = dt, progress = true)
train_data = hcat(sol.u...)'
tlorenz = sol.t

# generate the test data
IC_validate = [10.1; 10.0; 10.0]
tspan2 = (0.0, 50.0)
prob2 = ODEProblem(lorenz!, IC_validate, tspan2, p)
sol2 = solve(prob2, Tsit5(), saveat = dt, progress = true)
test_data = hcat(sol2.u...)'
t2 = sol2.t

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
const σ = 0.006 # coupling strength
const R0 = 0.5
# different weights for edges, because the resitivity of the edges are always positive
w_ij = [pdf(Normal(), x) for x in range(-1, 1, length=ne(g_directed))]
gs = [Spline1D(tlorenz, (W_in*train_data')[i,:], k=2) for i in 1:nv(g_directed)]
# Tuple of parameters for nodes and edges
p = (gs,σ * w_ij)
#Initial conditions
x0 = W_in*u0

# Solving the ODE
using OrdinaryDiffEq

tspan = (0.0, 300.0)
datasize = length(tlorenz)
tsteps = range(tspan[1], tspan[2], length=datasize)
prob = ODEProblem(fhn_network!, x0, tspan, p)
sol = solve(prob, Tsit5(), saveat=tsteps)

using GLMakie
fig = Figure()
ax = GLMakie.Axis(fig[1, 1], xlabel = "Time", ylabel = "u", title = "FitzHugh-Nagumo network")
t= sol.t
u = sol[1:2:128,:]
# use only the u values
diff_data = u
for i in 1:64
    lines!(ax, t, u[i,:], label="Oscillator $i")
    #GLMakie.heatmap!(ax, t, i*ones(length(t)), sol[i,:], colormap = :viridis)
    #text!(ax, t[end], u[i,end]+0.1, text=string("Oscillator ", i), align=(:right, :center))
end
#axislegend(ax, position = :rt)
fig
GLMakie.save("RC_FHN_NN.png", fig)

# instead of ridge regression, we can use neural network to fit the output weights
using Flux
using Lux
using LuxCUDA
using Flux: crossentropy, onecold, onehotbatch, params, mse
using Flux.Data: DataLoader
using Statistics: mean
using Random
using Optimisers
using MLDataUtils
using Zygote

#=
# using the neural network to fit output weights
const dev = gpu_device()
const dev_cpu = cpu_device()
model = Lux.Chain(Lux.Dense(64, 100, swish), Lux.Dense(100, dim_system))
rng = Random.default_rng()
nn_rc, st_rc = Lux.setup(rng, model) |> dev

function loss(x, y, model, ps, st)
    pred, st = model(x, ps, st)
    loss = sum((pred .- y).^2)
    return loss, st
end
loss_function(ps, st, x, y) = loss(x, y, model, ps, st)

function train_model(model, ps, st, train_data, epochs=10000, batch_size=1024, learning_rate=0.001)
    loss_history = []
    loss_value = 0.0
    opt = ADAM(learning_rate)
    st_opt = Optimisers.setup(opt, st)
    train_dataloader = DataLoader(train_data, batchsize=batch_size, shuffle=true) |> dev
    for epoch in 1:epochs
        for (x,y) in train_dataloader
            (loss_value, st), back = Zygote.pullback(loss, x, y, model, ps, st)
            grads = back((one(loss_value),nothing))[4]
            st_opt, ps = Optimisers.update(st_opt, ps, grads)

            push!(loss_history, loss_value)
        end
        if epoch % 100 == 0
            println("Epoch $epoch, Loss: $loss_value")
        end
    end
    return ps, st
end

ps, st = Lux.setup(rng, model) |> dev
data = (u, train_data')
ps, st = train_model(model, ps, st, data)


println("Final Loss: ", loss(dev(u), dev(train_data'), model, ps, st)[1])
=# 

using FluxOptTools
using Optim 
# using the neural network to fit output weights
model = Flux.Chain(Flux.Dense(64, 256, swish), Flux.Dense(256, dim_system))
loss(model) = mean(abs2, model(u) .- train_data')
opt = Flux.ADAM(0.01)
Zygote.refresh()
ps = Flux.params(model)
lossfun, gradfun, fg!, p0 = optfuns(()->loss(model), ps)
res = Optim.optimize(Optim.only_fg!(fg!), p0, BFGS(), Optim.Options(iterations=1000, store_trace=true))
#data = [(u, train_data)]
#Flux.train!(loss, params(model), data, opt)


# final Loss
println("Final Loss: ", loss(model))
R_test = zeros(dim_reservoir, length(t2))
# get prediction of the model for test data
#r_state = 1.0 ./ (1 .+ exp.(-(A*r_state + W_in*IC_validate)))
ut = W_in*IC_validate
gs = [Spline1D(t2, (W_in*test_data')[i,:], k=2) for i in 1:nv(g_directed)]
p = (gs,σ * w_ij)
prob2 = remake(prob, u0=ut, tspan=tspan2, p=p)
sol2 = solve(prob2, Tsit5(), saveat=t2)
r_test = sol2[1:2:128,:]
# plot the r_test
fig = Figure()
ax = GLMakie.Axis(fig[1, 1], xlabel = "Time", ylabel = "u", title = "FitzHugh-Nagumo network")
for i in 1:64
    lines!(ax, t2, r_test[i,:], label="Oscillator $i")
    #text!(ax, t[end], u[i,end]+0.1, text=string("Oscillator ", i), align=(:right, :center))
end
fig
r_state = zeros(128)
X_predicted = zeros(dim_system, length(t2))
X_predicted_rec = zeros(dim_system, length(t2))
res_sol = solve(prob2, Tsit5(), u0=W_in*IC_validate, tspan=(t2[1], t2[2]))
r_state_u = swish.(res_sol[1:2:128,end])

for t in 1:length(t2)-1
    X_predict = model(r_state_u) # Wout * rstate
    #println("X_predict: ", X_predict)
    #t == 10 ? break : nothing
    X_predicted_rec[:,t] = X_predict
    res_sol = solve(prob2, Tsit5(), u0=r_state, tspan=(t2[t], t2[t+1]), saveat=[t2[t], t2[t+1]]) # reservoir compute the next state
    
    r_state_u = swish.(res_sol[1:2:128,2])
    r_state = res_sol[:,2]
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
MSE = sum((test_data .- X_predicted).^2)/length(t2)
MSE_rec = sum((test_data[1:50,:] .- X_predicted_rec[1:50,:]).^2)/50
# plot the results
using GLMakie

fig = Figure()
ax1 = Axis(fig[1, 1], ylabel = "x")
ax2 = Axis(fig[2, 1], ylabel = "y")
ax3 = Axis(fig[3, 1], ylabel = "z", xlabel = "Time")
xlims!(ax1, t2[1], 15.0)
xlims!(ax2, t2[1], 15.0)
xlims!(ax3, t2[1], 15.0)
lines!(ax1, t2, test_data[:,1], color = :blue)
lines!(ax1, t2, X_predicted[1:length(t2),1], color = :red)

lines!(ax2, t2, test_data[:,2], color = :green)
lines!(ax2, t2, X_predicted[1:length(t2),2], color = :orange)

lines!(ax3, t2, test_data[:,3], color = :purple)
lines!(ax3, t2, X_predicted[1:length(t2),3], color = :cyan)

fig
GLMakie.save("lorenz_FHN_NN.png", fig)

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