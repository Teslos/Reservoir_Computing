using OrdinaryDiffEq
using LinearAlgebra
using NNlib: swish
using Flux: onehotbatch, shuffle, DataLoader 
using Random
using MLUtils: splitobs
using Lux 
using Flux: pullback, onecold, onehotbatch, softmax
using Optimisers
using Distributions
using Plots

include("drybean.jl")
include("spikerate.jl")

function load_data(x, y, train_ratio = 0.8, batchsize=256)
    rng = MersenneTwister(1234)
    num_samples = size(x, 2)
    classes = unique(y)

    # split the data
    y = onehotbatch(y, classes)
    (train_x, train_y), (test_x, test_y) = splitobs((x,y); at=train_ratio)
    return (
        # Use dataloader to automatically shuffle and batch the data
        DataLoader(collect.((train_x, train_y)); batchsize, shuffle = true),
        # dont shuffle the test data
        DataLoader(collect.((train_x, train_y)); batchsize, shuffle = false)
    )
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

# read the dry bean dataset
current_path = pwd()
db = drybean.read_drybean(current_path * "\\Reservoir-Computing-in-Julia\\data\\DryBeanDataset.csv")
x = Matrix(permutedims(db))
function normalize_rows(x::AbstractMatrix)
    x = x ./ maximum(x, dims=2)
    return x
end
# target vector
y = x[17,:]
x = normalize_rows(x[1:16,:])
classes = unique(y)
onehot_y = onehotbatch(y, classes)


train_dataloader, test_dataloader = load_data(x, y, 0.8)


function reservoir_init()
    dim_system = 16
    dim_reservoir = 500

    sigma = 0.1
    density = 0.05 # density of the reservoir
    beta = 0.01 # regularization parameter
    α = 1.0
    r_state = zeros(dim_reservoir)
    A = generate_reservoir(dim_reservoir, density)
    W_in = 2*sigma*(rand(dim_reservoir, dim_system) .- 0.5)

    W_out = zeros(dim_system, dim_reservoir)
    
    return A, W_in, W_out, r_state
end

function reservoir_compute(train_data, time_length, W_in, A, r_state)
    dim_reservoir = size(A, 1)
    R = zeros(dim_reservoir, time_length)
    α = 0.9
    for i in 1:time_length
        R[:,i] = r_state
        #r_state = 1.0 ./ (1 .+ exp.(-(A*r_state + W_in*train_data[i,:])))
        #r_state = 1.0 .+ tanh.(A*r_state + W_in*train_data[i,:])
        r_state = (1-α)*r_state + α * swish.(A*r_state + W_in*train_data[i,:]) # new state r_state(t+1)
    end
    return R # return the states of the reservoir
end

function create_model()
    return Lux.Chain(
        Lux.Dense(16, 16, sigmoid),
        Lux.Dense(16, 7), softmax
    )
end

# loss function for the model is crossentropy
function loss(x, y, model, ps, st)
    y_pred, st = model(x, ps, st)
   
    lossv = -mean(sum(y .* log.(softmax(y_pred)), dims=1))
    #println("lossv:",lossv)
    return lossv, st
end

function partition(data, batch_size)
    x, y = data
    return ((x[i:min(i+batch_size-1, end),:], y[:,i:min(i+batch_size-1, end)]) for i in 1:batch_size:size(x, 1))
end

function accuracy(model, ps, st, dataloader, A, W_in, W_out, r_state)
    beta = 0.01
    dim_reservoir = size(A, 1)
    total_correct, total = 0, 0
    st = Lux.testmode(st)
    for (x,y) in dataloader
        target_class = onecold(y)
        
        x = x'
        R = reservoir_compute(x, size(x,1), W_in, A, r_state)
        W_out = (x'*R')*inv( (R*R') + beta * I(dim_reservoir) )
        y_out = W_out*R[:,1:end] # output of the reservoir
        predict_class = model(y_out, ps, st)
        
        predicted_class = onecold(Array(first(predict_class)))
        #println("predicted_class:",predicted_class)
        #exit()
        total_correct += sum(target_class .== predicted_class)
        total += length(y)
    end
    return total_correct / total
end

loss_function(x, y, model, ps, states) = loss(x, y, model, ps, states)

function train_model(model, train_dataloader, test_dataloader; epochs=10, batch_size=256, learning_rate=0.001)
    ps, st = Lux.setup(rng, model)
    dim_reservoir = 500
    beta = 0.01 # regularization parameter
    opt = Optimisers.Adam(learning_rate)
    st_opt = Optimisers.setup(opt, ps)
    
    A, W_in, W_out, r_state  = reservoir_init()

    # train loop
    for epoch in 1:epochs
        for (x, y) in train_dataloader
            #println("x:",size(x),"y:",size(y))
            # do reservoir computation
            x = x'
            R = reservoir_compute(x, size(x,1), W_in, A, r_state)
            # using the ridge regression to fit output weights
            W_out = (x'*R')*inv( (R*R') + beta * I(dim_reservoir) )
            y_out = W_out*R[:,1:end]
            
            (loss_value, st), back = pullback(loss_function, y_out, y, model, ps, st)
        
            grads = back((one(loss_value),nothing))[4]
            st_opt, ps = Optimisers.update(st_opt, ps, grads)
        end
        #println("train_labels:", size(train_data[2]))

        train_acc = accuracy(model, ps, st, train_dataloader,A, W_in, W_out, r_state)

        test_acc = accuracy(model, ps, st, test_dataloader, A, W_in, W_out, r_state)
        println("Epoch $epoch, Train Accuracy: $train_acc, Test Accuracy: $test_acc")
    end
end


# Create model 
model = create_model()
rng = Random.seed!(1234)
nn_rc, st_rc = Lux.setup(rng, model)

# Train the model
train_model(model, train_dataloader, test_dataloader, epochs=100, batch_size=256, learning_rate=0.001)



# using the ridge regression to fit output weights
W_out = (train_data'*R')*inv( (R*R') + beta * I(dim_reservoir) )

X_predicted = zeros(length(t2), dim_system)
r_state = 1.0 ./ (1 .+ exp.(-(A*r_state + W_in*test_data[end,:]))) # initial state
for i in 1:length(t2)
    X_predicted[i,:] = W_out*r_state
    r_state = 1.0 ./ (1 .+ exp.(-(A*r_state + W_in*X_predicted[i,:])))
    #r_state = 1.0 .+ tanh.(A*r_state + W_in*X_predicted[i,:])
    #r_state = (1-α)*r_state + α * swish.(A*r_state + W_in*X_predicted[i,:]) # new state r_state(t+1)
end

# plot the results
using GLMakie

fig = Figure()
for i in 1:4
    ax = Axis(fig[i, 1])
    lines!(ax, t2, test_data[:,i], color = :blue)
    lines!(ax, t2, X_predicted[:,i], color = :red)
end


fig
GLMakie.save("dry_bean_BOMBAY.png", fig)

# Create a new figure
fig = Figure(resolution = (800, 600))

# 3D plot
ax = Axis3(fig[1, 1], title = "Predicting dry bean", xlabel = "x", ylabel = "y", zlabel = "z")
lines!(ax, test_data[:, 1], test_data[:, 2], test_data[:, 3], color = :blue, label = "True")
lines!(ax, X_predicted[:, 1], X_predicted[:, 2], X_predicted[:, 3], color = :red, label = "Predicted")

# Add grid and legend
axislegend(ax)

# Display the figure
fig
GLMakie.save("drybean3d.png", fig)