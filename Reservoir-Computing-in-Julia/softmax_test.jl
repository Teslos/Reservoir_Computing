using NNlib
# testing softmax function
y_true = [1 0 0; 0 0 1; 1 0 0]
y_pred = [0.7 0.2 0.1; 0.1 0.1 0.8; 0.3 0.3 0.4]
println(softmax(y_true))

loss_value = -1/size(y_true, 1) * sum(y_true .* log.(y_pred))

function my_softmax(x)
    return exp.(x) ./ sum(exp.(x), dims=1)
end

my_softmax(y_true)