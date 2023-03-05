using Flux
using Flux.Losses
using Flux: onehotbatch, onehot
using Statistics: mean

export run_on_conll, run_on_ontonotes, onehot_y, train, accuracy_onehot, evaluate


const BATCH_SIZE = 32
EPOCHS = 1 #for demonstrating purposes
const LSTM_OUTPUT_DIM = 32
ASSETS_PATHS = chop(dirname(@__FILE__), tail=3) 

"""
    onehot_y(y_data::Vector{Int32})

Function to turn y labels vector to one hot Flux representation.
    
# Arguments
- `y_data`: Vector of labels   
"""
function onehot_y(y_data::Vector{Int32})
    return onehotbatch(y_data, 1:3)
end

"""
    train(x_data::Array{Float32}, y_data::Flux.OneHotMatrix, valid_x::Array{Float32}, valid_y::Flux.OneHotMatrix, graph_file_path::String)

Function to train my RNN and output classification model.
    
# Arguments
- `x_data`: Array of x traning features
- `y_data`: OneHotMatrix of training labels
- `valid_x`: Array of x features from validation set
- `valid_y`: OneHotMatrix of validation labels
- `graph_file_path`: Path to file which stores training statistics for future evaluation in notebook     
"""
function train(x_data::Array{Float32}, y_data::Flux.OneHotMatrix, valid_x::Array{Float32}, valid_y::Flux.OneHotMatrix, graph_file_path::String)
    cur_epoch = 0
    dataset = Flux.Data.DataLoader((x_data, y_data), batchsize=BATCH_SIZE, shuffle=true)
    graph_file = open(graph_file_path, "w")

    model = Chain(
        LSTM(EMBEDDING_DIM => LSTM_OUTPUT_DIM),
        Dense(LSTM_OUTPUT_DIM => IOB_DIM),
        softmax
    )

    function loss(x, y)
        b = size(x,2)
        predictions = [model(x[:,i]) for i=1:b]
        value = crossentropy(hcat(predictions...), y)
        Flux.reset!(model)
        
        return value
    end

    evalcb = function()
        b = size(valid_x,2)
        new_predictions_tmp = [model(valid_x[:,i]) for i=1:b]
        new_predictions = hcat(new_predictions_tmp...)
        loss_val = crossentropy(new_predictions, valid_y)
        accuracy = accuracy_onehot(new_predictions, valid_y)

        @info "Accuracy on validation set: $accuracy"
        @info "Loss on validation set: $loss_val"
        println(graph_file, "$cur_epoch $loss_val $accuracy")
    end    

    optimizer = ADAM()

    for epoch in 1:EPOCHS
        cur_epoch = epoch
        @info "Epoch #$epoch"
        Flux.train!(loss, Flux.params(model), dataset, optimizer, cb=Flux.throttle(evalcb, 20))
        println("\n")
    end    

    close(graph_file)
    
    return model
end    

"""
    run_on_conll()

Function to run training on conll2003 dataset.     
"""
function run_on_conll()
    train_x, train_y, test_x, test_y, valid_x, valid_y = prepare_conll_dataset()
    graph_file_path = joinpath(ASSETS_PATHS, "data", "loss_conll.txt")   #"../data/loss_conll.txt"
    model = train(train_x, onehot_y(train_y), valid_x, onehot_y(valid_y), graph_file_path)
    
    evaluate(model, test_x, onehot_y(test_y), "conll2003")
end

"""
    run_on_ontonotes()

Function to run training on OntoNotes5.0 dataset.     
"""
function run_on_ontonotes()
    train_x, train_y, test_x, test_y, valid_x, valid_y = prepare_ontonotes_dataset()
    graph_file_path = joinpath(ASSETS_PATHS, "data", "loss_ontonotes.txt")    #"../data/loss_ontonotes.txt"
    model = train(train_x, onehot_y(train_y), valid_x, onehot_y(valid_y), graph_file_path)

    evaluate(model, test_x, onehot_y(test_y), "OntoNotes5.0")
end

"""
    accuracy_onehot(y_pred::Matrix{Float32}, y::Flux.OneHotArrays.OneHotMatrix{UInt32, Vector{UInt32}})

Function to compute accuracy of y predictions against groundtruth labels.
    
# Arguments
- `y_pred`: Matrix of prediction labels
- `y`: OneHot of ground truth labels   
"""
function accuracy_onehot(y_pred::Matrix{Float32}, y::Flux.OneHotArrays.OneHotMatrix{UInt32, Vector{UInt32}})
    return mean(Flux.onecold(y_pred) .== Flux.onecold(y))
end

"""
    evaluate(model, test_x::Array{Float32}, test_y::Flux.OneHotMatrix, dataset_name::String)
    
Function to run and evaluate trained model on testing set.
    
# Arguments
- `model`: Trained RNN model
- `test_x`: Array of testing x features
- `test_y`: OneHot of testing y labels
- `dataset_name`: Name of evaluating dataset   
"""
function evaluate(model, test_x::Array{Float32}, test_y::Flux.OneHotMatrix, dataset_name::String)
    println("\n")
    println("NER classification report on $dataset_name")
    
    _, cols = size(test_x)
    println("Number of words: $cols")
    println("Number of labels: 3")
    println("Labels: I, O, B")

    b = size(test_x, 2)
    new_predictions_tmp = [model(test_x[:, i]) for i=1:b]
    new_predictions = hcat(new_predictions_tmp...)
    loss_value = crossentropy(new_predictions, test_y)
    accuracy = accuracy_onehot(new_predictions, test_y)

    println("Loss on testing set: $loss_value")
    println("Accuracy on testing set: $accuracy")
end
