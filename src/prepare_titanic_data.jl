using DataFrames
using CSV
using Statistics: mean
using StatsBase: countmap

export get_train_features, get_test_features, notdefined_replacement


ASSETS_PATH = chop(dirname(@__FILE__), tail=3) 

"""
    notdefined_replacement(col_val::Any, replace_val::Any)

Function to replace missing or NaN value from DataFrame.
    
# Arguments
- `col_val`: Value from DataFrame to check
- `replace_val`: Replacement value in case of NaN or missing    
"""
notdefined_replacement(col_val::Any, replace_val::Any) = ismissing(col_val) || (col_val isa Number && isnan(col_val)) ? replace_val : col_val

"""
    get_train_features()

Function to get training features from Titanic dataset
"""
function get_train_features()
    train_data = CSV.read(joinpath(ASSETS_PATH, "data", "titanic", "train.csv"), DataFrame) #"../data/titanic/train.csv"

    select!(train_data, Not([:Name, :Ticket, :Cabin, :PassengerId, :Survived]))
    train_data[!, :Family_size] = train_data[!, :SibSp] .+ train_data[!, :Parch] .+ 1
    select!(train_data, Not([:SibSp, :Parch]))
    train_data[!, :Age] = notdefined_replacement.(train_data[!, :Age], round(mean(skipmissing(train_data[!, :Age])), digits=1))
    train_data[!, :Fare] = notdefined_replacement.(train_data[!, :Fare], round(mean(skipmissing(train_data[!, :Fare])), digits=1))    
    train_data[!, :Embarked] = notdefined_replacement.(train_data[!, :Embarked], findmax(countmap(train_data.Embarked))[2])

    sex_value_map = Dict("male" => 1, "female" => 2)
    train_data[!, :Sex] = [sex_value_map[item] for item in train_data[!, :Sex]]

    embarked_value_map = Dict("Q" => 1, "S" => 2, "C" => 3)
    train_data[!, :Embarked] = [embarked_value_map[item] for item in train_data[!, :Embarked]]

    train_x = Matrix{Float64}(train_data)
    train_y = CSV.File(joinpath(ASSETS_PATH, "data", "titanic", "train.csv"); select=[2]).Survived      #"../data/titanic/train.csv"

    return train_x, train_y
end

"""
    get_test_features()

Function to get testing features from Titanic dataset
"""
function get_test_features()
    test_data = CSV.read(joinpath(ASSETS_PATH, "data", "titanic", "test.csv"), DataFrame)  #"../data/titanic/test.csv"

    select!(test_data, Not([:Name, :Ticket, :Cabin, :PassengerId]))
    test_data[!, :Family_size] = test_data[!, :SibSp] .+ test_data[!, :Parch] .+ 1
    select!(test_data, Not([:SibSp, :Parch]))
    test_data[!, :Age] = notdefined_replacement.(test_data[!, :Age], round(mean(skipmissing(test_data[!, :Age])), digits=1))
    test_data[!, :Fare] = notdefined_replacement.(test_data[!, :Fare], round(mean(skipmissing(test_data[!, :Fare])), digits=1))    
    test_data[!, :Embarked] = notdefined_replacement.(test_data[!, :Embarked], findmax(countmap(test_data.Embarked))[2])

    sex_value_map = Dict("male" => 1, "female" => 2)
    test_data[!, :Sex] = [sex_value_map[item] for item in test_data[!, :Sex]]

    embarked_value_map = Dict("Q" => 1, "S" => 2, "C" => 3)
    test_data[!, :Embarked] = [embarked_value_map[item] for item in test_data[!, :Embarked]]

    test_x = Matrix{Float64}(test_data)
    test_y = CSV.File(joinpath(ASSETS_PATH, "data", "titanic", "test_survived.csv"); select=[2]).Survived  #"../data/titanic/test_survived.csv"
     
    return test_x, test_y
end
