# dry beans dataset
module drybean
export read_drybean
# https://archive.ics.uci.edu/ml/datasets/Dry+Bean+Dataset

using CSV
using DataFrames

# Read the dry bean dataset
function read_drybean(file="./data/DryBeanDataset.csv")
    df = CSV.read(file, DataFrame)
    return df
end

end

