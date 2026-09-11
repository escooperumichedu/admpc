using CSV, DataFrames, Statistics, Printf

data_path = raw"C:\Users\ellio\git\gcn-dmpc\final-analysis\first-6-decompositions\MLE-classifier-on-1-6\ML-preds-GCN-train-floats-and-labels.csv"

label_col = :label        
edge_start_col = 2  
num_arch = 6              
eps = 1e-12 # Numerical stability for variance denominator            

df = CSV.read(data_path, DataFrame)

# Filter by architecture
df1 = filter(row -> row[label_col] == 1, df)
df2 = filter(row -> row[label_col] == 2, df)
df3 = filter(row -> row[label_col] == 3, df)
df4 = filter(row -> row[label_col] == 4, df)
df5 = filter(row -> row[label_col] == 5, df)
df6 = filter(row -> row[label_col] == 6, df)

dfs = [df1, df2, df3, df4, df5, df6]

edge_cols = names(df)[edge_start_col:end]
E = length(edge_cols) # number of edges (210)

between_var = zeros(Float64, E)
within_var = zeros(Float64, E)
weights = zeros(Float64, E)

for (e_idx, colname) in enumerate(edge_cols)

    # For this index, compute stats per architecture
    mu = zeros(Float64, num_arch)   # mean per architecture
    sigma2 = zeros(Float64, num_arch)   # variance per architecture

    for k in 1:num_arch
        pk = dfs[k][!, colname]
        mu[k]  = mean(pk) # Mean prediction for this edge in this architecture k
        sigma2[k] = var(pk; corrected=false) # Population variance for this edge in this architecture k
    end                                                       

    # BETWEEN pop variance: how different the architecture means are
    between_var[e_idx] = var(mu; corrected=false)

    # WITHIN variance: average variance inside architectures
    within_var[e_idx] = mean(sigma2)

    # Raw SNR weight
    weights[e_idx] = between_var[e_idx] / (within_var[e_idx] + eps)
end

println("weight vector:")
println(weights)

CSV.write(raw"C:\Users\ellio\git\gcn-dmpc\final-analysis\first-6-decompositions\MLE-classifier-on-1-6\edge-weights-210.csv", DataFrame(weights=weights))