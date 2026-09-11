
using CSV, DataFrames, LinearAlgebra, Statistics

ml_path = raw"C:\Users\ellio\git\gcn-dmpc\final-analysis\first-6-decompositions\MLE-classifier-on-1-6\ML-preds-GCN-floats.csv"

d1_path = raw"C:\Users\ellio\git\gcn-dmpc\supervised-edge-prediction\perfect-dmpc1.csv"
d2_path = raw"C:\Users\ellio\git\gcn-dmpc\supervised-edge-prediction\perfect-dmpc2.csv"
d3_path = raw"C:\Users\ellio\git\gcn-dmpc\supervised-edge-prediction\perfect-dmpc3.csv"
d4_path = raw"C:\Users\ellio\git\gcn-dmpc\supervised-edge-prediction\perfect-dmpc4.csv"
d5_path = raw"C:\Users\ellio\git\gcn-dmpc\supervised-edge-prediction\perfect-dmpc5.csv"
d6_path = raw"C:\Users\ellio\git\gcn-dmpc\supervised-edge-prediction\perfect-dmpc6.csv"

weights_path = raw"C:\Users\ellio\git\gcn-dmpc\final-analysis\first-6-decompositions\MLE-classifier-on-1-6\edge-weights-210.csv"

out_scores_path = raw"C:\Users\ellio\git\gcn-dmpc\final-analysis\first-6-decompositions\MLE-classifier-on-1-6\ML-weighted-bernoulli-loglik-scores-GCN-floats.csv"
out_pred_path   = raw"C:\Users\ellio\git\gcn-dmpc\final-analysis\first-6-decompositions\MLE-classifier-on-1-6\ML-weighted-bernoulli-predicted-arch-GCN-floats.csv"

edge_col_start = 22

# Numerical stability for logs
eps = 1e-12

# Parse a CSV cell to Float64 when possible; return `nothing` for non-numeric values.
function to_float_or_nothing(x)
    if x === missing
        return nothing
    elseif x isa Number
        return Float64(x)
    elseif x isa AbstractString
        return tryparse(Float64, strip(x))
    else
        return tryparse(Float64, string(x))
    end
end

# Load a vector from CSV that might be 1xM or Mx1 (header optional).
# Non-numeric cells (e.g., x1,x2,... header text) are ignored.
function load_vec(path::AbstractString)
    df = CSV.read(path, DataFrame; header=false)
    X = Matrix(df)

    vals = Float64[]
    @inbounds for i in axes(X, 1), j in axes(X, 2)
        v = to_float_or_nothing(X[i, j])
        if v !== nothing
            push!(vals, v)
        end
    end

    if isempty(vals)
        error("No numeric values found in: $path")
    end
    return vals
end

"""
    weighted_bernoulli_loglik(p, a, w; eps=1e-12)

Compute sum_e w_e [ a_e log(p_e) + (1-a_e) log(1-p_e) ].
Assumes a is 0/1 (or Bool). Clamps p to [eps, 1-eps].
"""
function weighted_bernoulli_loglik(p::AbstractVector, a::AbstractVector, w::AbstractVector;
                                  eps::Float64=1e-12)
    @assert length(p) == length(a) == length(w)
    s = 0.0
    @inbounds for i in eachindex(p)
        wi = w[i]
        if wi == 0.0
            continue
        end
        pi = p[i]
        # clamp
        if pi < eps
            pi = eps
        elseif pi > 1 - eps
            pi = 1 - eps
        end
        ai = a[i]
        s += wi * (ai * log(pi) + (1 - ai) * log(1 - pi))
    end
    return s
end

# -------------------------
# Load data
# -------------------------
ml_df = CSV.read(ml_path, DataFrame)

# ML predictions matrix (N x 210)
F = Matrix{Float64}(ml_df[:, edge_col_start:end])
N, M = size(F)

# Load templates (1 x 210 each)
d1 = vec(Matrix(CSV.read(d1_path, DataFrame; header=false)))
d2 = vec(Matrix(CSV.read(d2_path, DataFrame; header=false)))
d3 = vec(Matrix(CSV.read(d3_path, DataFrame; header=false)))
d4 = vec(Matrix(CSV.read(d4_path, DataFrame; header=false)))
d5 = vec(Matrix(CSV.read(d5_path, DataFrame; header=false)))
d6 = vec(Matrix(CSV.read(d6_path, DataFrame; header=false)))

templates = [Float64.(d1), Float64.(d2), Float64.(d3), Float64.(d4), Float64.(d5), Float64.(d6)]

# Basic checks
@assert M == length(templates[1]) "ML output has $M edges, but templates have $(length(templates[1]))"
for k in 1:6
    @assert length(templates[k]) == M
end

# Template edge counts for optional sparsity penalty
template_edge_counts = [sum(templates[k]) for k in 1:6]

# Load weights
w = load_vec(weights_path)
@assert length(w) == M "Weight vector length $(length(w)) does not match ML output length $M"

scores = zeros(Float64, N, 6)

@inbounds for i in 1:N
    p = view(F, i, :)
    for k in 1:6
        ll = weighted_bernoulli_loglik(p, templates[k], w; eps=eps)

        scores[i, k] = ll
    end
end

# Predicted architecture = argmax score (1..6)
pred_arch = [argmax(view(scores, i, :)) for i in 1:N]

CSV.write(out_scores_path, DataFrame(scores, :auto))
CSV.write(out_pred_path, DataFrame(pred_arch = pred_arch))

println("Done.")
println("Wrote scores to: ", out_scores_path)
println("Wrote predictions to: ", out_pred_path)
