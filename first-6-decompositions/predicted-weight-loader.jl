#############################
# Fully-connected weighted graph + spectral clustering + plot (Julia)
#############################

using Graphs
using GraphPlot
using Colors
using Plots
using Compose
using LaTeXStrings
import Cairo, Fontconfig   # For saving the figure

using LinearAlgebra
using Clustering
using Random
using SimpleWeightedGraphs
using CSV
using DataFrames

# Kronecker delta definition
Main.δ(x, y) = ==(x, y)

# ---------------------------
# Node labels
# ---------------------------
nodelabels = [
    "V_1", "V_2", "V_3",           # 1 : 3
    "T_1", "T_2", "T_3",           # 4 : 6
    "x_A1", "x_A2", "x_A3",        # 7 : 9
    "x_B1", "x_B2", "x_B3",        # 10 : 12
    "F_1", "F_2", "F_3",           # 13 : 15
    "F_f1", "F_f2", "F_R",         # 16 : 18
    "Q_1", "Q_2", "Q_3"            # 19 : 21
]

# global all_Q = zeros(Float64, 218)
kk = 58
# for kk = 29

# println(kk)

n = length(nodelabels)

W = fill(0.0, n, n)

for i in 1:n
    W[i, i] = 0.0
end


# Load up edge weights from perfect DMPC 1
# df = CSV.read("C:\\Users\\ellio\\git\\gcn-dmpc\\supervised-edge-prediction\\perfect-dmpc8.csv", DataFrame; header = false)
# vals = Float64.(vec(Matrix(df)))
# load up the edge weights
# df = CSV.read("C:\\Users\\ellio\\git\\gcn-dmpc\\supervised-edge-prediction\\toy-architecture-int.csv", DataFrame; header=false)
# vals = Float64.(vec(Matrix(df)))


# Load up edge weights from the machine learning file
# global f = CSV.read("C:\\Users\\escooper\\git\\gcn-dmpc\\supervised-edge-prediction\\ML-preds.csv", DataFrame).instance[6]
global df = CSV.read("C:\\Users\\ellio\\git\\gcn-dmpc\\final-analysis\\first-6-decompositions\\MLE-classifier-on-1-6\\ML-preds-GCN-floats.csv", DataFrame)[:, 2:end]  # skip the first column (instance)
# vals = Float64.(Vector(df[kk, :]))   # length = 210

# println("Problem instance: $(CSV.read("C:\\Users\\ellio\\git\\gcn-dmpc\\supervised-edge-prediction\\ML-preds.csv", DataFrame).instance[kk])")

global idx = 1
for i in 1:(n-1)
    for j in (i+1):n
        W[i, j] = vals[idx]
        global idx += 1
    end
end

# enforce symmetry (reflect across diagonal)
W = max.(W, W')

# ---------------------------
# Spectral clustering on weighted graph
# ---------------------------

function spectral_communities(A::AbstractMatrix{<:Real}; k::Int=3, seed::Int=1)
    n = size(A, 1)
    @assert size(A, 2) == n "A must be square"

    Wloc = Array{Float64}(A)

    # Remove self-loops for clustering
    @inbounds for i in 1:n
        Wloc[i, i] = 0.0
    end

    # weighted degree
    d = vec(sum(Wloc, dims=2))

    invsqrt_d = similar(d)
    @inbounds for i in 1:n
        invsqrt_d[i] = d[i] > 0 ? 1 / sqrt(d[i]) : 0.0
    end

    # normalized Laplacian L = I - D^{-1/2} W D^{-1/2}
    S = Wloc .* (invsqrt_d * invsqrt_d')
    L = I - S

    F = eigen(Symmetric(Matrix(L)))   # ascending eigenvalues
    U = F.vectors[:, 1:k]

    # row-normalize embeddings
    @inbounds for i in 1:n
        normi = norm(U[i, :])
        if normi > 0
            U[i, :] ./= normi
        end
    end

    Random.seed!(seed)
    km = kmeans(permutedims(U), k; maxiter=300)  # observations as columns
    return vec(km.assignments)
end

c = spectral_communities(W; k=3, seed=1)
println("c_community_detection = ", c)

# ---------------------------
# Weighted modularity (using W as adjacency)
# ---------------------------
n = size(W, 1)

# weighted degrees
d = vec(sum(W, dims=2))

# total edge weight
m = sum(d) / 2

Q = (1 / (2m)) * sum(
    (W[i, j] - (d[i] * d[j]) / (2m)) * δ(c[i], c[j])
    for i in 1:n, j in 1:n
)

k = maximum(c)
a = zeros(Float64, k)
for s in 1:k
    a[s] = sum(d[c.==s]) / (2m)   # community "volume" fraction
end

# “How close am I to the best imaginable outcome for this k-way split?”
Qmax_partition = 1.0 - sum(a .^ 2)  # upper bound IF no between-community edges
Qnorm = Q / Qmax_partition

println("Q = ", Q)
println("Qmax_partition = ", Qmax_partition)
println("Qnorm = ", clamp(Qnorm, -10, 1))

# all_Q[kk] = clamp(Qnorm, -10, 1)

# all_c_values[jj] = clamp(Qnorm / 0.6666, -10, 1)
# ---------------------------
# Colors for nodes by community
# ---------------------------
color_map = Dict(
    1 => "#90EE90",  # light green
    2 => "#00FFFF",  # cyan
    3 => "#FF7F50",  # coral
    4 => "#DAA520",  # goldenrod
    5 => "#6A5ACD",  # slate blue / purple
    6 => "#1F3A5F",   # deep steel blue
    7 => "#C0392B"  # deep crimson / muted red
)

nodefills = [get(color_map, c[i], "#FFFFFF") for i in 1:n]

# ---------------------------
# Build a fully-connected graph object for plotting
# (GraphPlot draws from the graph structure, not from W)
# ---------------------------
wg_full = SimpleWeightedGraph(W)  # undirected weighted graph
g_full = Graph(wg_full)          # unweighted view for GraphPlot

# ---------------------------
# Edge widths as a function of edge strength
# ---------------------------
elist = collect(edges(g_full))
eweights = [W[src(e), dst(e)] for e in elist]

# Map weights -> line widths (tune these)
wmin, wmax = minimum(eweights), maximum(eweights)
lw_min, lw_max = 0.15, 2.5

# Avoid divide-by-zero if all weights are equal
den = (wmax - wmin)
p = 2.0  # >1 emphasizes strong edges, <1 emphasizes weak edges
edgewidths = den > 0 ?
             (lw_min .+ (lw_max - lw_min) .* (((eweights .- wmin) ./ den) .^ p)) :
             fill((lw_min + lw_max) / 2, length(eweights))

gp = gplot(
    g_full,
    nodelabel=nodelabels,
    layout=shell_layout,
    NODELABELSIZE=2,
    nodesize=2,
    nodefillc=nodefills,
    EDGELINEWIDTH=edgewidths
)

# draw(PNG("C:\\Users\\escooper\\git\\gcn-dmpc\\supervised-edge-prediction\\instance-1404.png", 800px, 800px), gp)
# end
