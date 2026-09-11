


true_lab = [4 3 3 3 3 2 3 1 3 1 1 3 4 2 4 3 1 1 3 2 4 2 4 1 1 1 3 1 2 3 2 4 4 4 4 3 2 1 3 1 1 4 1 3 3 2 2 4 3 3 3 1 3 1 1 1 3 4 1 3 1 2 2 1 2 1 1 1 1 2 3 3 2 4 4 2 3 1 4 1 2 1 3 3 3 3 1 3 2 2 1 1 3 3 3 1 3 3 3 2 3 2 1 1 4 3 2 1 3 1 4 1 1 2 2 4 2 1 4 4 2 1 3 4 1 2 1 1 2 2 1 3 1 3 3 4 4 3 3 3 1 2 1 3 2 3 3 3 2 3 3 2 1 1 3 4 4 1 1 4 3 3 4 3 1 2 2 3 4 4 2 1 3 1 1 4 1 4 4 1 4 4 4 4 3 1 4 1 1 1 4 4 1 3 1 3 4 2 4 3 3 1 1 2 4 1 1 4 1 4 1 2 4 1 1 3 4 2 3 3 4 3 3 2 1 3 1 1 4 3 1 4 3 1 3 4 4 1 2 2 3 3 4 4 4 1 2 2 4 2 1 2 4 2 3 2 1 3 2 1 3 3 3 1 4 4 4 1 3 4 1 4 4 3 4 3 4 2 1 1 1 3 4 1 3 3 3 3 3 1 2 2 3 1 1 4 3 1 3 4 1 1 1 4 3 1 1 1 1 3 1 1 3 4 1 1 3 4 2 2 1 4 2 4 2 3 4 1 3 2 3 3 3 4 3 3 3 4 2 1 3 3 3 4 4 2 2 4 2 3 1 1 1 4 2 1 3 3 3 1 1 2 3 4 1 1 4 3 1 1 2 1 2 2 1 1]
classifier_lab = [4 3 3 3 3 2 3 1 3 1 1 3 4 2 4 3 1 1 3 2 4 2 4 1 1 1 3 1 2 3 2 4 4 4 2 3 2 1 3 1 1 4 1 4 4 3 2 3 3 3 3 1 3 1 2 3 3 4 4 4 1 2 2 1 2 1 1 2 1 2 3 3 2 4 4 1 3 1 3 1 2 1 4 3 3 3 4 3 2 2 1 1 4 3 4 1 3 3 3 2 3 2 1 1 4 3 2 1 3 1 4 1 1 1 2 4 2 1 4 4 2 1 4 4 4 2 1 1 2 1 1 2 1 3 3 3 1 3 3 3 1 2 1 3 2 3 4 3 2 3 3 2 1 1 3 4 4 1 1 4 3 3 4 3 1 2 1 3 3 4 2 1 3 1 1 4 1 4 4 4 4 3 1 1 3 1 3 1 1 1 4 4 1 3 1 3 4 3 4 3 3 1 3 2 4 3 1 4 1 4 4 1 4 1 1 3 4 1 3 3 4 4 4 2 2 3 1 1 4 3 1 4 3 1 3 1 4 1 2 1 3 3 4 3 4 1 2 4 4 2 2 2 4 2 3 4 1 3 2 1 3 3 4 1 4 4 3 1 3 4 1 2 3 3 4 3 4 2 1 1 1 3 3 1 1 3 3 3 4 1 1 2 3 1 1 4 3 1 3 4 1 4 3 4 3 1 1 1 1 3 4 4 3 4 4 1 4 4 1 1 3 3 4 4 1 3 4 1 3 1 3 3 1 4 3 1 3 3 2 1 3 3 3 4 3 2 2 1 2 4 1 1 1 1 1 1 3 3 3 1 1 3 3 3 2 2 4 3 1 1 2 1 2 2 1 1]

# --- Confusion Matrix ---
classes = 1:4
n = length(classes)

conf = zeros(Int, n, n)
for (t, p) in zip(vec(true_lab), vec(classifier_lab))
    conf[t, p] += 1
end

# Pretty print: precision as bottom row, recall as right column
col_w  = 8
sep_w  = col_w * (n + 2)  # +1 for label col, +1 for recall col
println()
print(rpad("T\\P", col_w))
for c in classes; print(lpad(string(c), col_w)); end
println(lpad("Recall", col_w))
println(repeat("-", sep_w))

for i in classes
    tp  = conf[i, i]
    fn  = sum(conf[i, :]) - tp
    rec = (tp + fn) > 0 ? tp / (tp + fn) : 0.0
    print(rpad(string(i), col_w))
    for j in classes; print(lpad(string(conf[i, j]), col_w)); end
    println(lpad(string(round(rec, digits=3)), col_w))
end

println(repeat("-", sep_w))
print(rpad("Prec", col_w))
for j in classes
    tp   = conf[j, j]
    fp   = sum(conf[:, j]) - tp
    prec = (tp + fp) > 0 ? tp / (tp + fp) : 0.0
    print(lpad(string(round(prec, digits=3)), col_w))
end
println()

total       = sum(conf)
overall_acc = sum(conf[i, i] for i in classes) / total
println("\nOverall Accuracy: $(round(overall_acc * 100, digits=2))%")
println()
