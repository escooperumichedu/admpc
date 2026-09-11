using CSV, DataFrames, LaTeXStrings, Plots
import Plots: mm

default(
    framestyle              = :box,
    background_color        = :white,
    background_color_inside = :white,
    foreground_color_border = :black,
    foreground_color_axis   = :black,
    foreground_color_text   = :black,
    foreground_color_guide  = :black,
    tick_direction          = :out,
    grid                    = false,
    minorticks              = false,
    linewidth               = 1.5,
    legend_background_color = :white,
    legend_foreground_color = :black,
)

c1 = "#1B263B"
c2 = "#FF6B6B"
c3 = "#D4A017"
c4 = "#06D6A0"
c5 = "#3399FF"
c6 = "#FF66CC"
c7 = "#CCFF66"
c8 = "#FF3300"

d = CSV.read("C:\\Users\\ellio\\git\\gcn-dmpc\\plots\\plot-parity\\parity-data.csv", DataFrame)

fnt = Plots.font("Cambria", 12)

colors = [c1, c2, c3, c4, c5, c6, c7, c8]
point_colors = [colors[p] for p in d.pred]

sse_cmpc = sum((d.pi_class .- d.pi_cmpc).^2)
sse_best = sum((d.pi_class .- d.pi_best).^2)
println("SSE (pi_class vs pi_cmpc): ", sse_cmpc)
println("SSE (pi_class vs pi_best): ", sse_best)

scatter(d.pi_cmpc, d.pi_class, c = point_colors, label = false, dpi = 600,
    xlabel = L"\pi_{\mathrm{CMPC}}", ylabel = L"\pi_{\mathrm{class}}", markerstrokewidth = 0.5, markerstrokecolor = :black, alpha = 0.8,
    xscale = :log10, yscale = :log10, tickfont = fnt, guidefont = fnt,
    xformatter = _ -> "", yformatter = _ -> "", size = (500, 500),
    grid = true, gridcolor = :lightgray, gridalpha = 0.5, gridlinewidth = 0.5,
    minorticks = 9, minorgrid = true, minorgridcolor = :lightgray, minorgridalpha = 0.3, minorgridlinewidth = 0.3)
lims = (min(minimum(d.pi_cmpc), minimum(d.pi_class)), max(maximum(d.pi_cmpc), maximum(d.pi_class)))
plot!([lims[1], lims[2]], [lims[1], lims[2]], c = :black, lw = 1.5, ls = :dash, label = false)

savefig("C:\\Users\\ellio\\git\\gcn-dmpc\\plots\\plot-parity\\parity-plot-cmpc.png")

scatter(d.pi_best, d.pi_class, c = point_colors, label = false, dpi = 600,
    xlabel = L"\pi_{\mathrm{best}}", ylabel = L"\pi_{\mathrm{class}}", markerstrokewidth = 0.5, markerstrokecolor = :black, alpha = 0.8,
    xscale = :log10, yscale = :log10, tickfont = fnt, guidefont = fnt,
    xformatter = _ -> "", yformatter = _ -> "", size = (500, 500),
    grid = true, gridcolor = :lightgray, gridalpha = 0.5, gridlinewidth = 0.5,
    minorticks = 9, minorgrid = true, minorgridcolor = :lightgray, minorgridalpha = 0.3, minorgridlinewidth = 0.3)
lims2 = (min(minimum(d.pi_best), minimum(d.pi_class)), max(maximum(d.pi_best), maximum(d.pi_class)))
plot!([lims2[1], lims2[2]], [lims2[1], lims2[2]], c = :black, lw = 1.5, ls = :dash, label = false)

savefig("C:\\Users\\ellio\\git\\gcn-dmpc\\plots\\plot-parity\\parity-plot-best.png")