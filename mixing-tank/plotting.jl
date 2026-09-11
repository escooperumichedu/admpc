using CSV, DataFrames, LaTeXStrings, Plots
import Plots: mm

# ============================================================
#  Plot the DMPC1 vs DMPC2 trajectories (states & inputs) and
#  the ISE/ISC/PI-vs-coordinations sweep, in the MATLAB-style
#  formatting of high-csim-instance/plotting.jl.
#
#  Toggle INSTANCE between "hisim" and "lowsim" to plot either
#  the high- or low-cosine-similarity CSVs exported by
#  sweep-coordinations.jl and compare-1-vs-2.jl.
# ============================================================

# ── Which instance to plot: "hisim" or "lowsim" ─────────────
const INSTANCE = "hisim"     # change to "lowsim" for the low cos-sim data

# MATLAB-style defaults
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
    linewidth               = 0.5,
    legend_background_color = :white,
    legend_foreground_color = :black,
)

# Per-decomposition colors (shared across all plots)
c_dmpc1 = "#1B263B"   # dark navy
c_dmpc2 = "#E8573F"   # burnt orange-red
c_sp    = "#D4A017"   # amber (setpoints)

# Per-metric colors for the coordination sweep
c_ise = "#1B263B"     # dark navy
c_isc = "#D4A017"     # amber
c_pi  = "#E8573F"     # burnt orange-red

fnt = Plots.font("computer modern", 12)

dir = "C:\\Users\\ellio\\OneDrive\\Documents\\git\\gcn-dmpc\\mixing-tank\\revision\\"
suffix = "_" * INSTANCE        # "_hisim" or "_lowsim"

# Legends are shown only for the hisim case; lowsim plots are drawn legend-free.
show_legend = (INSTANCE == "hisim")

traj  = CSV.read(dir * "compare-1-vs-2-states-inputs" * suffix * ".csv", DataFrame)
sweep = CSV.read(dir * "coordination-sweep" * suffix * ".csv",          DataFrame)

# ── Temperature: DMPC1 vs DMPC2 (+ setpoint) ────────────────
pT = plot(traj.time, traj.DMPC1_T, label = "DMPC1", lw = 3, c = c_dmpc1, ls = :solid)
plot!(traj.time, traj.DMPC2_T,     label = "DMPC2", lw = 3, c = c_dmpc2, ls = :solid)
plot!(traj.time, traj.T_sp,        label = L"T_{sp}", lw = 2, c = c_sp,  ls = :dash,
      xlabel = L"t", ylabel = L"T", legend = show_legend ? :best : false,
      dpi = 300, tickfont = fnt, guidefont = fnt, size = (500, 300))
savefig(dir * "traj-T" * suffix * ".png")

# ── Volume: DMPC1 vs DMPC2 (+ setpoint) ─────────────────────
pV = plot(traj.time, traj.DMPC1_V, label = "DMPC1", lw = 3, c = c_dmpc1, ls = :solid)
plot!(traj.time, traj.DMPC2_V,     label = "DMPC2", lw = 3, c = c_dmpc2, ls = :solid)
plot!(traj.time, traj.V_sp,        label = L"V_{sp}", lw = 2, c = c_sp,  ls = :dash,
      xlabel = L"t", ylabel = L"V", legend = show_legend ? :best : false,
      dpi = 300, tickfont = fnt, guidefont = fnt, size = (500, 300))
savefig(dir * "traj-V" * suffix * ".png")

# ── Flows: color = decomposition, solid = hot (q_h), dot = cold (q_c) ──
pq = plot(traj.time, traj.DMPC1_qh, label = "DMPC1 " * L"q_h", lw = 3, c = c_dmpc1, ls = :solid)
plot!(traj.time, traj.DMPC1_qc,     label = "DMPC1 " * L"q_c", lw = 3, c = c_dmpc1, ls = :dot)
plot!(traj.time, traj.DMPC2_qh,     label = "DMPC2 " * L"q_h", lw = 3, c = c_dmpc2, ls = :solid)
plot!(traj.time, traj.DMPC2_qc,     label = "DMPC2 " * L"q_c", lw = 3, c = c_dmpc2, ls = :dot,
      xlabel = L"t", ylabel = "Flow", legend = show_legend ? (0.75, 0.75) : false,
      dpi = 300, tickfont = fnt, guidefont = fnt, size = (500, 300))
savefig(dir * "traj-flows" * suffix * ".png")

# ── Combined trajectory figure (T, V, flows) ────────────────
plot(pT, pV, pq,
    layout        = (1, 3),
    size          = (1400, 300),
    dpi           = 600,
    tickfont      = fnt,
    guidefont     = fnt,
    left_margin   = 12mm,
    right_margin  = 4mm,
    bottom_margin = 10mm,
    top_margin    = 4mm,
)
savefig(dir * "traj-combined" * suffix * ".png")

# ── Coordination sweep: ISE / ISC / PI vs coordinations ─────
# color = metric, solid = DMPC1, dot = DMPC2 (matches the flows convention)
pISE = plot(sweep.coordination, sweep.DMPC1_ISE, label = "DMPC1", lw = 3, c = c_dmpc1, ls = :solid,
            marker = :circle, ms = 4)
plot!(sweep.coordination, sweep.DMPC2_ISE,       label = "DMPC2", lw = 3, c = c_dmpc2, ls = :solid,
      marker = :square, ms = 4,
      xlabel = "Coordinations", ylabel = "ISE", legend = show_legend ? :best : false,
      dpi = 300, tickfont = fnt, guidefont = fnt, size = (500, 300))

pISC = plot(sweep.coordination, sweep.DMPC1_ISC, label = "DMPC1", lw = 3, c = c_dmpc1, ls = :solid,
            marker = :circle, ms = 4)
plot!(sweep.coordination, sweep.DMPC2_ISC,       label = "DMPC2", lw = 3, c = c_dmpc2, ls = :solid,
      marker = :square, ms = 4,
      xlabel = "Coordinations", ylabel = "ISC", legend = false,
      dpi = 300, tickfont = fnt, guidefont = fnt, size = (500, 300))

pPI = plot(sweep.coordination, sweep.DMPC1_PI, label = "DMPC1", lw = 3, c = c_dmpc1, ls = :solid,
           marker = :circle, ms = 4)
plot!(sweep.coordination, sweep.DMPC2_PI,      label = "DMPC2", lw = 3, c = c_dmpc2, ls = :solid,
      marker = :square, ms = 4,
      xlabel = "Coordinations", ylabel = L"\Phi = \mathrm{ISE} + \mathrm{ISC}", legend = false,
      dpi = 300, tickfont = fnt, guidefont = fnt, size = (500, 300))

savefig(pPI, dir * "sweep-PI" * suffix * ".png")

# ── Combined sweep figure (ISE, ISC, PI) ────────────────────
plot(pISE, pISC, pPI,
    layout        = (1, 3),
    size          = (1400, 300),
    dpi           = 600,
    tickfont      = fnt,
    guidefont     = fnt,
    left_margin   = 12mm,
    right_margin  = 4mm,
    bottom_margin = 10mm,
    top_margin    = 4mm,
)
savefig(dir * "sweep-combined" * suffix * ".png")

println("Plotted INSTANCE=\"$INSTANCE\" → saved traj-*$suffix.png and sweep-*$suffix.png in revision/")
