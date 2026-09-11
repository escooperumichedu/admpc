using JuMP, Ipopt, NLsolve, Plots, DifferentialEquations, Printf, LinearAlgebra, LsqFit, Statistics

include("params.jl")

# ── Reactor dynamics ─────────────────────────────────────────────────────────

function reactor_dynamics!(du, u, p, t)
    V, T = u
    q_h, q_c = p
    du[1] = q_h + q_c - q_outlet
    du[2] = (q_h / V) * (T_h - T) + (q_c / V) * (T_c - T)
end

function simulate_input_horizon(q_h_opt, q_c_opt)
    V_vec = zeros(N + 1);  V_vec[1] = V_init
    T_vec = zeros(N + 1);  T_vec[1] = T_init
    for ind = 1:N
        x0   = [V_vec[ind], T_vec[ind]]
        prob = ODEProblem(reactor_dynamics!, x0, (0.0, dt), (q_h_opt[ind], q_c_opt[ind]))
        soln = solve(prob, Rosenbrock23(), alg_hints=[:stiff],
                     reltol=1e-8, abstol=1e-8, save_everystep=false)
        V_vec[ind+1] = soln.u[end][1]
        T_vec[ind+1] = soln.u[end][2]
    end
    return V_vec, T_vec
end

# ── Decentralized MPC 1: subsystem 1 owns {q_h, V},  subsystem 2 owns {q_c, T} ──
# No coordination — each subsystem treats the other's input as fixed at setpoint.

function decentralized_controller_1()
    # Subsystem 1: controls q_h to regulate V; assumes q_c = q_c_sp
    q_h_opt = let
        MPC = Model(Ipopt.Optimizer);  set_silent(MPC)
        JuMP.@variables MPC begin
            q_h[k=0:N], (lower_bound=0.1, upper_bound=10 * q_h_sp)
            V[k=0:N],   (lower_bound=0.1, upper_bound=20.0)
        end
        @constraint(MPC, V[0] == V_init)
        @NLconstraints MPC begin
            dV_dt[k=0:N-1], V[k+1] == V[k] + (q_h[k] + q_c_sp - q_outlet) * dt
        end
        @NLobjective(MPC, Min, sum(w.vol  * (V[k]   - V_sp)^2  +
                                   w.flow * (q_h[k] - q_h_sp)^2 for k = 0:N))
        JuMP.optimize!(MPC)
        Vector(JuMP.value.(q_h))
    end

    # Subsystem 2: controls q_c to regulate T; assumes q_h = q_h_sp and V = V_sp
    q_c_opt = let
        MPC = Model(Ipopt.Optimizer);  set_silent(MPC)
        JuMP.@variables MPC begin
            q_c[k=0:N], (lower_bound=0.1, upper_bound=10 * q_c_sp)
            T[k=0:N],   (lower_bound=200, upper_bound=500.0)
        end
        @constraint(MPC, T[0] == T_init)
        @NLconstraints MPC begin
            dT_dt[k=0:N-1], T[k+1] == T[k] + ((q_h_sp / V_sp) * (T_h - T[k]) +
                                                (q_c[k]  / V_sp) * (T_c - T[k])) * dt
        end
        @NLobjective(MPC, Min, sum(w.temp * (T[k]   - T_sp)^2  +
                                   w.flow * (q_c[k] - q_c_sp)^2 for k = 0:N))
        JuMP.optimize!(MPC)
        Vector(JuMP.value.(q_c))
    end

    return q_h_opt, q_c_opt
end

# ── Decentralized MPC 2: subsystem 1 owns {q_h, T},  subsystem 2 owns {q_c, V} ──
# No coordination — each subsystem treats the other's input as fixed at setpoint.

function decentralized_controller_2()
    # Subsystem 1: controls q_h to regulate T; assumes q_c = q_c_sp and V = V_sp
    q_h_opt = let
        MPC = Model(Ipopt.Optimizer);  set_silent(MPC)
        JuMP.@variables MPC begin
            q_h[k=0:N], (lower_bound=0.1, upper_bound=10 * q_h_sp)
            T[k=0:N],   (lower_bound=200, upper_bound=500.0)
        end
        @constraint(MPC, T[0] == T_init)
        @NLconstraints MPC begin
            dT_dt[k=0:N-1], T[k+1] == T[k] + ((q_h[k]  / V_sp) * (T_h - T[k]) +
                                                (q_c_sp  / V_sp) * (T_c - T[k])) * dt
        end
        @NLobjective(MPC, Min, sum(w.temp * (T[k]   - T_sp)^2  +
                                   w.flow * (q_h[k] - q_h_sp)^2 for k = 0:N))
        JuMP.optimize!(MPC)
        Vector(JuMP.value.(q_h))
    end

    # Subsystem 2: controls q_c to regulate V; assumes q_h = q_h_sp
    q_c_opt = let
        MPC = Model(Ipopt.Optimizer);  set_silent(MPC)
        JuMP.@variables MPC begin
            q_c[k=0:N], (lower_bound=0.1, upper_bound=10 * q_c_sp)
            V[k=0:N],   (lower_bound=0.1, upper_bound=20.0)
        end
        @constraint(MPC, V[0] == V_init)
        @NLconstraints MPC begin
            dV_dt[k=0:N-1], V[k+1] == V[k] + (q_h_sp + q_c[k] - q_outlet) * dt
        end
        @NLobjective(MPC, Min, sum(w.vol  * (V[k]   - V_sp)^2  +
                                   w.flow * (q_c[k] - q_c_sp)^2 for k = 0:N))
        JuMP.optimize!(MPC)
        Vector(JuMP.value.(q_c))
    end

    return q_h_opt, q_c_opt
end

# ── τ-averaged RGA and cosine similarity (recomputed here for display) ────────

function linearized_dynamics!(du, u, p, t)
    T_ = u[1]
    q_h_, q_c_, V, T, q_h, q_c = p
    V_ = q_h_ + q_c_
    du[1] = ((q_c + q_h) * T - q_c * T_c - q_h * T_h) / (V^2) * V_ +
            (-(q_c + q_h) / V) * T_ +
            (-(T - T_h) / V) * q_h_ +
            (-(T - T_c) / V) * q_c_
end

function calculateRGA(V, T, q_h, q_c)
    G = zeros(2, 2)
    for k = 1:2
        local q_h_ = (k == 1) ? 0.01 : 0
        local q_c_ = (k == 2) ? 0.01 : 0
        prob = ODEProblem(linearized_dynamics!, [0.0], (0.0, 10.0),
                          (q_h_, q_c_, V, T, q_h, q_c))
        soln = solve(prob, Rosenbrock23(), alg_hints=[:stiff],
                     reltol=1e-8, abstol=1e-8, save_everystep=false)
        G[1, k] = q_h_ + q_c_
        G[2, k] = soln.u[end][1]
    end
    return G .* transpose(pinv(G))
end

RGA_init = calculateRGA(V_init, T_init, q_h_init, q_c_init)
RGA_sp   = calculateRGA(V_sp,   T_sp,   q_h_sp,   q_c_sp)

# Volume & temperature trajectories for tau fitting
V_t = zeros(N + 1);  V_t[1] = V_init
T_t = zeros(N + 1);  T_t[1] = T_init
t_arr = (0:N) .* dt
for k = 1:N
    V_t[k+1] = V_t[k] + dt * (V_sp - V_t[k]) / tau_vol
    T_t[k+1] = T_t[k] + dt * ((q_h_sp / V_t[k]) * (T_h - T_t[k]) +
                                (q_c_sp / V_t[k]) * (T_c - T_t[k]))
end
T_inf = T_t[end];  T0 = T_t[1]
fit_T    = curve_fit((t, p) -> T_inf .+ (T0 - T_inf) .* exp.(-t ./ p[1]),
                     collect(t_arr), T_t, [1.0])
tau_temp = fit_T.param[1]

RGA_t(t) = [
    RGA_sp[1,1] + (RGA_init[1,1] - RGA_sp[1,1]) * exp(-t / tau_vol)   RGA_sp[1,2] + (RGA_init[1,2] - RGA_sp[1,2]) * exp(-t / tau_vol);
    RGA_sp[2,1] + (RGA_init[2,1] - RGA_sp[2,1]) * exp(-t / tau_temp)  RGA_sp[2,2] + (RGA_init[2,2] - RGA_sp[2,2]) * exp(-t / tau_temp)
]
RGA_tuple = RGA_t.(t_arr)
tau_averaged_RGA = [
    mean(RGA_tuple[i][1,1] for i in 1:N+1)   mean(RGA_tuple[i][1,2] for i in 1:N+1);
    mean(RGA_tuple[i][2,1] for i in 1:N+1)   mean(RGA_tuple[i][2,2] for i in 1:N+1)
]
row_V   = tau_averaged_RGA[1, :]
row_T   = tau_averaged_RGA[2, :]
cos_sim = csim(row_V, row_T)

println()
println("τ-averaged RGA (Λ_tau):")
@printf("  [%6.3f  %6.3f]   ← V-control row\n  [%6.3f  %6.3f]   ← T-control row\n",
        tau_averaged_RGA[1,1], tau_averaged_RGA[1,2],
        tau_averaged_RGA[2,1], tau_averaged_RGA[2,2])
@printf("Cosine similarity (V-row, T-row): %.4f\n", cos_sim)
if cos_sim > 0.5
    println("  → HIGH: controlling one output benefits the other. Decomposition has little impact.")
elseif cos_sim > 0.0
    println("  → MODERATE: some alignment, but decomposition still matters.")
else
    println("  → LOW / NEGATIVE: controlling one output hurts the other. Decomposition is critical.")
end

# ── Run both decentralized MPC strategies ────────────────────────────────────

println("\nRunning Decentralized MPC 1  (subsystem 1: {q_h→V}  |  subsystem 2: {q_c→T}) ...")
q_h_opt1, q_c_opt1 = decentralized_controller_1()
V1, T1 = simulate_input_horizon(q_h_opt1, q_c_opt1)
ISE1 = sum(w.temp * (T1 .- T_sp).^2 + w.vol * (V1 .- V_sp).^2)
ISC1 = sum(w.flow * (q_h_opt1 .- q_h_sp).^2 + w.flow * (q_c_opt1 .- q_c_sp).^2)
PI1  = ISE1 + ISC1

println("Running Decentralized MPC 2  (subsystem 1: {q_h→T}  |  subsystem 2: {q_c→V}) ...")
q_h_opt2, q_c_opt2 = decentralized_controller_2()
V2, T2 = simulate_input_horizon(q_h_opt2, q_c_opt2)
ISE2 = sum(w.temp * (T2 .- T_sp).^2 + w.vol * (V2 .- V_sp).^2)
ISC2 = sum(w.flow * (q_h_opt2 .- q_h_sp).^2 + w.flow * (q_c_opt2 .- q_c_sp).^2)
PI2  = ISE2 + ISC2

println()
@printf("Decentralized MPC 1 │ ISE=%.2f  ISC=%.2f  PI=%.2f\n", ISE1, ISC1, PI1)
@printf("Decentralized MPC 2 │ ISE=%.2f  ISC=%.2f  PI=%.2f\n", ISE2, ISC2, PI2)
@printf("ΔPI                 │ PI1 - PI2 = %.2f  (negative → Decentralized MPC 1 better)\n", PI1 - PI2)

if PI1 < PI2
    println("Best decomposition: Decentralized MPC 1  {q_h→V, q_c→T}")
else
    println("Best decomposition: Decentralized MPC 2  {q_h→T, q_c→V}")
end

# ── Plots ─────────────────────────────────────────────────────────────────────
t_steps = 1:(N+1)

pqh = plot(t_steps, q_h_opt1, label="Dec. MPC 1", lw=2, linetype=:steppost,
           ylabel="Hot stream flow (qh)", xlabel="Step")
plot!(t_steps, q_h_opt2, label="Dec. MPC 2", lw=2, linetype=:steppost)

pqc = plot(t_steps, q_c_opt1, label="Dec. MPC 1", lw=2, linetype=:steppost,
           ylabel="Cold stream flow (qc)", xlabel="Step")
plot!(t_steps, q_c_opt2, label="Dec. MPC 2", lw=2, linetype=:steppost)

pv = plot(t_steps, V1, label="Dec. MPC 1", lw=2, ylabel="Volume (V)", xlabel="Step")
plot!(t_steps, V2, label="Dec. MPC 2", lw=2, ls=:dash)
hline!([V_sp], color=:gold, ls=:dot, lw=2, label="Set point")

pt = plot(t_steps, T1, label="Dec. MPC 1", lw=2, ylabel="Temperature (T)", xlabel="Step")
plot!(t_steps, T2, label="Dec. MPC 2", lw=2)
hline!([T_sp], color=:gold, ls=:dot, lw=2, label="Set point")

display(plot(pqh, pqc, pt, pv, layout=(2,2), size=(900,600),
             title=["Hot flow" "Cold flow" "Temperature" "Volume"]))
