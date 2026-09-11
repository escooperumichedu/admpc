# ============================================================
#  Run CENTRALIZED MPC (CMPC) for both the high- and low-
#  cosine-similarity instances and export trajectories/stats
#  to revision/, matching the CSV schema used by the DMPC
#  states-inputs and coordination-sweep files.
#
#  CMPC logic is copied from cmpc.jl but parameterized by the
#  scenario so it does NOT depend on the current params.jl.
# ============================================================
using JuMP, Ipopt, NLsolve, Printf

const T_h = 400.0
const T_c = 300.0
const dt  = 0.1

# Scenario parameters (must match params.jl presets used for the DMPC CSVs)
struct Scenario
    name::String
    N::Int
    q_h_init::Float64; q_c_init::Float64; V_init::Float64
    q_h_sp::Float64;   q_c_sp::Float64;   V_sp::Float64
end

hisim  = Scenario("hisim",  60, 2.0, 8.0, 5.0, 11.0, 6.0, 5.5)
lowsim = Scenario("lowsim", 60, 2.0, 8.0, 5.0, 0.5,  9.0, 8.1)

ss_temp(q_h,q_c,V) = nlsolve(
    (F,x)->(F[1]=(q_h/V)*(T_h-x[1])+(q_c/V)*(T_c-x[1])), [(T_h+T_c)/2]).zero[1]

# Weights: hardcoded to EXACTLY match params.jl (line 81) so CMPC is on the
# same objective scale as the decentralized / distributed DMPC CSVs.
#   w = Weights(w_vol, w_temp, w_flow) = (10.0, 0.001, 10.0)
const W_VOL  = 10.0
const W_TEMP = 0.001
const W_FLOW = 10.0

# ── Derived quantities per scenario ─────────────────────────
function derive(s::Scenario)
    q_outlet = s.q_h_sp + s.q_c_sp
    T_sp   = ss_temp(s.q_h_sp,   s.q_c_sp,   s.V_sp)
    T_init = ss_temp(s.q_h_init, s.q_c_init, s.V_init)
    (; q_outlet, T_sp, T_init, w_vol=W_VOL, w_temp=W_TEMP, w_flow=W_FLOW)
end

# ── Plant simulation (RK4; matches the DMPC sweep harness) ───
function simulate(s, d, q_h_opt, q_c_opt)
    f(u,qh,qc) = [ qh+qc-d.q_outlet,
                   (qh/u[1])*(T_h-u[2]) + (qc/u[1])*(T_c-u[2]) ]
    V=zeros(s.N+1); V[1]=s.V_init; T=zeros(s.N+1); T[1]=d.T_init
    for k=1:s.N
        u=[V[k],T[k]]; qh=q_h_opt[k]; qc=q_c_opt[k]
        k1=f(u,qh,qc); k2=f(u.+0.5dt.*k1,qh,qc)
        k3=f(u.+0.5dt.*k2,qh,qc); k4=f(u.+dt.*k3,qh,qc)
        un=u .+ (dt/6).*(k1 .+ 2 .*k2 .+ 2 .*k3 .+ k4)
        V[k+1]=un[1]; T[k+1]=un[2]
    end
    V,T
end

# ── Centralized MPC (verbatim structure from cmpc.jl) ───────
function centralized_controller(s, d)
    N=s.N
    MPC = Model(Ipopt.Optimizer); set_silent(MPC)
    # Bounds match the decentralized / distributed DMPC controllers exactly
    # (compare-1-vs-2.jl): flows 0.1 .. 10*q_sp, V 0.1 .. 20, T 200 .. 500.
    JuMP.@variables MPC begin
        q_h[k=0:N], (lower_bound=0.1, upper_bound=10 * s.q_h_sp)
        q_c[k=0:N], (lower_bound=0.1, upper_bound=10 * s.q_c_sp)
        V[k=0:N],   (lower_bound=0.1, upper_bound=20.0)
        T[k=0:N],   (lower_bound=200.0, upper_bound=500.0)
    end
    @constraints MPC begin
        V[0] == s.V_init
        T[0] == d.T_init
    end
    @NLconstraints MPC begin
        dV_dt[k=0:N-1], V[k+1] == V[k] + (q_h[k] + q_c[k] - d.q_outlet) * dt
        dT_dt[k=0:N-1], T[k+1] == T[k] + ((q_h[k]/V[k])*(T_h-T[k]) +
                                          (q_c[k]/V[k])*(T_c-T[k])) * dt
    end
    @NLobjective(MPC, Min, sum(
        d.w_vol *(V[k]  -s.V_sp)^2 + d.w_temp*(T[k]  -d.T_sp)^2 +
        d.w_flow*(q_h[k]-s.q_h_sp)^2 + d.w_flow*(q_c[k]-s.q_c_sp)^2 for k=0:N))
    JuMP.optimize!(MPC)
    Vector(JuMP.value.(q_h)), Vector(JuMP.value.(q_c))
end

# ── Run one instance and export CSVs ────────────────────────
function run_instance(s::Scenario)
    d = derive(s)
    q_h_opt, q_c_opt = centralized_controller(s, d)
    V_cl, T_cl = simulate(s, d, q_h_opt, q_c_opt)
    ISE = sum(d.w_temp*(T_cl.-d.T_sp).^2 + d.w_vol*(V_cl.-s.V_sp).^2)
    ISC = sum(d.w_flow*(q_h_opt.-s.q_h_sp).^2 + d.w_flow*(q_c_opt.-s.q_c_sp).^2)
    PI  = ISE + ISC

    suffix = "_" * s.name
    # trajectory CSV — schema matches compare-1-vs-2-states-inputs_*.csv
    open(joinpath(@__DIR__, "cmpc-states-inputs"*suffix*".csv"), "w") do f
        println(f, "step,time,CMPC_qh,CMPC_qc,CMPC_V,CMPC_T,qh_sp,qc_sp,V_sp,T_sp")
        for k in 1:(s.N+1)
            t_k = (k-1)*dt
            qh = k<=s.N ? q_h_opt[k] : q_h_opt[s.N]
            qc = k<=s.N ? q_c_opt[k] : q_c_opt[s.N]
            @printf(f, "%d,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n",
                    k-1, t_k, qh, qc, V_cl[k], T_cl[k],
                    s.q_h_sp, s.q_c_sp, s.V_sp, d.T_sp)
        end
    end
    # stats CSV
    open(joinpath(@__DIR__, "cmpc-stats"*suffix*".csv"), "w") do f
        println(f, "instance,PI,ISE,ISC,T_sp,T_init,q_outlet")
        @printf(f, "%s,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n",
                s.name, PI, ISE, ISC, d.T_sp, d.T_init, d.q_outlet)
    end
    @printf("CMPC %-7s │ Tsp=%.1f  ISE=%.3f  ISC=%.3f  PI=%.3f  → cmpc-states-inputs%s.csv\n",
            s.name, d.T_sp, ISE, ISC, PI, suffix)
end

println("Running CMPC for both instances ...\n")
run_instance(hisim)
run_instance(lowsim)
println("\nDone. CSVs written to revision/.")
