### Initialize pkgs
using JuMP, Ipopt, Plots, DataFrames, LinearAlgebra, Statistics, DifferentialEquations, LaTeXStrings, CSV
include("cmpc.jl")
include("dmpc_1.jl")
include("dmpc_2.jl")
include("dmpc_3.jl")
include("dmpc_4.jl")
include("dmpc_5.jl")
include("dmpc_6.jl")
include("dmpc_7.jl")
include("dmpc_8.jl")

# data = CSV.read("C:\\Users\\goson\\Desktop\\git\\gcn-dmpc\\reactor-separator\\set-point-tracking\\set-point-mpc-probs.csv", DataFrame)
# data = CSV.read("C:\\Users\\ellio\\git\\gcn-dmpc\\second-iteration-of-sims\\DMPC\\set-point-mpc-probs-n3000.csv", DataFrame)


# data = CSV.read("C:\\Users\\ellio\\OneDrive\\Documents\\git\\gcn-dmpc\\case-study-1\\DMPC\\problem-instances\\sec-n-1500-random-sample\\set-point-mpc-probs-n1500.csv", DataFrame)
data = CSV.read("C:\\Users\\ellio\\OneDrive\\Documents\\git\\gcn-dmpc\\case-study-1\\DMPC\\problem-instances\\sec-n-1500-random-sample\\set-point-mpc-probs-n3000.csv", DataFrame)

# Select which instance you want to run

for h = 1

    println("Sample $h...")

    global ISE_vec = zeros(8)
    global ISC_vec = zeros(8)
    global PI_vec = zeros(8)

    global problem_instance = data[h, 1:end]

    for kk in [1, 2, 3, 4, 5, 6, 7, 8]

        println("DMPC $kk...")

        # Horizon and discretizations
        global N = 80 * 3 # Control horizon2
        global dt = 0.025 / 3 # Sampling time of the system
        global tau = 0.05 * 3 # Volume holdup time constant

        # Initial condition and set point
        V_1_init = problem_instance[1]
        V_2_init = problem_instance[2]
        V_3_init = problem_instance[3]

        T_1_init = problem_instance[4]
        T_2_init = problem_instance[5]
        T_3_init = problem_instance[6]

        x_A1_init = problem_instance[7]
        x_A2_init = problem_instance[8]
        x_A3_init = problem_instance[9]

        x_B1_init = problem_instance[10]
        x_B2_init = problem_instance[11]
        x_B3_init = problem_instance[12]

        V_1_sp = problem_instance[13]
        V_2_sp = problem_instance[14]
        V_3_sp = problem_instance[15]

        T_1_sp = problem_instance[16]
        T_2_sp = problem_instance[17]
        T_3_sp = problem_instance[18]

        x_A1_sp = problem_instance[19]
        x_A2_sp = problem_instance[20]
        x_A3_sp = problem_instance[21]

        x_B1_sp = problem_instance[22]
        x_B2_sp = problem_instance[23]
        x_B3_sp = problem_instance[24]

        F_1_sp = problem_instance[25]
        F_2_sp = problem_instance[26]
        F_3_sp = problem_instance[27]
        F_f1_sp = problem_instance[28]
        F_f2_sp = problem_instance[29]
        F_R_sp = problem_instance[30]
        Q_1_sp = problem_instance[31]
        Q_2_sp = problem_instance[32]
        Q_3_sp = problem_instance[33]

        x_init = [V_1_init, V_2_init, V_3_init, T_1_init, T_2_init, T_3_init, x_A1_init, x_A2_init, x_A3_init, x_B1_init, x_B2_init, x_B3_init]
        x_sp = [V_1_sp, V_2_sp, V_3_sp, T_1_sp, T_2_sp, T_3_sp, x_A1_sp, x_A2_sp, x_A3_sp, x_B1_sp, x_B2_sp, x_B3_sp]
        u_sp = [F_1_sp, F_2_sp, F_3_sp, F_f1_sp, F_f2_sp, F_R_sp, Q_1_sp, Q_2_sp, Q_3_sp]

        if kk == 1
            u_opt, x_pred = dmpc1_mpc(x_init, x_sp, u_sp, 300)
        elseif kk == 2
            u_opt, x_pred = dmpc2_mpc(x_init, x_sp, u_sp, 300)
        elseif kk == 3
            u_opt, x_pred = dmpc3_mpc(x_init, x_sp, u_sp, 300)
        elseif kk == 4
            u_opt, x_pred = dmpc4_mpc(x_init, x_sp, u_sp, 300)
        elseif kk == 5
            u_opt, x_pred = dmpc5_mpc(x_init, x_sp, u_sp, 300)
        elseif kk == 6
            u_opt, x_pred = dmpc6_mpc(x_init, x_sp, u_sp, 100)
        elseif kk == 7
            u_opt, x_pred = dmpc7_mpc(x_init, x_sp, u_sp, 100)
        elseif kk == 8
            u_opt, x_pred = dmpc8_mpc(x_init, x_sp, u_sp, 100)
        end

        local F_1 = u_opt[1]
        local F_2 = u_opt[2]
        local F_3 = u_opt[3]
        local F_f1 = u_opt[4]
        local F_f2 = u_opt[5]
        local F_R = u_opt[6]
        local Q_1 = u_opt[7]
        local Q_2 = u_opt[8]
        local Q_3 = u_opt[9]


        local V_1 = zeros(N + 1)
        local V_2 = zeros(N + 1)
        local V_3 = zeros(N + 1)

        local T_1 = zeros(N + 1)
        local T_2 = zeros(N + 1)
        local T_3 = zeros(N + 1)

        local x_A1 = zeros(N + 1)
        local x_A2 = zeros(N + 1)
        local x_A3 = zeros(N + 1)

        local x_B1 = zeros(N + 1)
        local x_B2 = zeros(N + 1)
        local x_B3 = zeros(N + 1)

        V_1[1] = V_1_init
        V_2[1] = V_2_init
        V_3[1] = V_3_init

        T_1[1] = T_1_init
        T_2[1] = T_2_init
        T_3[1] = T_3_init

        x_A1[1] = x_A1_init
        x_A2[1] = x_A2_init
        x_A3[1] = x_A3_init

        x_B1[1] = x_B1_init
        x_B2[1] = x_B2_init
        x_B3[1] = x_B3_init


        function reactor_dynamics(u, p, t)


            V1, V2, V3,
            T1, T2, T3,
            xA1, xA2, xA3,
            xB1, xB2, xB3 = u

            dV1 = (F_f1[k] + F_R[k] - F_1[k])
            dV2 = (F_f2[k] + F_1[k] - F_2[k])
            dV3 = (F_2[k] - F_R[k] * epsilon - F_R[k] - F_3[k])

            dT1 = ((F_f1[k] / V1) * (T_f - T1) + (F_R[k] / V1) * (T3 - T1) + Q_1[k] / (rho * cp * V1) - (m / cp) * H_1 * k_1 * exp(-E_1 / (R * T1)) * xA1 - (m / cp) * H_2 * k_2 * exp(-E_2 / (R * T1)) * xB1)
            dT2 = ((F_1[k] / V2) * (T1 - T2) + (F_f2[k] / V2) * (T_f - T2) + Q_2[k] / (rho * cp * V2) - (m / cp) * H_1 * k_1 * exp(-E_1 / (R * T2)) * xA2 - (m / cp) * H_2 * k_2 * exp(-E_2 / (R * T2)) * xB2)
            dT3 = ((F_2[k] / V3) * (T2 - T3) + Q_3[k] / (rho * cp * V3))

            dxA1 = ((F_f1[k] / V1) * (x_Af - xA1) + (F_R[k] / V1) * ((alpha_A * xA3) / (alpha_A * xA3 + alpha_B * xB3 + alpha_C * (1 - xA3 - xB3)) - xA1) - k_1 * exp(-E_1 / (R * T1)) * xA1)
            dxA2 = ((F_1[k] / V2) * (xA1 - xA2) + (F_f2[k] / V2) * (x_Af - xA2) - k_1 * exp(-E_1 / (R * T2)) * xA2)
            dxA3 = ((F_2[k] / V3) * (xA2 - xA3) - ((F_R[k] + F_R[k] * epsilon) / V3) * ((alpha_A * xA3) / (alpha_A * xA3 + alpha_B * xB3 + alpha_C * (1 - xA3 - xB3)) - xA3))

            dxB2 = ((F_1[k] / V2) * (xB1 - xB2) + (F_f2[k] / V2) * (x_Bf - xB2) + k_1 * exp(-E_1 / (R * T2)) * xA2 - k_2 * exp(-E_2 / (R * T2)) * xB2)
            dxB1 = ((F_f1[k] / V1) * (x_Bf - xB1) + (F_R[k] / V1) * ((alpha_B * xB3) / (alpha_A * xA3 + alpha_B * xB3 + alpha_C * (1 - xA3 - xB3)) - xB1) + k_1 * exp(-E_1 / (R * T1)) * xA1 - k_2 * exp(-E_2 / (R * T1)) * xB1)
            dxB3 = ((F_2[k] / V3) * (xB2 - xB3) - ((F_R[k] + F_R[k] * epsilon) / V3) * ((alpha_B * xB3) / (alpha_A * xA3 + alpha_B * xB3 + alpha_C * (1 - xA3 - xB3)) - xB3))

            return [dV1, dV2, dV3, dT1, dT2, dT3, dxA1, dxA2, dxA3, dxB1, dxB2, dxB3]
        end


        function timeout_condition(u, t, integrator)
            return (time() - integrator.p.start_time) > 10.5  # stop after 10.5 seconds
        end

        function timeout_affect!(integrator)
            terminate!(integrator)
        end

        cb = DiscreteCallback(timeout_condition, timeout_affect!)

        for ind = 1:N

            global k = ind

            tspan = (0.0, dt)
            x0 = [V_1[k], V_2[k], V_3[k], T_1[k], T_2[k], T_3[k], x_A1[k], x_A2[k], x_A3[k], x_B1[k], x_B2[k], x_B3[k]]
            prob = ODEProblem(reactor_dynamics, x0, tspan, (; start_time=time()))
            soln = solve(prob, Rosenbrock23(), callback=cb, alg_hints=[:stiff], reltol=1e-8, abstol=1e-8, save_everystep=false)

            V_1[k+1] = soln.u[end][1]
            V_2[k+1] = soln.u[end][2]
            V_3[k+1] = soln.u[end][3]

            T_1[k+1] = soln.u[end][4]
            T_2[k+1] = soln.u[end][5]
            T_3[k+1] = soln.u[end][6]

            x_A1[k+1] = soln.u[end][7]
            x_A2[k+1] = soln.u[end][8]
            x_A3[k+1] = soln.u[end][9]

            x_B1[k+1] = soln.u[end][10]
            x_B2[k+1] = soln.u[end][11]
            x_B3[k+1] = soln.u[end][12]

        end

        # ISE
        ISE = sum((w.con * (x_A1[k] - x_A1_sp)^2 + w.con * (x_A2[k] - x_A2_sp)^2 + w.con * (x_A3[k] - x_A3_sp)^2 +
                   w.con * (x_B1[k] - x_B1_sp)^2 + w.con * (x_B2[k] - x_B2_sp)^2 + w.con * (x_B3[k] - x_B3_sp)^2 +
                   w.temp * (T_1[k] - T_1_sp)^2 + w.temp * (T_2[k] - T_2_sp)^2 + w.temp * (T_3[k] - T_3_sp)^2 +
                   w.vol * (V_1[k] - V_1_sp)^2 + w.vol * (V_2[k] - V_2_sp)^2 + w.vol * (V_3[k] - V_3_sp)^2) for k = 1:N+1) * dt

        # ISC
        ISC = sum((w.flow * (F_f1[k] - F_f1_sp)^2 + w.flow * (F_f2[k] - F_f2_sp)^2 + w.flow * (F_1[k] - F_1_sp)^2 +
                   w.flow * (F_2[k] - F_2_sp)^2 + w.flow * (F_3[k] - F_3_sp)^2 + w.flow * (F_R[k] - F_R_sp)^2 +
                   w.heat * (Q_1[k] - Q_1_sp)^2 + w.heat * (Q_2[k] - Q_2_sp)^2 + w.heat * (Q_3[k] - Q_3_sp)^2) for k = 1:N) * dt

        PI = ISE + ISC

        ISE_vec[kk] = ISE
        ISC_vec[kk] = ISC
        PI_vec[kk] = PI

        traj = DataFrame([V_1, V_2, V_3, T_1, T_2, T_3, x_A1, x_A2, x_A3, x_B1, x_B2, x_B3, F_1, F_2, F_3, F_f1, F_f2, F_R, Q_1, Q_2, Q_3], [:V_1, :V_2, :V_3, :T_1, :T_2, :T_3, :x_A1, :x_A2, :x_A3, :x_B1, :x_B2, :x_B3, :F_1, :F_2, :F_3, :F_f1, :F_f2, :F_R, :Q_1, :Q_2, :Q_3])

        println("
        $ISE
        $ISC
        $PI
        ")

        copy_paste = vcat(ISE_vec', ISC_vec', PI_vec')
        copy_paste = DataFrame(copy_paste, [:DMPC1, :DMPC2, :DMPC3, :DMPC4, :DMPC5, :DMPC6, :DMPC7, :DMPC8])

        # CSV.write("C:\\Users\\ellio\\git\\rerun\\traj-$h-dmpc-$kk.csv", traj)

    end

    copy_paste = vcat(ISE_vec', ISC_vec', PI_vec')
    copy_paste = DataFrame(copy_paste, [:DMPC1, :DMPC2, :DMPC3, :DMPC4, :DMPC5, :DMPC6, :DMPC7, :DMPC8])
    # CSV.write("C:\\Users\\ellio\\git\\rerun\\set-point-prob-$h-matrix.csv", copy_paste)

end

