### Initialize pkgs
using JuMP, Ipopt, Plots, DataFrames, LinearAlgebra, Statistics

# A -> B -> C taking place in a CSTR-CSTR-Separator process

# Initialize parameters for the system
# Feed stream properties
global T_f = 359.1 # K
global x_Af = 1 # unitless
global x_Bf = 0 # unitless

# Physical properties
global m = 2.79 # mol/kg
global rho = 1000 # kg/m3
global cp = 4.2 # kJ/(kg*K)
global R = 8.314e-3 # kJ/(mol*K)

# Reaction rate constants
global k_1 = 9.97e6 # 1/hr
global k_2 = 9e6 # 1/hr

# Activation energies and enthalpies of reaction
global E_1 = 50 # kJ/mol
global E_2 = 60 # kJ/mol
global H_1 = -60 # kJ/mol
global H_2 = -70 # kJ/mol

# Activity of species
global alpha_A = 5 # unitless
global alpha_B = 1 # unitless
global alpha_C = 0.5 # unitless

# Purge ratio
global epsilon = 0.02


# Structure containing the communicated inputs
mutable struct SharedInputs

    F_1
    F_2
    F_3
    F_f1
    F_f2
    F_R
    Q_1
    Q_2
    Q_3

end

# Structure containing the communicated states
mutable struct SharedStates

    V_1
    V_2
    V_3
    T_1
    T_2
    T_3
    x_A1
    x_A2
    x_A3
    x_B1
    x_B2
    x_B3

end


# Function for calculating reactor dynamics guesses at large time steps
function reactor_dynamics(u, p, t)

    V1, V2, V3,
    T1, T2, T3,
    xA1, xA2, xA3,
    xB1, xB2, xB3 = u

    dV1 = (u_F_f1[k] + u_F_R[k] - u_F_1[k])
    dV2 = (u_F_f2[k] + u_F_1[k] - u_F_2[k])
    dV3 = (u_F_2[k] - u_F_R[k] * epsilon - u_F_R[k] - u_F_3[k])

    dT1 = ((u_F_f1[k] / V1) * (T_f - T1) + (u_F_R[k] / V1) * (T3 - T1) + u_Q_1[k] / (rho * cp * V1) - (m / cp) * H_1 * k_1 * exp(-E_1 / (R * T1)) * xA1 - (m / cp) * H_2 * k_2 * exp(-E_2 / (R * T1)) * xB1)
    dT2 = ((u_F_1[k] / V2) * (T1 - T2) + (u_F_f2[k] / V2) * (T_f - T2) + u_Q_2[k] / (rho * cp * V2) - (m / cp) * H_1 * k_1 * exp(-E_1 / (R * T2)) * xA2 - (m / cp) * H_2 * k_2 * exp(-E_2 / (R * T2)) * xB2)
    dT3 = ((u_F_2[k] / V3) * (T2 - T3) + u_Q_3[k] / (rho * cp * V3))

    dxA1 = ((u_F_f1[k] / V1) * (x_Af - xA1) + (u_F_R[k] / V1) * ((alpha_A * xA3) / (alpha_A * xA3 + alpha_B * xB3 + alpha_C * (1 - xA3 - xB3)) - xA1) - k_1 * exp(-E_1 / (R * T1)) * xA1)
    dxA2 = ((u_F_1[k] / V2) * (xA1 - xA2) + (u_F_f2[k] / V2) * (x_Af - xA2) - k_1 * exp(-E_1 / (R * T2)) * xA2)
    dxA3 = ((u_F_2[k] / V3) * (xA2 - xA3) - ((u_F_R[k] + u_F_R[k] * epsilon) / V3) * ((alpha_A * xA3) / (alpha_A * xA3 + alpha_B * xB3 + alpha_C * (1 - xA3 - xB3)) - xA3))

    dxB2 = ((u_F_1[k] / V2) * (xB1 - xB2) + (u_F_f2[k] / V2) * (x_Bf - xB2) + k_1 * exp(-E_1 / (R * T2)) * xA2 - k_2 * exp(-E_2 / (R * T2)) * xB2)
    dxB1 = ((u_F_f1[k] / V1) * (x_Bf - xB1) + (u_F_R[k] / V1) * ((alpha_B * xB3) / (alpha_A * xA3 + alpha_B * xB3 + alpha_C * (1 - xA3 - xB3)) - xB1) + k_1 * exp(-E_1 / (R * T1)) * xA1 - k_2 * exp(-E_2 / (R * T1)) * xB1)
    dxB3 = ((u_F_2[k] / V3) * (xB2 - xB3) - ((u_F_R[k] + u_F_R[k] * epsilon) / V3) * ((alpha_B * xB3) / (alpha_A * xA3 + alpha_B * xB3 + alpha_C * (1 - xA3 - xB3)) - xB3))

    return [dV1, dV2, dV3, dT1, dT2, dT3, dxA1, dxA2, dxA3, dxB1, dxB2, dxB3]

end

function dmpc8_mpc(x_init, x_sp, u_sp, max_coordinations)

    V_1_init = x_init[1]
    V_2_init = x_init[2]
    V_3_init = x_init[3]

    T_1_init = x_init[4]
    T_2_init = x_init[5]
    T_3_init = x_init[6]

    x_A1_init = x_init[7]
    x_A2_init = x_init[8]
    x_A3_init = x_init[9]

    x_B1_init = x_init[10]
    x_B2_init = x_init[11]
    x_B3_init = x_init[12]

    V_1_sp = x_sp[1]
    V_2_sp = x_sp[2]
    V_3_sp = x_sp[3]

    T_1_sp = x_sp[4]
    T_2_sp = x_sp[5]
    T_3_sp = x_sp[6]

    x_A1_sp = x_sp[7]
    x_A2_sp = x_sp[8]
    x_A3_sp = x_sp[9]

    x_B1_sp = x_sp[10]
    x_B2_sp = x_sp[11]
    x_B3_sp = x_sp[12]

    F_1_sp = u_sp[1]
    F_2_sp = u_sp[2]
    F_3_sp = u_sp[3]
    F_f1_sp = u_sp[4]
    F_f2_sp = u_sp[5]
    F_R_sp = u_sp[6]
    Q_1_sp = u_sp[7]
    Q_2_sp = u_sp[8]
    Q_3_sp = u_sp[9]

    u = SharedInputs(F_1_sp .* ones(N + 1), F_2_sp .* ones(N + 1), F_3_sp .* ones(N + 1), F_f1_sp .* ones(N + 1), F_f2_sp .* ones(N + 1), F_R_sp .* ones(N + 1), Q_1_sp .* ones(N + 1), Q_2_sp .* ones(N + 1), Q_3_sp .* ones(N + 1))

    # Initialize guesses for state variables based on nominal steady state inputs at the current instance

    global u_F_1 = u.F_1
    global u_F_2 = u.F_2
    global u_F_3 = u.F_3
    global u_F_f1 = u.F_f1
    global u_F_f2 = u.F_f2
    global u_F_R = u.F_R
    global u_Q_1 = u.Q_1
    global u_Q_2 = u.Q_2
    global u_Q_3 = u.Q_3


    V_1_guess = zeros(N + 1)
    V_2_guess = zeros(N + 1)
    V_3_guess = zeros(N + 1)
    T_1_guess = zeros(N + 1)
    T_2_guess = zeros(N + 1)
    T_3_guess = zeros(N + 1)
    x_A1_guess = zeros(N + 1)
    x_A2_guess = zeros(N + 1)
    x_A3_guess = zeros(N + 1)
    x_B1_guess = zeros(N + 1)
    x_B2_guess = zeros(N + 1)
    x_B3_guess = zeros(N + 1)

    V_1_guess[1] = V_1_init
    V_2_guess[1] = V_2_init
    V_3_guess[1] = V_3_init
    T_1_guess[1] = T_1_init
    T_2_guess[1] = T_2_init
    T_3_guess[1] = T_3_init
    x_A1_guess[1] = x_A1_init
    x_A2_guess[1] = x_A2_init
    x_A3_guess[1] = x_A3_init
    x_B1_guess[1] = x_B1_init
    x_B2_guess[1] = x_B2_init
    x_B3_guess[1] = x_B3_init

    for ind = 1:N

        global k = ind

        tspan = (0.0, dt)
        x0 = [V_1_guess[k], V_2_guess[k], V_3_guess[k], T_1_guess[k], T_2_guess[k], T_3_guess[k], x_A1_guess[k], x_A2_guess[k], x_A3_guess[k], x_B1_guess[k], x_B2_guess[k], x_B3_guess[k]]
        prob = ODEProblem(reactor_dynamics, x0, tspan)
        soln = solve(prob, Rosenbrock23(), alg_hints=[:stiff], reltol=1e-8, abstol=1e-8, save_everystep=false)

        V_1_guess[k+1] = soln.u[end][1]
        V_2_guess[k+1] = soln.u[end][2]
        V_3_guess[k+1] = soln.u[end][3]

        T_1_guess[k+1] = soln.u[end][4]
        T_2_guess[k+1] = soln.u[end][5]
        T_3_guess[k+1] = soln.u[end][6]

        x_A1_guess[k+1] = soln.u[end][7]
        x_A2_guess[k+1] = soln.u[end][8]
        x_A3_guess[k+1] = soln.u[end][9]

        x_B1_guess[k+1] = soln.u[end][10]
        x_B2_guess[k+1] = soln.u[end][11]
        x_B3_guess[k+1] = soln.u[end][12]

    end

    x = SharedStates(V_1_guess, V_2_guess, V_3_guess, T_1_guess, T_2_guess, T_3_guess, x_A1_guess, x_A2_guess, x_A3_guess, x_B1_guess, x_B2_guess, x_B3_guess)

    function subsystem1()
        # Initialize MPC model using Ipopt
        MPC = JuMP.Model(Ipopt.Optimizer)

        # Declare variables
        JuMP.@variables MPC begin

            # Flow rates - Feed - Manipulated Inputs
            F_f1[k=0:N], (lower_bound=0.2 * F_f1_sp, upper_bound=1.8 * F_f1_sp, start=F_f1_sp)

            # Flow rate - Recycle Stream - Manipulated Inputs
            F_R[k=0:N], (lower_bound=0.2 * F_R_sp, upper_bound=1.8 * F_R_sp, start=F_R_sp)

            # Heat rates - Manipulated Inputs - Manipulated Variables
            Q_1[k=0:N], (lower_bound=0.2 * Q_1_sp, upper_bound=1.8 * Q_1_sp, start=Q_1_sp)

            # Volumes - State Variables
            V_1[k=0:N], (lower_bound=0.2 * min(V_1_sp, V_1_init), upper_bound=1.8 * max(V_1_sp, V_1_init), start=V_1_sp)

            # Concentrations - State Variables
            x_A1[k=0:N], (lower_bound=0.0001, upper_bound=0.9999, start = x_A1_sp)

            x_B1[k=0:N], (lower_bound=0.0001, upper_bound=0.9999, start = x_B1_sp)

            # Temperatures - State Variables
            T_1[k=0:N], (lower_bound=0.2 * min(T_1_sp, T_1_init), upper_bound=1.8 * max(T_1_sp, T_1_init), start=T_1_sp)


        end

        # Declaire constraints, initial conditions and concentration feasibility limits
        @constraints MPC begin

            # Initial values for state variables
            V_1_initial, V_1[0] == V_1_init
            x_A1_initial, x_A1[0] == x_A1_init
            x_B1_initial, x_B1[0] == x_B1_init
            T_1_initial, T_1[0] == T_1_init

            # Constraint on undesired product, C, such that the concentration is a physically meaningful value
            x_C1_constraint[k=0:N], 0 <= 1 - x_A1[k] - x_B1[k] <= 1

        end

        # Declaire NLconstraints, derived from process dynamics
        @NLconstraints MPC begin

            # Discretized governing differential equations for volume
            dV_1dt[k=0:N-1], V_1[k] + (F_f1[k] + F_R[k] - u.F_1[k+1]) * dt == V_1[k+1]

            # Discretized governing differential equations for temperature
            dT_1dt[k=0:N-1], T_1[k] + ((F_f1[k] / V_1[k]) * (T_f - T_1[k]) + (F_R[k] / V_1[k]) * (x.T_3[k+1] - T_1[k]) + Q_1[k] / (rho * cp * V_1[k]) - (m / cp) * H_1 * k_1 * exp(-E_1 / (R * T_1[k])) * x_A1[k] - (m / cp) * H_2 * k_2 * exp(-E_2 / (R * T_1[k])) * x_B1[k]) * dt == T_1[k+1]

            # Discretized governing differential equations for concentration A
            dx_A1dt[k=0:N-1], x_A1[k] + ((F_f1[k] / V_1[k]) * (x_Af - x_A1[k]) + (F_R[k] / V_1[k]) * ((alpha_A * x.x_A3[k+1]) / (alpha_A * x.x_A3[k+1] + alpha_B * x.x_B3[k+1] + alpha_C * (1 - x.x_A3[k+1] - x.x_B3[k+1])) - x_A1[k]) - k_1 * exp(-E_1 / (R * T_1[k])) * x_A1[k]) * dt == x_A1[k+1]
            dx_B1dt[k=0:N-1], x_B1[k] + ((F_f1[k] / V_1[k]) * (x_Bf - x_B1[k]) + (F_R[k] / V_1[k]) * ((alpha_B * x.x_B3[k+1]) / (alpha_A * x.x_A3[k+1] + alpha_B * x.x_B3[k+1] + alpha_C * (1 - x.x_A3[k+1] - x.x_B3[k+1])) - x_B1[k]) + k_1 * exp(-E_1 / (R * T_1[k])) * x_A1[k] - k_2 * exp(-E_2 / (R * T_1[k])) * x_B1[k]) * dt == x_B1[k+1]

            # Volume hold-up constraints
            holdUp1[k in 0:N], F_f1[k] + F_R[k] - u.F_1[k+1] == -(V_1[k] - V_1_sp) / tau

        end

        @NLobjective(MPC, Min, sum(

            # PI = sum(ISE + ISC)

            # ISE
            (w.con * (x_A1[k] - x_A1_sp)^2 +
             w.con * (x_B1[k] - x_B1_sp)^2 +
             w.temp * (T_1[k] - T_1_sp)^2 +
             w.vol * (V_1[k] - V_1_sp)^2)
            +
            # ISC
            (w.flow * (F_f1[k] - F_f1_sp)^2 +
             w.flow * (F_R[k] - F_R_sp)^2 +
             w.heat * (Q_1[k] - Q_1_sp)^2)

            for k = 0:N))

        # Set attributes for the solver
        MOI.set(MPC, MOI.Silent(), true)

        # Optimize the solution
        JuMP.optimize!(MPC)

        obj_val = JuMP.objective_value(MPC)

        # Obtain optimal inputs and corresponding obtained state predictions
        u_opt = Vector(JuMP.value.(F_f1)), Vector(JuMP.value.(F_R)), Vector(JuMP.value.(Q_1))
        x_pred = Vector(JuMP.value.(V_1)), Vector(JuMP.value.(T_1)), Vector(JuMP.value.(x_A1)), Vector(JuMP.value.(x_B1))
        return u_opt, x_pred, obj_val
    end

    function subsystem2()
        # Initialize MPC model using Ipopt
        MPC = JuMP.Model(Ipopt.Optimizer)

        # Declare variables
        JuMP.@variables MPC begin

            # Flow rates - Feed - Manipulated Inputs
            F_f2[k=0:N], (lower_bound=0.2 * F_f2_sp, upper_bound=1.8 * F_f2_sp, start=F_f2_sp)

            # Flow rate - Recycle Stream - Manipulated Inputs
            F_1[k=0:N], (lower_bound=0.2 * F_1_sp, upper_bound=1.8 * F_1_sp, start=F_1_sp)
            F_2[k=0:N], (lower_bound=0.2 * F_2_sp, upper_bound=1.8 * F_2_sp, start=F_2_sp)

            # Heat rates - Manipulated Inputs - Manipulated Variables
            Q_2[k=0:N], (lower_bound=0.2 * Q_2_sp, upper_bound=1.8 * Q_2_sp, start=Q_2_sp)

            # Volumes - State Variables
            V_2[k=0:N], (lower_bound=0.2 * min(V_2_sp, V_2_init), upper_bound=1.8 * max(V_2_sp, V_2_init), start=V_2_sp)

            # Concentrations - State Variables
            x_A2[k=0:N], (lower_bound=0.0001, upper_bound=0.9999, start = x_A2_sp)

            x_B2[k=0:N], (lower_bound=0.0001, upper_bound=0.9999, start = x_B2_sp)

            # Temperatures - State Variables
            T_2[k=0:N], (lower_bound = 0.2 * min(T_2_sp, T_2_init), upper_bound=1.8 * max(T_2_sp, T_2_init), start=T_2_sp)


        end

        # Declaire constraints, initial conditions and concentration feasibility limits
        @constraints MPC begin

            # Initial values for state variables
            V_2_initial, V_2[0] == V_2_init
            x_A2_initial, x_A2[0] == x_A2_init
            x_B2_initial, x_B2[0] == x_B2_init
            T_2_initial, T_2[0] == T_2_init

            # Constraint on undesired product, C, such that the concentration is a physically meaningful value
            x_C2_constraint[k=0:N], 0 <= 1 - x_A2[k] - x_B2[k] <= 1

        end

        # Declaire NLconstraints, derived from process dynamics
        @NLconstraints MPC begin

            # Discretized governing differential equations for volume
            dV_2dt[k=0:N-1], V_2[k] + (F_f2[k] + F_1[k] - F_2[k]) * dt == V_2[k+1]

            # Discretized governing differential equations for temperature
            dT_2dt[k=0:N-1], T_2[k] + ((F_1[k] / V_2[k]) * (x.T_1[k+1] - T_2[k]) + (F_f2[k] / V_2[k]) * (T_f - T_2[k]) + Q_2[k] / (rho * cp * V_2[k]) - (m / cp) * H_1 * k_1 * exp(-E_1 / (R * T_2[k])) * x_A2[k] - (m / cp) * H_2 * k_2 * exp(-E_2 / (R * T_2[k])) * x_B2[k]) * dt == T_2[k+1]

            # Discretized governing differential equations for concentration A
            dx_A2dt[k=0:N-1], x_A2[k] + ((F_1[k] / V_2[k]) * (x.x_A1[k+1] - x_A2[k]) + (F_f2[k] / V_2[k]) * (x_Af - x_A2[k]) - k_1 * exp(-E_1 / (R * T_2[k])) * x_A2[k]) * dt == x_A2[k+1]
            dx_B2dt[k=0:N-1], x_B2[k] + ((F_1[k] / V_2[k]) * (x.x_B1[k+1] - x_B2[k]) + (F_f2[k] / V_2[k]) * (x_Bf - x_B2[k]) + k_1 * exp(-E_1 / (R * T_2[k])) * x_A2[k] - k_2 * exp(-E_2 / (R * T_2[k])) * x_B2[k]) * dt == x_B2[k+1]

            # Volume hold-up constraints
            holdUp2[k in 0:N], F_f2[k] + F_1[k] - F_2[k] == -(V_2[k] - V_2_sp) / tau


        end

        @NLobjective(MPC, Min, sum(

            # PI = sum(ISE + ISC)

            # ISE
            (w.con * (x_A2[k] - x_A2_sp)^2 +
             w.con * (x_B2[k] - x_B2_sp)^2 +
             w.temp * (T_2[k] - T_2_sp)^2 +
             w.vol * (V_2[k] - V_2_sp)^2)
            +
            # ISC
            (w.flow * (F_f2[k] - F_f2_sp)^2 +
             w.flow * (F_1[k] - F_1_sp)^2 +
             w.flow * (F_2[k] - F_2_sp)^2 +
             w.heat * (Q_2[k] - Q_2_sp)^2)

            for k = 0:N))

        # Set attributes for the solver
        MOI.set(MPC, MOI.Silent(), true)

        # Optimize the solution
        JuMP.optimize!(MPC)

        obj_val = JuMP.objective_value(MPC)

        # Obtain optimal inputs and corresponding obtained state predictions
        u_opt = Vector(JuMP.value.(F_f2)), Vector(JuMP.value.(F_1)), Vector(JuMP.value.(F_2)),  Vector(JuMP.value.(Q_2))
        x_pred = Vector(JuMP.value.(V_2)), Vector(JuMP.value.(T_2)), Vector(JuMP.value.(x_A2)), Vector(JuMP.value.(x_B2))
        return u_opt, x_pred, obj_val
    end

    function subsystem3()
        # Initialize MPC model using Ipopt
        MPC = JuMP.Model(Ipopt.Optimizer)

        # Declare variables
        JuMP.@variables MPC begin

           # Flow rates - Feed - Manipulated Inputs
            # F_2[k=0:N], (lower_bound=0.2 * F_2_sp, upper_bound=1.8 * F_2_sp, start = F_2_sp)
            F_3[k=0:N], (lower_bound=0.2 * F_3_sp, upper_bound=1.8 * F_3_sp, start = F_3_sp)

            # Heat rates - Manipulated Inputs - Manipulated Variables
            Q_3[k=0:N], (lower_bound=0.2 * Q_3_sp, upper_bound=1.8 * Q_3_sp, start = Q_3_sp)

            # Volumes - State Variables
            V_3[k=0:N], (lower_bound=0.2 * min(V_3_sp, V_3_init), upper_bound=1.8 * max(V_3_sp, V_3_init), start=V_3_sp)

            # Concentrations - State Variables
            x_A3[k=0:N], (lower_bound=0.0001, upper_bound=0.9999, start = x_A3_sp)
            x_B3[k=0:N], (lower_bound=0.0001, upper_bound=0.9999, start = x_B3_sp)

            # Temperatures - State Variables
            T_3[k=0:N], (lower_bound = 0.2 * min(T_3_sp, T_3_init), upper_bound=1.8 * max(T_3_sp, T_3_init), start=T_3_sp)


        end

        # Declaire constraints, initial conditions and concentration feasibility limits
        @constraints MPC begin

            # Initial values for state variables
            V_3_initial, V_3[0] == V_3_init
            x_A3_initial, x_A3[0] == x_A3_init
            x_B3_initial, x_B3[0] == x_B3_init
            T_3_initial, T_3[0] == T_3_init

            # Constraint on undesired product, C, such that the concentration is a physically meaningful value
            x_C3_constraint[k=0:N], 0 <= 1 - x_A3[k] - x_B3[k] <= 1

        end

        # Declaire NLconstraints, derived from process dynamics
        @NLconstraints MPC begin

            # Discretized governing differential equations for volume
            dV_3dt[k=0:N-1], V_3[k] + (u.F_2[k+1] - u.F_R[k+1] * epsilon - u.F_R[k+1] - F_3[k]) * dt == V_3[k+1]

            # Discretized governing differential equations for temperature
            dT_3dt[k=0:N-1], T_3[k] + ((u.F_2[k+1] / V_3[k]) * (x.T_2[k+1] - T_3[k]) + Q_3[k] / (rho * cp * V_3[k])) * dt == T_3[k+1]

            # Discretized governing differential equations for concentration A

            dx_A3dt[k=0:N-1], x_A3[k] + ((u.F_2[k+1] / V_3[k]) * (x.x_A2[k+1] - x_A3[k]) - ((u.F_R[k+1] + u.F_R[k+1] * epsilon) / V_3[k]) * ((alpha_A * x_A3[k]) / (alpha_A * x_A3[k] + alpha_B * x_B3[k] + alpha_C * (1 - x_A3[k] - x_B3[k])) - x_A3[k])) * dt == x_A3[k+1]
            dx_B3dt[k=0:N-1], x_B3[k] + ((u.F_2[k+1] / V_3[k]) * (x.x_B2[k+1] - x_B3[k]) - ((u.F_R[k+1] + u.F_R[k+1] * epsilon) / V_3[k]) * ((alpha_B * x_B3[k]) / (alpha_A * x_A3[k] + alpha_B * x_B3[k] + alpha_C * (1 - x_A3[k] - x_B3[k])) - x_B3[k])) * dt == x_B3[k+1]

            # Volume hold-up constraints
            holdUp3[k in 0:N], u.F_2[k+1] - F_3[k] - u.F_R[k+1] * epsilon - u.F_R[k+1] == -(V_3[k] - V_3_sp) / tau

        end

        @NLobjective(MPC, Min, sum(

            # PI = sum(ISE + ISC)

            # ISE
            (w.con * (x_A3[k] - x_A3_sp)^2 +
             w.con * (x_B3[k] - x_B3_sp)^2 +
             w.temp * (T_3[k] - T_3_sp)^2 +
             w.vol * (V_3[k] - V_3_sp)^2)
            +
            # ISC
            (
             w.flow * (F_3[k] - F_3_sp)^2 +
             w.heat * (Q_3[k] - Q_3_sp)^2)

            for k = 0:N))

        # Set attributes for the solver
        MOI.set(MPC, MOI.Silent(), true)

        # Optimize the solution
        JuMP.optimize!(MPC)

        obj_val = JuMP.objective_value(MPC)

        # Obtain optimal inputs and corresponding obtained state predictions
        u_opt = Vector(JuMP.value.(F_3)), Vector(JuMP.value.(Q_3))
        x_pred = Vector(JuMP.value.(V_3)), Vector(JuMP.value.(T_3)), Vector(JuMP.value.(x_A3)), Vector(JuMP.value.(x_B3))
        return u_opt, x_pred, obj_val
    end


    for j = 1:max_coordinations
        # Compute the inputs and obtain shared states for each subsystem controller
        u_opt, x_pred, obj_val = subsystem1()
        F_f1_opt = u_opt[1]
        F_R_opt = u_opt[2]
        Q_1_opt = u_opt[3]
        V_1_pred = x_pred[1]
        T_1_pred = x_pred[2]
        x_A1_pred = x_pred[3]
        x_B1_pred = x_pred[4]

        u_opt, x_pred, obj_val = subsystem2()
        F_f2_opt = u_opt[1]
        F_1_opt = u_opt[2]
        F_2_opt = u_opt[3]
        Q_2_opt = u_opt[4]
        V_2_pred = x_pred[1]
        T_2_pred = x_pred[2]
        x_A2_pred = x_pred[3]
        x_B2_pred = x_pred[4]

        u_opt, x_pred, obj_val = subsystem3()
        F_3_opt = u_opt[1]
        Q_3_opt = u_opt[2]
        V_3_pred = x_pred[1]
        T_3_pred = x_pred[2]
        x_A3_pred = x_pred[3]
        x_B3_pred = x_pred[4]

        # Compile all estimates and inputs, assign to shared structure for network communication
        u_opt = [F_1_opt, F_2_opt, F_3_opt, F_f1_opt, F_f2_opt, F_R_opt, Q_1_opt, Q_2_opt, Q_3_opt]
        x_pred = [V_1_pred, V_2_pred, V_3_pred, T_1_pred, T_2_pred, T_3_pred, x_A1_pred, x_A2_pred, x_A3_pred, x_B1_pred, x_B2_pred, x_B3_pred]

        u.F_1 = u_opt[1]
        u.F_2 = u_opt[2]
        u.F_3 = u_opt[3]
        u.F_f1 = u_opt[4]
        u.F_f2 = u_opt[5]
        u.F_R = u_opt[6]
        u.Q_1 = u_opt[7]
        u.Q_2 = u_opt[8]
        u.Q_3 = u_opt[9]

        x.V_1 = x_pred[1]
        x.V_2 = x_pred[2]
        x.V_3 = x_pred[3]
        x.T_1 = x_pred[4]
        x.T_2 = x_pred[5]
        x.T_3 = x_pred[6]
        x.x_A1 = x_pred[7]
        x.x_A2 = x_pred[8]
        x.x_A3 = x_pred[9]
        x.x_B1 = x_pred[10]
        x.x_B2 = x_pred[11]
        x.x_B3 = x_pred[12]

    end
    u_opt = [u.F_1, u.F_2, u.F_3, u.F_f1, u.F_f2, u.F_R, u.Q_1, u.Q_2, u.Q_3]
    x_pred = [x.V_1, x.V_2, x.V_3, x.T_1, x.T_2, x.T_3, x.x_A1, x.x_A2, x.x_A3, x.x_B1, x.x_B2, x.x_B3]

    return u_opt, x_pred
end