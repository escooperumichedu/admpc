### Initialize pkgs
using JuMP, Ipopt, Plots, DataFrames, LinearAlgebra, Statistics

# A -> B -> C taking place in a CSTR-CSTR-Separator process

### Initialize parameters for the system
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

# Volume holdup time constant
global tau = 0.1

# Horizon and discretizations
global N = 45  # Control horizon
global dt = 0.025 # Sampling time of the system

# Purge ratio
global epsilon = 0.02

# Structure containing the objective weights
mutable struct Weights

    # Weights for output variables
    con::Float64
    temp::Float64
    vol::Float64

    # Weights for input variables
    flow::Float64
    heat::Float64

end

w = Weights(1e3, 1e1, 1e3, 1e2, 1e-3)

function cmpc_mpc(x_init, x_sp, u_sp)

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

    # Initialize MPC model using Ipopt
    MPC = JuMP.Model(Ipopt.Optimizer)

    # Declare variables
    JuMP.@variables MPC begin

        # Flow rates - Feed - Manipulated Inputs
        F_f1[k=0:N], (lower_bound=0.2 * F_f1_sp, upper_bound=1.8 * F_f1_sp, start=F_f1_sp)
        F_f2[k=0:N], (lower_bound=0.2 * F_f2_sp, upper_bound=1.8 * F_f2_sp, start=F_f2_sp)

        # Flow rates - Effluent - Manipulated Inputs
        F_1[k=0:N], (lower_bound=0.2 * F_1_sp, upper_bound=1.8 * F_1_sp, start=F_1_sp)
        F_2[k=0:N], (lower_bound=0.2 * F_2_sp, upper_bound=1.8 * F_2_sp, start=F_2_sp)
        F_3[k=0:N], (lower_bound=0.2 * F_3_sp, upper_bound=1.8 * F_3_sp, start=F_3_sp)

        # Flow rate - Recycle Stream - Manipulated Inputs
        F_R[k=0:N], (lower_bound=0.2 * F_R_sp, upper_bound=1.8 * F_R_sp, start=F_R_sp)

        # Heat rates - Manipulated Inputs - Manipulated Variables
        Q_1[k=0:N], (lower_bound=0.2 * Q_1_sp, upper_bound=1.8 * Q_1_sp, start=Q_1_sp)
        Q_2[k=0:N], (lower_bound=0.2 * Q_2_sp, upper_bound=1.8 * Q_2_sp, start=Q_2_sp)
        Q_3[k=0:N], (lower_bound=0.2 * Q_3_sp, upper_bound=1.8 * Q_3_sp, start=Q_3_sp)

        # Volumes - State Variables
        V_1[k=0:N], (lower_bound=0.2 * min(V_1_sp, V_1_init), upper_bound=1.8 * max(V_1_sp, V_1_init), start=V_1_sp)
        V_2[k=0:N], (lower_bound=0.2 * min(V_2_sp, V_2_init), upper_bound=1.8 * max(V_2_sp, V_2_init), start=V_2_sp)
        V_3[k=0:N], (lower_bound=0.2 * min(V_3_sp, V_3_init), upper_bound=1.8 * max(V_3_sp, V_3_init), start=V_3_sp)

        # Concentrations - State Variables
        x_A1[k=0:N], (lower_bound=0.0001, upper_bound=0.9999, start=x_A1_sp)
        x_A2[k=0:N], (lower_bound=0.0001, upper_bound=0.9999, start=x_A2_sp)
        x_A3[k=0:N], (lower_bound=0.0001, upper_bound=0.9999, start=x_A3_sp)

        x_B1[k=0:N], (lower_bound=0.0001, upper_bound=0.9999, start=x_B1_sp)
        x_B2[k=0:N], (lower_bound=0.0001, upper_bound=0.9999, start=x_B2_sp)
        x_B3[k=0:N], (lower_bound=0.0001, upper_bound=0.9999, start=x_B3_sp)

        # Temperatures - State Variables
        T_1[k=0:N], (lower_bound=0.2 * min(T_1_sp, T_1_init), upper_bound=1.8 * max(T_1_sp, T_1_init), start=T_1_sp)
        T_2[k=0:N], (lower_bound=0.2 * min(T_2_sp, T_2_init), upper_bound=1.8 * max(T_2_sp, T_2_init), start=T_2_sp)
        T_3[k=0:N], (lower_bound=0.2 * min(T_3_sp, T_3_init), upper_bound=1.8 * max(T_3_sp, T_3_init), start=T_3_sp)

    end

    # Declaire constraints, initial conditions and concentration feasibility limits
    @constraints MPC begin

        # Initial values for state variables
        V_1_initial, V_1[0] == V_1_init
        V_2_initial, V_2[0] == V_2_init
        V_3_initial, V_3[0] == V_3_init

        x_A1_initial, x_A1[0] == x_A1_init
        x_A2_initial, x_A2[0] == x_A2_init
        x_A3_initial, x_A3[0] == x_A3_init

        x_B1_initial, x_B1[0] == x_B1_init
        x_B2_initial, x_B2[0] == x_B2_init
        x_B3_initial, x_B3[0] == x_B3_init

        T_1_initial, T_1[0] == T_1_init
        T_2_initial, T_2[0] == T_2_init
        T_3_initial, T_3[0] == T_3_init

        # Constraint on undesired product, C, such that the concentration is a physically meaningful value
        x_C1_constraint[k=0:N], 0 <= 1 - x_A1[k] - x_B1[k] <= 1
        x_C2_constraint[k=0:N], 0 <= 1 - x_A2[k] - x_B2[k] <= 1
        x_C3_constraint[k=0:N], 0 <= 1 - x_A3[k] - x_B3[k] <= 1

    end

    # Declaire NLconstraints, derived from process dynamics
    @NLconstraints MPC begin

        # Discretized governing differential equations for volume
        dV_1dt[k=0:N-1], V_1[k] + (F_f1[k] + F_R[k] - F_1[k]) * dt == V_1[k+1]
        dV_2dt[k=0:N-1], V_2[k] + (F_f2[k] + F_1[k] - F_2[k]) * dt == V_2[k+1]
        dV_3dt[k=0:N-1], V_3[k] + (F_2[k] - F_R[k] * epsilon - F_R[k] - F_3[k]) * dt == V_3[k+1]

        # Discretized governing differential equations for temperature
        dT_1dt[k=0:N-1], T_1[k] + ((F_f1[k] / V_1[k]) * (T_f - T_1[k]) + (F_R[k] / V_1[k]) * (T_3[k] - T_1[k]) + Q_1[k] / (rho * cp * V_1[k]) - (m / cp) * H_1 * k_1 * exp(-E_1 / (R * T_1[k])) * x_A1[k] - (m / cp) * H_2 * k_2 * exp(-E_2 / (R * T_1[k])) * x_B1[k]) * dt == T_1[k+1]
        dT_2dt[k=0:N-1], T_2[k] + ((F_1[k] / V_2[k]) * (T_1[k] - T_2[k]) + (F_f2[k] / V_2[k]) * (T_f - T_2[k]) + Q_2[k] / (rho * cp * V_2[k]) - (m / cp) * H_1 * k_1 * exp(-E_1 / (R * T_2[k])) * x_A2[k] - (m / cp) * H_2 * k_2 * exp(-E_2 / (R * T_2[k])) * x_B2[k]) * dt == T_2[k+1]
        dT_3dt[k=0:N-1], T_3[k] + ((F_2[k] / V_3[k]) * (T_2[k] - T_3[k]) + Q_3[k] / (rho * cp * V_3[k])) * dt == T_3[k+1]

        # Discretized governing differential equations for concentration A
        dx_A1dt[k=0:N-1], x_A1[k] + ((F_f1[k] / V_1[k]) * (x_Af - x_A1[k]) + (F_R[k] / V_1[k]) * ((alpha_A * x_A3[k]) / (alpha_A * x_A3[k] + alpha_B * x_B3[k] + alpha_C * (1 - x_A3[k] - x_B3[k])) - x_A1[k]) - k_1 * exp(-E_1 / (R * T_1[k])) * x_A1[k]) * dt == x_A1[k+1]
        dx_B1dt[k=0:N-1], x_B1[k] + ((F_f1[k] / V_1[k]) * (x_Bf - x_B1[k]) + (F_R[k] / V_1[k]) * ((alpha_B * x_B3[k]) / (alpha_A * x_A3[k] + alpha_B * x_B3[k] + alpha_C * (1 - x_A3[k] - x_B3[k])) - x_B1[k]) + k_1 * exp(-E_1 / (R * T_1[k])) * x_A1[k] - k_2 * exp(-E_2 / (R * T_1[k])) * x_B1[k]) * dt == x_B1[k+1]

        dx_A2dt[k=0:N-1], x_A2[k] + ((F_1[k] / V_2[k]) * (x_A1[k] - x_A2[k]) + (F_f2[k] / V_2[k]) * (x_Af - x_A2[k]) - k_1 * exp(-E_1 / (R * T_2[k])) * x_A2[k]) * dt == x_A2[k+1]
        dx_B2dt[k=0:N-1], x_B2[k] + ((F_1[k] / V_2[k]) * (x_B1[k] - x_B2[k]) + (F_f2[k] / V_2[k]) * (x_Bf - x_B2[k]) + k_1 * exp(-E_1 / (R * T_2[k])) * x_A2[k] - k_2 * exp(-E_2 / (R * T_2[k])) * x_B2[k]) * dt == x_B2[k+1]

        dx_A3dt[k=0:N-1], x_A3[k] + ((F_2[k] / V_3[k]) * (x_A2[k] - x_A3[k]) - ((F_R[k] + F_R[k] * epsilon) / V_3[k]) * ((alpha_A * x_A3[k]) / (alpha_A * x_A3[k] + alpha_B * x_B3[k] + alpha_C * (1 - x_A3[k] - x_B3[k])) - x_A3[k])) * dt == x_A3[k+1]
        dx_B3dt[k=0:N-1], x_B3[k] + ((F_2[k] / V_3[k]) * (x_B2[k] - x_B3[k]) - ((F_R[k] + F_R[k] * epsilon) / V_3[k]) * ((alpha_B * x_B3[k]) / (alpha_A * x_A3[k] + alpha_B * x_B3[k] + alpha_C * (1 - x_A3[k] - x_B3[k])) - x_B3[k])) * dt == x_B3[k+1]

        # # Volume hold-up constraints
        holdUp1[k in 0:N], F_f1[k] + F_R[k] - F_1[k] == -(V_1[k] - V_1_sp) / tau
        holdUp2[k in 0:N], F_f2[k] + F_1[k] - F_2[k] == -(V_2[k] - V_2_sp) / tau
        holdUp3[k in 0:N], F_2[k] - F_3[k] - F_R[k] * epsilon - F_R[k] == -(V_3[k] - V_3_sp) / tau

    end

    @NLobjective(MPC, Min, sum(

        # PI = sum(ISE + ISC)

        # ISE
        (w.con * (x_A1[k] - x_A1_sp)^2 + w.con * (x_A2[k] - x_A2_sp)^2 + w.con * (x_A3[k] - x_A3_sp)^2 +
         w.con * (x_B1[k] - x_B1_sp)^2 + w.con * (x_B2[k] - x_B2_sp)^2 + w.con * (x_B3[k] - x_B3_sp)^2 +
         w.temp * (T_1[k] - T_1_sp)^2 + w.temp * (T_2[k] - T_2_sp)^2 + w.temp * (T_3[k] - T_3_sp)^2 +
         w.vol * (V_1[k] - V_1_sp)^2 + w.vol * (V_2[k] - V_2_sp)^2 + w.vol * (V_3[k] - V_3_sp)^2) +

        # ISC
        (w.flow * (F_f1[k] - F_f1_sp)^2 + w.flow * (F_f2[k] - F_f2_sp)^2 + w.flow * (F_1[k] - F_1_sp)^2 +
         w.flow * (F_2[k] - F_2_sp)^2 + w.flow * (F_3[k] - F_3_sp)^2 + w.flow * (F_R[k] - F_R_sp)^2 +
         w.heat * (Q_1[k] - Q_1_sp)^2 + w.heat * (Q_2[k] - Q_2_sp)^2 + w.heat * (Q_3[k] - Q_3_sp)^2)

        for k = 0:N))

    # Set attributes for the solver
    # MOI.set(MPC, MOI.Silent(), true)

    # Optimize the solution
    JuMP.optimize!(MPC)

    obj_val = JuMP.objective_value(MPC)

    # Obtain optimal inputs
    u_opt = Vector(JuMP.value.(F_1)), Vector(JuMP.value.(F_2)), Vector(JuMP.value.(F_3)), Vector(JuMP.value.(F_f1)), Vector(JuMP.value.(F_f2)), Vector(JuMP.value.(F_R)), Vector(JuMP.value.(Q_1)), Vector(JuMP.value.(Q_2)), Vector(JuMP.value.(Q_3))
    return u_opt, obj_val

end

