using Distributed
using iLQGames
import iLQGames: dx
using Plots
using ForwardDiff
using iLQGames:
    SystemTrajectory
using iLQGames:
    LinearSystem
using Infiltrator
using Optim
using LinearAlgebra
using Distributed
using Dates
using Statistics
using JuMP
using Ipopt
include("../src/diff_solver.jl")
include("../src/inverse_game_solver.jl")
include("../src/experiment_utils.jl") # NOTICE!! Many functions are defined there.

nx, nu, ΔT, game_horizon = 8, 4, 0.1, 40

# setup the dynamics
struct DoubleUnicycle <: ControlSystem{ΔT,nx,nu} end
# state: (px, py, phi, v)
dx(cs::DoubleUnicycle, x, u, t) = SVector(x[4]cos(x[3]), x[4]sin(x[3]), u[1], u[2], 
                                    x[8]cos(x[7]), x[8]sin(x[7]), u[3], u[4])
dynamics = DoubleUnicycle()

# costs = (FunctionPlayerCost((g, x, u, t) -> (10*(x[1]-1)^2 + 0.1*(x[3]-pi/2)^2 + (x[4]-1)^2 + u[1]^2 + u[2]^2 - 0.1*((x[1]-x[5])^2 + (x[2]-x[6])^2))),
         # FunctionPlayerCost((g, x, u, t) -> ((x[5]-1)^2 + 0.1*(x[7]-pi/2)^2 + (x[8]-1)^2 + u[3]^2 + u[4]^2- 0.1*((x[1]-x[5])^2 + (x[2]-x[6])^2))))
costs = (FunctionPlayerCost((g, x, u, t) -> ( 4*(x[5]-1)^2 + 2*(x[4]-1)^2 + u[1]^2 + u[2]^2 - 0*((x[1]-x[5])^2 + (x[2]-x[6])^2))),
         FunctionPlayerCost((g, x, u, t) -> (  4*(x[5] - x[1])^2 + 2*(x[8]-1)^2 + u[3]^2 + u[4]^2 - 0*((x[1]-x[5])^2 + (x[2]-x[6])^2))))

# indices of inputs that each player controls
player_inputs = (SVector(1,2), SVector(3,4))
# the horizon of the game
g = GeneralGame(game_horizon, player_inputs, dynamics, costs)

# get a solver, choose initial conditions and solve (in about 9 ms with AD)
solver1 = iLQSolver(g, max_scale_backtrack=10, max_elwise_diff_step=Inf, equilibrium_type="OLNE_KKT")
x0 = SVector(0, 0.5, pi/2, 1,       1, 0, pi/2, 1)
c1, expert_traj1, strategies1 = solve(g, solver1, x0)

solver2 = iLQSolver(g, max_scale_backtrack=5, max_elwise_diff_step=Inf, equilibrium_type="FBNE_KKT")
c2, expert_traj2, strategies2 = solve(g, solver2, x0)

function parameterized_cost(θ::Vector)
    costs = (FunctionPlayerCost((g, x, u, t) -> ( θ[1]*(x[5]-1)^2+θ[4]*(x[1]-1)^2  + (2*(x[4]-1)^2 + u[1]^2 + u[2]^2) - 0*((x[1]-x[5])^2 + (x[2]-x[6])^2))),
             FunctionPlayerCost((g, x, u, t) -> ( θ[2]*(x[5]-0)^2+θ[3]*(x[5] - x[1])^2 + (2*(x[8]-1)^2 + u[3]^2 + u[4]^2) - 0*((x[1]-x[5])^2 + (x[2]-x[6])^2))))
    return costs
end

# θ_true = [10, 1, 1, 4, 1]
θ_true = [4, 0, 4]

obs_x_FB = transpose(mapreduce(permutedims, vcat, Vector([Vector(expert_traj2.x[t]) for t in 1:g.h])))
obs_u_FB = transpose(mapreduce(permutedims, vcat, Vector([Vector(expert_traj2.u[t]) for t in 1:g.h])))

obs_x = transpose(mapreduce(permutedims, vcat, Vector([Vector(expert_traj1.x[t]) for t in 1:g.h])))
obs_u = transpose(mapreduce(permutedims, vcat, Vector([Vector(expert_traj1.u[t]) for t in 1:g.h])))



# ------------------------------------ Optimization problem begin ------------------------------------------- #

function KKT_highway_forward_game_solve()
    θ=[4,0,4,0]
    model = Model(Ipopt.Optimizer)
    @variable(model, x[1:nx, 1:g.h])
    @variable(model, u[1:nu, 1:g.h])
    @variable(model, λ[1:2, 1:nx, 1:g.h])
    @objective(model, Min, 0)
    ΔT = 0.1
    for t in 1:g.h # for each time t within the game horizon
        if t != g.h # dJ1/dx
            @constraint(model,   λ[1,1,t] + 2*θ[4]*(x[1,t]-1)             - λ[1,1,t+1] == 0)
            @constraint(model,   λ[1,2,t]               - λ[1,2,t+1] == 0)
            @NLconstraint(model, λ[1,3,t]               - λ[1,3,t+1] + λ[1,1,t+1]*ΔT*x[4,t]*sin(x[3,t]) - λ[1,2,t+1]*ΔT*x[4,t]*cos(x[3,t])  == 0)
            @NLconstraint(model, λ[1,4,t] + 4*(x[4,t]-1)                - λ[1,4,t+1] - λ[1,1,t+1]*ΔT*cos(x[3,t]) - λ[1,2,t+1]*ΔT*sin(x[3,t]) == 0)
            @constraint(model,   λ[1,5,t] + 2*θ[1]*(x[5,t]-1)           - λ[1,5,t+1] == 0)
            @constraint(model,   λ[1,6,t]               - λ[1,6,t+1] == 0)
            @NLconstraint(model, λ[1,7,t]               - λ[1,7,t+1] + λ[1,5,t+1]*ΔT*x[8,t]*sin(x[7,t]) - λ[1,6,t+1]*ΔT*x[8,t]*cos(x[7,t]) == 0)
            @NLconstraint(model, λ[1,8,t]               - λ[1,8,t+1] - λ[1,5,t+1]*ΔT*cos(x[7,t]) - λ[1,6,t+1]*ΔT*sin(x[7,t]) == 0)
        else
            @constraint(model,   λ[1,1,t] + 2*θ[4]*(x[1,t]-1) == 0)
            @constraint(model,   λ[1,2,t] == 0)
            @NLconstraint(model, λ[1,3,t] == 0)
            @NLconstraint(model, λ[1,4,t] + 4*(x[4,t]-1)  == 0)
            @constraint(model,   λ[1,5,t] + 2*θ[1]*(x[5,t]-1) == 0)
            @constraint(model,   λ[1,6,t] == 0)
            @NLconstraint(model, λ[1,7,t] == 0)
            @NLconstraint(model, λ[1,8,t] == 0)
        end

        if t != g.h # dJ2/dx
            @constraint(model,   λ[2,1,t] + 2*θ[3]*(x[1,t]-x[5,t])                   - λ[2,1,t+1] == 0)
            @constraint(model,   λ[2,2,t]                   - λ[2,2,t+1] == 0)
            @NLconstraint(model, λ[2,3,t]                   - λ[2,3,t+1] + λ[2,1,t+1]*ΔT*x[4,t]*sin(x[3,t]) - λ[2,2,t+1]*ΔT*x[4,t]*cos(x[3,t])  == 0)
            @NLconstraint(model, λ[2,4,t]                   - λ[2,4,t+1] - λ[2,1,t+1]*ΔT*cos(x[3,t]) - λ[2,2,t+1]*ΔT*sin(x[3,t]) == 0)
            @constraint(model,   λ[2,5,t] + 2*θ[2]*x[5,t] + 2*θ[3]*(x[5,t]-x[1,t])                    - λ[2,5,t+1] == 0)
            @constraint(model,   λ[2,6,t]                   - λ[2,6,t+1] == 0)
            @NLconstraint(model, λ[2,7,t]                   - λ[2,7,t+1] + λ[2,5,t+1]*ΔT*x[8,t]*sin(x[7,t]) - λ[2,6,t+1]*ΔT*x[8,t]*cos(x[7,t]) == 0)
            @NLconstraint(model, λ[2,8,t] + 4*(x[8,t]-1)                  - λ[2,8,t+1] - λ[2,5,t+1]*ΔT*cos(x[7,t]) - λ[2,6,t+1]*ΔT*sin(x[7,t]) == 0)
        else
            @constraint(model,   λ[2,1,t] + 2*θ[3]*(x[1,t]-x[5,t]) == 0)
            @constraint(model,   λ[2,2,t]  == 0)
            @NLconstraint(model, λ[2,3,t]  == 0)
            @NLconstraint(model, λ[2,4,t]  == 0)
            @constraint(model,   λ[2,5,t] + 2*θ[2]*x[5,t] + 2*θ[3]*(x[5,t]-x[1,t]) == 0)
            @constraint(model,   λ[2,6,t]  == 0)
            @NLconstraint(model, λ[2,7,t]  == 0)
            @NLconstraint(model, λ[2,8,t] + 4*(x[8,t]-1) == 0)
        end

        # dJ1/du and dJ2/du
        @constraint(model, 2*u[1,t] - λ[1,3,t]*ΔT == 0)
        @constraint(model, 2*u[2,t] - λ[1,4,t]*ΔT == 0)
        @constraint(model, 2*u[3,t] - λ[2,7,t]*ΔT == 0)
        @constraint(model, 2*u[4,t] - λ[2,8,t]*ΔT == 0)
        if t == 1
            @NLconstraint(model, x[1,1] == x0[1] + ΔT * x0[4]*cos(x0[3]))
            @NLconstraint(model, x[2,1] == x0[2] + ΔT * x0[4]*sin(x0[3]))
            @NLconstraint(model, x[3,1] == x0[3] + ΔT * u[1,t])
            @NLconstraint(model, x[4,1] == x0[4] + ΔT * u[2,t])
            @NLconstraint(model, x[5,1] == x0[5] + ΔT * x0[8]*cos(x0[7]))
            @NLconstraint(model, x[6,1] == x0[6] + ΔT * x0[8]*sin(x0[7]))
            @NLconstraint(model, x[7,1] == x0[7] + ΔT * u[3,t])
            @NLconstraint(model, x[8,1] == x0[8] + ΔT * u[4,t])
        else
            @NLconstraint(model, x[1,t] == x[1,t-1] + ΔT * x[4,t-1]*cos(x[3,t-1]))
            @NLconstraint(model, x[2,t] == x[2,t-1] + ΔT * x[4,t-1]*sin(x[3,t-1]))
            @NLconstraint(model, x[3,t] == x[3,t-1] + ΔT * u[1,t])
            @NLconstraint(model, x[4,t] == x[4,t-1] + ΔT * u[2,t])
            @NLconstraint(model, x[5,t] == x[5,t-1] + ΔT * x[8,t-1]*cos(x[7,t-1]))
            @NLconstraint(model, x[6,t] == x[6,t-1] + ΔT * x[8,t-1]*sin(x[7,t-1]))
            @NLconstraint(model, x[7,t] == x[7,t-1] + ΔT * u[3,t])
            @NLconstraint(model, x[8,t] == x[8,t-1] + ΔT * u[4,t])
        end
    end
    # @constraint(model, θ .>= -0.1*ones(3))
    optimize!(model)
    return value.(x), value.(u), value.(θ)
end

JuMP_forward_sol=KKT_highway_forward_game_solve()

function parameterized_cost(θ::Vector)
    costs = (FunctionPlayerCost((g, x, u, t) -> ( θ[1]*(x[5]-1)^2  + (2*(x[4]-1)^2 + u[1]^2 + u[2]^2) - 0*((x[1]-x[5])^2 + (x[2]-x[6])^2))),
             FunctionPlayerCost((g, x, u, t) -> ( θ[2]*(x[5]-0)^2+θ[3]*(x[5] - x[1])^2 + (2*(x[8]-1)^2 + u[3]^2 + u[4]^2) - 0*((x[1]-x[5])^2 + (x[2]-x[6])^2))))
    return costs
end
# ------------------------------------ Optimization problem end ------------------------------------------- #
function KKT_highway_inverse_game_solve(obs_x, obs_u, init_θ)
    # θ=[4,0,4]
    model = Model(Ipopt.Optimizer)
    @variable(model, x[1:nx, 1:g.h])
    @variable(model, u[1:nu, 1:g.h])
    @variable(model, λ[1:2, 1:nx, 1:g.h])
    @variable(model, θ[1:4])
    set_start_value(θ[1], init_θ[1])
    set_start_value(θ[2], init_θ[2])
    set_start_value(θ[3], init_θ[3])
    set_start_value(θ[4], init_θ[4])
    # @objective(model, Min, 0)
    @objective(model, Min, sum(sum((x[ii,t] - obs_x[ii,t])^2 for ii in 1:nx ) for t in 1:game_horizon-1) + sum(sum((u[ii,t] - obs_u[ii,t])^2 for ii in 1:nu) for t in 1:game_horizon) )
    ΔT = 0.1

    for t in 1:g.h # for each time t within the game horizon
        if t != g.h # dJ1/dx
            @constraint(model,   λ[1,1,t] + 2*θ[4]*(x[1,t]-1)              - λ[1,1,t+1] == 0)
            @constraint(model,   λ[1,2,t]               - λ[1,2,t+1] == 0)
            @NLconstraint(model, λ[1,3,t]               - λ[1,3,t+1] + λ[1,1,t+1]*ΔT*x[4,t]*sin(x[3,t]) - λ[1,2,t+1]*ΔT*x[4,t]*cos(x[3,t])  == 0)
            @NLconstraint(model, λ[1,4,t] + 4*(x[4,t]-1)                - λ[1,4,t+1] - λ[1,1,t+1]*ΔT*cos(x[3,t]) - λ[1,2,t+1]*ΔT*sin(x[3,t]) == 0)
            @constraint(model,   λ[1,5,t] + 2*θ[1]*(x[5,t]-1)           - λ[1,5,t+1] == 0)
            @constraint(model,   λ[1,6,t]               - λ[1,6,t+1] == 0)
            @NLconstraint(model, λ[1,7,t]               - λ[1,7,t+1] + λ[1,5,t+1]*ΔT*x[8,t]*sin(x[7,t]) - λ[1,6,t+1]*ΔT*x[8,t]*cos(x[7,t]) == 0)
            @NLconstraint(model, λ[1,8,t]               - λ[1,8,t+1] - λ[1,5,t+1]*ΔT*cos(x[7,t]) - λ[1,6,t+1]*ΔT*sin(x[7,t]) == 0)
        else
            @constraint(model,   λ[1,1,t] + 2*θ[4]*(x[1,t]-1) == 0)
            @constraint(model,   λ[1,2,t] == 0)
            @NLconstraint(model, λ[1,3,t] == 0)
            @NLconstraint(model, λ[1,4,t] + 4*(x[4,t]-1)  == 0)
            @constraint(model,   λ[1,5,t] + 2*θ[1]*(x[5,t]-1) == 0)
            @constraint(model,   λ[1,6,t] == 0)
            @NLconstraint(model, λ[1,7,t] == 0)
            @NLconstraint(model, λ[1,8,t] == 0)
        end

        if t != g.h # dJ2/dx
            @constraint(model,   λ[2,1,t] + 2*θ[3]*(x[1,t]-x[5,t])                   - λ[2,1,t+1] == 0)
            @constraint(model,   λ[2,2,t]                   - λ[2,2,t+1] == 0)
            @NLconstraint(model, λ[2,3,t]                   - λ[2,3,t+1] + λ[2,1,t+1]*ΔT*x[4,t]*sin(x[3,t]) - λ[2,2,t+1]*ΔT*x[4,t]*cos(x[3,t])  == 0)
            @NLconstraint(model, λ[2,4,t]                   - λ[2,4,t+1] - λ[2,1,t+1]*ΔT*cos(x[3,t]) - λ[2,2,t+1]*ΔT*sin(x[3,t]) == 0)
            @constraint(model,   λ[2,5,t] + 2*θ[2]*x[5,t] + 2*θ[3]*(x[5,t]-x[1,t])                    - λ[2,5,t+1] == 0)
            @constraint(model,   λ[2,6,t]                   - λ[2,6,t+1] == 0)
            @NLconstraint(model, λ[2,7,t]                   - λ[2,7,t+1] + λ[2,5,t+1]*ΔT*x[8,t]*sin(x[7,t]) - λ[2,6,t+1]*ΔT*x[8,t]*cos(x[7,t]) == 0)
            @NLconstraint(model, λ[2,8,t] + 4*(x[8,t]-1)                  - λ[2,8,t+1] - λ[2,5,t+1]*ΔT*cos(x[7,t]) - λ[2,6,t+1]*ΔT*sin(x[7,t]) == 0)
        else
            @constraint(model,   λ[2,1,t] + 2*θ[3]*(x[1,t]-x[5,t]) == 0)
            @constraint(model,   λ[2,2,t]  == 0)
            @NLconstraint(model, λ[2,3,t]  == 0)
            @NLconstraint(model, λ[2,4,t]  == 0)
            @constraint(model,   λ[2,5,t] + 2*θ[2]*x[5,t] + 2*θ[3]*(x[5,t]-x[1,t]) == 0)
            @constraint(model,   λ[2,6,t]  == 0)
            @NLconstraint(model, λ[2,7,t]  == 0)
            @NLconstraint(model, λ[2,8,t] + 4*(x[8,t]-1) == 0)
        end

        # dJ1/du and dJ2/du
        @constraint(model, 2*u[1,t] - λ[1,3,t]*ΔT == 0)
        @constraint(model, 2*u[2,t] - λ[1,4,t]*ΔT == 0)
        @constraint(model, 2*u[3,t] - λ[2,7,t]*ΔT == 0)
        @constraint(model, 2*u[4,t] - λ[2,8,t]*ΔT == 0)
        if t == 1
            @NLconstraint(model, x[1,1] == x0[1] + ΔT * x0[4]*cos(x0[3]))
            @NLconstraint(model, x[2,1] == x0[2] + ΔT * x0[4]*sin(x0[3]))
            @NLconstraint(model, x[3,1] == x0[3] + ΔT * u[1,t])
            @NLconstraint(model, x[4,1] == x0[4] + ΔT * u[2,t])
            @NLconstraint(model, x[5,1] == x0[5] + ΔT * x0[8]*cos(x0[7]))
            @NLconstraint(model, x[6,1] == x0[6] + ΔT * x0[8]*sin(x0[7]))
            @NLconstraint(model, x[7,1] == x0[7] + ΔT * u[3,t])
            @NLconstraint(model, x[8,1] == x0[8] + ΔT * u[4,t])
        else
            @NLconstraint(model, x[1,t] == x[1,t-1] + ΔT * x[4,t-1]*cos(x[3,t-1]))
            @NLconstraint(model, x[2,t] == x[2,t-1] + ΔT * x[4,t-1]*sin(x[3,t-1]))
            @NLconstraint(model, x[3,t] == x[3,t-1] + ΔT * u[1,t])
            @NLconstraint(model, x[4,t] == x[4,t-1] + ΔT * u[2,t])
            @NLconstraint(model, x[5,t] == x[5,t-1] + ΔT * x[8,t-1]*cos(x[7,t-1]))
            @NLconstraint(model, x[6,t] == x[6,t-1] + ΔT * x[8,t-1]*sin(x[7,t-1]))
            @NLconstraint(model, x[7,t] == x[7,t-1] + ΔT * u[3,t])
            @NLconstraint(model, x[8,t] == x[8,t-1] + ΔT * u[4,t])
        end
    end
    @constraint(model, θ .>= -0.0*ones(4))
    optimize!(model)
    return value.(x), value.(u), value.(θ)
end

tmp = KKT_highway_inverse_game_solve(obs_x_FB[:,2:end], obs_u_FB, [1,1,1,1])


anim1 = @animate for i in 1:game_horizon
    plot( [tmp[1][1,i], tmp[1][1,i]], [tmp[1][2,i], tmp[1][2,i]], markershape = :square, label = "player 1, JuMP", xlims = (-0.5, 1.5), ylims = (0, 6))
    plot!([tmp[1][5,i], tmp[1][5,i]], [tmp[1][6,i], tmp[1][6,i]], markershape = :square, label = "player 2, JuMP", xlims = (-0.5, 1.5), ylims = (0, 6))
    plot!([0], seriestype = "vline", color = "black", label = "")
    plot!([1], seriestype = "vline", color = "black", label = "") 
end

gif(anim1, "lane_guiding_JuMP.gif", fps = 10)



anim2 = @animate for i in 1:game_horizon
    plot( [obs_x[1,i], obs_x[1,i]], [obs_x[2,i], obs_x[2,i]], markershape = :square, label = "player 1, iLQ OLNE", xlims = (-0.5, 1.5), ylims = (0, 6))
    plot!([obs_x[5,i], obs_x[5,i]], [obs_x[6,i], obs_x[6,i]], markershape = :square, label = "player 2, iLQ OLNE", xlims = (-0.5, 1.5), ylims = (0, 6))    
    plot!([0], seriestype = "vline", color = "black", label = "")
    plot!([1], seriestype = "vline", color = "black", label = "")
end

gif(anim2, "lane_guiding_OL_iLQ.gif", fps = 10)

