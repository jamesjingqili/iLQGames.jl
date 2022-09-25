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

nx, nu, ΔT, game_horizon = 9, 4, 0.1, 40

# setup the dynamics
struct DoubleUnicycle <: ControlSystem{ΔT,nx,nu} end
# state: (px, py, phi, v)
dx(cs::DoubleUnicycle, x, u, t) = SVector(x[4]cos(x[3]), x[4]sin(x[3]), u[1], u[2], 
                                    x[8]cos(x[7]), x[8]sin(x[7]), u[3], u[4], 0)
dynamics = DoubleUnicycle()

# costs = (FunctionPlayerCost((g, x, u, t) -> (10*(x[1]-1)^2 + 0.1*(x[3]-pi/2)^2 + (x[4]-1)^2 + u[1]^2 + u[2]^2 - 0.1*((x[1]-x[5])^2 + (x[2]-x[6])^2))),
         # FunctionPlayerCost((g, x, u, t) -> ((x[5]-1)^2 + 0.1*(x[7]-pi/2)^2 + (x[8]-1)^2 + u[3]^2 + u[4]^2- 0.1*((x[1]-x[5])^2 + (x[2]-x[6])^2))))
costs = (FunctionPlayerCost((g, x, u, t) -> ( 8*(x[5]-x[9])^2 + 0*(x[1])^2 + 2*(u[1]^2 + u[2]^2) - 0*((x[1]-x[5])^2 + (x[2]-x[6])^2))),
         FunctionPlayerCost((g, x, u, t) -> ( 4*(x[5] - x[1])^2 + 4*(x[8]-1)^2 + 2*(u[3]^2 + u[4]^2) - 0*((x[1]-x[5])^2 + (x[2]-x[6])^2))))

# indices of inputs that each player controls
player_inputs = (SVector(1,2), SVector(3,4))
# the horizon of the game
g = GeneralGame(game_horizon, player_inputs, dynamics, costs)

# get a solver, choose initial conditions and solve (in about 9 ms with AD)
solver1 = iLQSolver(g, max_scale_backtrack=10, max_elwise_diff_step=Inf, equilibrium_type="OLNE_costate")
x0 = SVector(0, 0.5, pi/2, 1,       1, 0, pi/2, 1, 0.1)
c1, expert_traj1, strategies1 = solve(g, solver1, x0)

solver2 = iLQSolver(g, max_scale_backtrack=5, max_elwise_diff_step=Inf, equilibrium_type="FBNE_costate")
c2, expert_traj2, strategies2 = solve(g, solver2, x0)


x1_OL, y1_OL = [expert_traj1.x[i][1] for i in 1:game_horizon], [expert_traj1.x[i][2] for i in 1:game_horizon];
x2_OL, y2_OL = [expert_traj1.x[i][5] for i in 1:game_horizon], [expert_traj1.x[i][6] for i in 1:game_horizon];
anim1 = @animate for i in 1:game_horizon
    plot([x1_OL[i], x1_OL[i]], [y1_OL[i], y1_OL[i]], markershape = :square, label = "player 1, OL", xlims = (-0.5, 1.5), ylims = (0, 6))
    plot!([x2_OL[i], x2_OL[i]], [y2_OL[i], y2_OL[i]], markershape = :square, label = "player 2, OL", xlims = (-0.5, 1.5), ylims = (0, 6))
    plot!([0], seriestype = "vline", color = "black", label = "")
    plot!([1], seriestype = "vline", color = "black", label = "") 
end
gif(anim1, "lane_guiding_OL_moving.gif", fps = 10)
x1_FB, y1_FB = [expert_traj2.x[i][1] for i in 1:game_horizon], [expert_traj2.x[i][2] for i in 1:game_horizon];
x2_FB, y2_FB = [expert_traj2.x[i][5] for i in 1:game_horizon], [expert_traj2.x[i][6] for i in 1:game_horizon];
anim2 = @animate for i in 1:game_horizon
    plot([x1_FB[i], x1_FB[i]], [y1_FB[i], y1_FB[i]], markershape = :square, label = "player 1, FB", xlims = (-0.5, 1.5), ylims = (0, 6))
    plot!([x2_FB[i], x2_FB[i]], [y2_FB[i], y2_FB[i]], markershape = :square, label = "player 2, FB", xlims = (-0.5, 1.5), ylims = (0, 6))    
    plot!([0], seriestype = "vline", color = "black", label = "")
    plot!([1], seriestype = "vline", color = "black", label = "")
end
gif(anim2, "lane_guiding_FB_moving.gif", fps = 10)



function parameterized_cost(θ::Vector)
    costs = (FunctionPlayerCost((g, x, u, t) -> ( θ[1]*(x[5]-x[9])^2 + θ[2]*x[1]^2  + 4*(u[1]^2 + u[2]^2) - 0*((x[1]-x[5])^2 + (x[2]-x[6])^2))),
             FunctionPlayerCost((g, x, u, t) -> ( θ[3]*(x[5] - x[1])^2 + θ[4]*(x[8]-1)^2 + 4*(u[3]^2 + u[2]^2) - 0*((x[1]-x[5])^2 + (x[2]-x[6])^2))))
    return costs
end

# θ_true = [10, 1, 1, 4, 1]
# We design the experiment such that the expert trajectory navigates to zero, but we want to generalize to elsewhere.
θ_true = [6, 0, 4, 2]
obs_x_FB = transpose(mapreduce(permutedims, vcat, Vector([Vector(expert_traj2.x[t]) for t in 1:g.h])))
obs_u_FB = transpose(mapreduce(permutedims, vcat, Vector([Vector(expert_traj2.u[t]) for t in 1:g.h])))
obs_x_OL = transpose(mapreduce(permutedims, vcat, Vector([Vector(expert_traj1.x[t]) for t in 1:g.h])))
obs_u_OL = transpose(mapreduce(permutedims, vcat, Vector([Vector(expert_traj1.u[t]) for t in 1:g.h])))

# ------------------------------------ Optimization problem begin ------------------------------------------- #

function KKT_highway_forward_game_solve( x0,g)
    θ=[8,0,4,4]
    model = Model(Ipopt.Optimizer)
    @variable(model, x[1:nx, 1:g.h])
    @variable(model, u[1:nu, 1:g.h])
    @variable(model, λ[1:2, 1:nx, 1:g.h])
    @objective(model, Min, 0)
    ΔT = 0.1
    for t in 1:g.h # for each time t within the game horizon
        if t != g.h # dJ1/dx
            @constraint(model,   λ[1,1,t] + 2*θ[2]*x[1,t]             - λ[1,1,t+1] == 0)
            @constraint(model,   λ[1,2,t]               - λ[1,2,t+1] == 0)
            @NLconstraint(model, λ[1,3,t]               - λ[1,3,t+1] + λ[1,1,t+1]*ΔT*x[4,t]*sin(x[3,t]) - λ[1,2,t+1]*ΔT*x[4,t]*cos(x[3,t])  == 0)
            @NLconstraint(model, λ[1,4,t]               - λ[1,4,t+1] - λ[1,1,t+1]*ΔT*cos(x[3,t]) - λ[1,2,t+1]*ΔT*sin(x[3,t]) == 0)
            @constraint(model,   λ[1,5,t] + 2*θ[1]*(x[5,t]-x[9,t])           - λ[1,5,t+1] == 0)
            @constraint(model,   λ[1,6,t]               - λ[1,6,t+1] == 0)
            @NLconstraint(model, λ[1,7,t]               - λ[1,7,t+1] + λ[1,5,t+1]*ΔT*x[8,t]*sin(x[7,t]) - λ[1,6,t+1]*ΔT*x[8,t]*cos(x[7,t]) == 0)
            @NLconstraint(model, λ[1,8,t]               - λ[1,8,t+1] - λ[1,5,t+1]*ΔT*cos(x[7,t]) - λ[1,6,t+1]*ΔT*sin(x[7,t]) == 0)
            @NLconstraint(model, λ[1,9,t] + 2*θ[1]*(x[9,t]-x[5,t])              - λ[1,9,t+1] == 0)
        else
            @constraint(model,   λ[1,1,t] + 2*θ[2]*x[1,t] == 0)
            @constraint(model,   λ[1,2,t] == 0)
            @NLconstraint(model, λ[1,3,t] == 0)
            @NLconstraint(model, λ[1,4,t] == 0)
            @constraint(model,   λ[1,5,t] + 2*θ[1]*(x[5,t]-x[9,t]) == 0)
            @constraint(model,   λ[1,6,t] == 0)
            @NLconstraint(model, λ[1,7,t] == 0)
            @NLconstraint(model, λ[1,8,t] == 0)
            @NLconstraint(model, λ[1,9,t] + 2*θ[1]*(x[9,t]-x[5,t]) == 0)
        end

        if t != g.h # dJ2/dx
            @constraint(model,   λ[2,1,t] + 2*θ[3]*(x[1,t]-x[5,t])                   - λ[2,1,t+1] == 0)
            @constraint(model,   λ[2,2,t]                   - λ[2,2,t+1] == 0)
            @NLconstraint(model, λ[2,3,t]                   - λ[2,3,t+1] + λ[2,1,t+1]*ΔT*x[4,t]*sin(x[3,t]) - λ[2,2,t+1]*ΔT*x[4,t]*cos(x[3,t])  == 0)
            @NLconstraint(model, λ[2,4,t]                   - λ[2,4,t+1] - λ[2,1,t+1]*ΔT*cos(x[3,t]) - λ[2,2,t+1]*ΔT*sin(x[3,t]) == 0)
            @constraint(model,   λ[2,5,t] + 2*θ[3]*(x[5,t]-x[1,t])                    - λ[2,5,t+1] == 0)
            @constraint(model,   λ[2,6,t]                   - λ[2,6,t+1] == 0)
            @NLconstraint(model, λ[2,7,t]                   - λ[2,7,t+1] + λ[2,5,t+1]*ΔT*x[8,t]*sin(x[7,t]) - λ[2,6,t+1]*ΔT*x[8,t]*cos(x[7,t]) == 0)
            @NLconstraint(model, λ[2,8,t] + 2*θ[4]*(x[8,t]-1)                  - λ[2,8,t+1] - λ[2,5,t+1]*ΔT*cos(x[7,t]) - λ[2,6,t+1]*ΔT*sin(x[7,t]) == 0)
            @NLconstraint(model, λ[2,9,t]                   -λ[2,9,t+1] == 0)
        else
            @constraint(model,   λ[2,1,t] + 2*θ[3]*(x[1,t]-x[5,t]) == 0)
            @constraint(model,   λ[2,2,t]  == 0)
            @NLconstraint(model, λ[2,3,t]  == 0)
            @NLconstraint(model, λ[2,4,t]  == 0)
            @constraint(model,   λ[2,5,t] + 2*θ[3]*(x[5,t]-x[1,t]) == 0)
            @constraint(model,   λ[2,6,t]  == 0)
            @NLconstraint(model, λ[2,7,t]  == 0)
            @NLconstraint(model, λ[2,8,t] + 2*θ[4]*(x[8,t]-1) == 0)
            @NLconstraint(model, λ[2,9,t] == 0)
        end

        # dJ1/du and dJ2/du
        @constraint(model, 4*u[1,t] - λ[1,3,t]*ΔT == 0)
        @constraint(model, 4*u[2,t] - λ[1,4,t]*ΔT == 0)
        @constraint(model, 4*u[3,t] - λ[2,7,t]*ΔT == 0)
        @constraint(model, 4*u[4,t] - λ[2,8,t]*ΔT == 0)
        if t == 1
            @NLconstraint(model, x[1,1] == x0[1] + ΔT * x0[4]*cos(x0[3]))
            @NLconstraint(model, x[2,1] == x0[2] + ΔT * x0[4]*sin(x0[3]))
            @NLconstraint(model, x[3,1] == x0[3] + ΔT * u[1,t])
            @NLconstraint(model, x[4,1] == x0[4] + ΔT * u[2,t])
            @NLconstraint(model, x[5,1] == x0[5] + ΔT * x0[8]*cos(x0[7]))
            @NLconstraint(model, x[6,1] == x0[6] + ΔT * x0[8]*sin(x0[7]))
            @NLconstraint(model, x[7,1] == x0[7] + ΔT * u[3,t])
            @NLconstraint(model, x[8,1] == x0[8] + ΔT * u[4,t])
            @NLconstraint(model, x[9,1] == x0[9])
        else
            @NLconstraint(model, x[1,t] == x[1,t-1] + ΔT * x[4,t-1]*cos(x[3,t-1]))
            @NLconstraint(model, x[2,t] == x[2,t-1] + ΔT * x[4,t-1]*sin(x[3,t-1]))
            @NLconstraint(model, x[3,t] == x[3,t-1] + ΔT * u[1,t])
            @NLconstraint(model, x[4,t] == x[4,t-1] + ΔT * u[2,t])
            @NLconstraint(model, x[5,t] == x[5,t-1] + ΔT * x[8,t-1]*cos(x[7,t-1]))
            @NLconstraint(model, x[6,t] == x[6,t-1] + ΔT * x[8,t-1]*sin(x[7,t-1]))
            @NLconstraint(model, x[7,t] == x[7,t-1] + ΔT * u[3,t])
            @NLconstraint(model, x[8,t] == x[8,t-1] + ΔT * u[4,t])
            @NLconstraint(model, x[9,t] == x[9,t-1])
        end
    end
    # @constraint(model, θ .>= -0.1*ones(3))
    optimize!(model)
    return value.(x), value.(u), value.(θ), model
end

for_sol=KKT_highway_forward_game_solve(x0, g)




anim1 = @animate for i in 1:game_horizon
    plot( [for_sol[1][1,i], for_sol[1][1,i]], [for_sol[1][2,i], for_sol[1][2,i]], markershape = :square, label = "player 1, JuMP", xlims = (-0.5, 1.5), ylims = (0, 6))
    plot!([for_sol[1][5,i], for_sol[1][5,i]], [for_sol[1][6,i], for_sol[1][6,i]], markershape = :square, label = "player 2, JuMP", xlims = (-0.5, 1.5), ylims = (0, 6))
    plot!([0], seriestype = "vline", color = "black", label = "")
    plot!([1], seriestype = "vline", color = "black", label = "") 
end
gif(anim1, "lane_guiding_for_JuMP.gif", fps = 10)

anim2 = @animate for i in 1:game_horizon
    plot( [obs_x_OL[1,i], obs_x_OL[1,i]], [obs_x_OL[2,i], obs_x_OL[2,i]], markershape = :square, label = "player 1, iLQ OLNE", xlims = (-0.5, 1.5), ylims = (0, 6))
    plot!([obs_x_OL[5,i], obs_x_OL[5,i]], [obs_x_OL[6,i], obs_x_OL[6,i]], markershape = :square, label = "player 2, iLQ OLNE", xlims = (-0.5, 1.5), ylims = (0, 6))    
    plot!([0], seriestype = "vline", color = "black", label = "")
    plot!([1], seriestype = "vline", color = "black", label = "")
end
gif(anim2, "lane_guiding_OL_iLQ.gif", fps = 10)



# ------------------------------------ Optimization problem end ------------------------------------------- #

function KKT_highway_inverse_game_solve(obs_x, obs_u, init_θ, x0, obs_time_list = 1:game_horizon-1, obs_state_index_list = [1,3,4,5,7,8], obs_control_index_list = [1,2,3,4])
    # θ=[4,0,4]
    model = Model(Ipopt.Optimizer)
    JuMP.set_silent(model)
    @variable(model, x[1:nx, 1:g.h])
    @variable(model, u[1:nu, 1:g.h])
    @variable(model, λ[1:2, 1:nx, 1:g.h])
    @variable(model, θ[1:4])
    set_start_value.(θ, init_θ)
    set_start_value.(x[1:nx, 1:g.h-1], obs_x)
    set_start_value.(u, obs_u)
    # @objective(model, Min, 0)
    @constraint(model, θ[1] + θ[2] == 6 )
    @constraint(model, θ[3] + θ[4] == 6 )
    @constraint(model, θ.>=0)
    @objective(model, Min, sum(sum((x[ii,t] - obs_x[ii,t])^2 for ii in obs_state_index_list ) for t in obs_time_list) + sum(sum((u[ii,t] - obs_u[ii,t])^2 for ii in obs_control_index_list) for t in obs_time_list) )
    for t in 1:g.h # for each time t within the game horizon
        if t != g.h # dJ1/dx
            @constraint(model,   λ[1,1,t] + 2*θ[2]*x[1,t]             - λ[1,1,t+1] == 0)
            @constraint(model,   λ[1,2,t]               - λ[1,2,t+1] == 0)
            @NLconstraint(model, λ[1,3,t]               - λ[1,3,t+1] + λ[1,1,t+1]*ΔT*x[4,t]*sin(x[3,t]) - λ[1,2,t+1]*ΔT*x[4,t]*cos(x[3,t])  == 0)
            @NLconstraint(model, λ[1,4,t]               - λ[1,4,t+1] - λ[1,1,t+1]*ΔT*cos(x[3,t]) - λ[1,2,t+1]*ΔT*sin(x[3,t]) == 0)
            @constraint(model,   λ[1,5,t] + 2*θ[1]*(x[5,t]-x[9,t])           - λ[1,5,t+1] == 0)
            @constraint(model,   λ[1,6,t]               - λ[1,6,t+1] == 0)
            @NLconstraint(model, λ[1,7,t]               - λ[1,7,t+1] + λ[1,5,t+1]*ΔT*x[8,t]*sin(x[7,t]) - λ[1,6,t+1]*ΔT*x[8,t]*cos(x[7,t]) == 0)
            @NLconstraint(model, λ[1,8,t]               - λ[1,8,t+1] - λ[1,5,t+1]*ΔT*cos(x[7,t]) - λ[1,6,t+1]*ΔT*sin(x[7,t]) == 0)
            @NLconstraint(model, λ[1,9,t] + 2*θ[1]*(x[9,t]-x[5,t])              - λ[1,9,t+1] == 0)
        else
            @constraint(model,   λ[1,1,t] + 2*θ[2]*x[1,t] == 0)
            @constraint(model,   λ[1,2,t] == 0)
            @NLconstraint(model, λ[1,3,t] == 0)
            @NLconstraint(model, λ[1,4,t] == 0)
            @constraint(model,   λ[1,5,t] + 2*θ[1]*(x[5,t]-x[9,t]) == 0)
            @constraint(model,   λ[1,6,t] == 0)
            @NLconstraint(model, λ[1,7,t] == 0)
            @NLconstraint(model, λ[1,8,t] == 0)
            @NLconstraint(model, λ[1,9,t] + 2*θ[1]*(x[9,t]-x[5,t]) == 0)
        end

        if t != g.h # dJ2/dx
            @constraint(model,   λ[2,1,t] + 2*θ[3]*(x[1,t]-x[5,t])                   - λ[2,1,t+1] == 0)
            @constraint(model,   λ[2,2,t]                   - λ[2,2,t+1] == 0)
            @NLconstraint(model, λ[2,3,t]                   - λ[2,3,t+1] + λ[2,1,t+1]*ΔT*x[4,t]*sin(x[3,t]) - λ[2,2,t+1]*ΔT*x[4,t]*cos(x[3,t])  == 0)
            @NLconstraint(model, λ[2,4,t]                   - λ[2,4,t+1] - λ[2,1,t+1]*ΔT*cos(x[3,t]) - λ[2,2,t+1]*ΔT*sin(x[3,t]) == 0)
            @constraint(model,   λ[2,5,t] + 2*θ[3]*(x[5,t]-x[1,t])                    - λ[2,5,t+1] == 0)
            @constraint(model,   λ[2,6,t]                   - λ[2,6,t+1] == 0)
            @NLconstraint(model, λ[2,7,t]                   - λ[2,7,t+1] + λ[2,5,t+1]*ΔT*x[8,t]*sin(x[7,t]) - λ[2,6,t+1]*ΔT*x[8,t]*cos(x[7,t]) == 0)
            @NLconstraint(model, λ[2,8,t] + 2*θ[4]*(x[8,t]-1)                  - λ[2,8,t+1] - λ[2,5,t+1]*ΔT*cos(x[7,t]) - λ[2,6,t+1]*ΔT*sin(x[7,t]) == 0)
            @NLconstraint(model, λ[2,9,t]                   -λ[2,9,t+1] == 0)
        else
            @constraint(model,   λ[2,1,t] + 2*θ[3]*(x[1,t]-x[5,t]) == 0)
            @constraint(model,   λ[2,2,t]  == 0)
            @NLconstraint(model, λ[2,3,t]  == 0)
            @NLconstraint(model, λ[2,4,t]  == 0)
            @constraint(model,   λ[2,5,t] + 2*θ[3]*(x[5,t]-x[1,t]) == 0)
            @constraint(model,   λ[2,6,t]  == 0)
            @NLconstraint(model, λ[2,7,t]  == 0)
            @NLconstraint(model, λ[2,8,t] + 2*θ[4]*(x[8,t]-1) == 0)
            @NLconstraint(model, λ[2,9,t] == 0)
        end

        # dJ1/du and dJ2/du
        @constraint(model, 4*u[1,t] - λ[1,3,t]*ΔT == 0)
        @constraint(model, 4*u[2,t] - λ[1,4,t]*ΔT == 0)
        @constraint(model, 4*u[3,t] - λ[2,7,t]*ΔT == 0)
        @constraint(model, 4*u[4,t] - λ[2,8,t]*ΔT == 0)
        if t == 1
            @NLconstraint(model, x[1,1] == x0[1] + ΔT * x0[4]*cos(x0[3]))
            @NLconstraint(model, x[2,1] == x0[2] + ΔT * x0[4]*sin(x0[3]))
            @NLconstraint(model, x[3,1] == x0[3] + ΔT * u[1,t])
            @NLconstraint(model, x[4,1] == x0[4] + ΔT * u[2,t])
            @NLconstraint(model, x[5,1] == x0[5] + ΔT * x0[8]*cos(x0[7]))
            @NLconstraint(model, x[6,1] == x0[6] + ΔT * x0[8]*sin(x0[7]))
            @NLconstraint(model, x[7,1] == x0[7] + ΔT * u[3,t])
            @NLconstraint(model, x[8,1] == x0[8] + ΔT * u[4,t])
            @NLconstraint(model, x[9,1] == x0[9])
        else
            @NLconstraint(model, x[1,t] == x[1,t-1] + ΔT * x[4,t-1]*cos(x[3,t-1]))
            @NLconstraint(model, x[2,t] == x[2,t-1] + ΔT * x[4,t-1]*sin(x[3,t-1]))
            @NLconstraint(model, x[3,t] == x[3,t-1] + ΔT * u[1,t])
            @NLconstraint(model, x[4,t] == x[4,t-1] + ΔT * u[2,t])
            @NLconstraint(model, x[5,t] == x[5,t-1] + ΔT * x[8,t-1]*cos(x[7,t-1]))
            @NLconstraint(model, x[6,t] == x[6,t-1] + ΔT * x[8,t-1]*sin(x[7,t-1]))
            @NLconstraint(model, x[7,t] == x[7,t-1] + ΔT * u[3,t])
            @NLconstraint(model, x[8,t] == x[8,t-1] + ΔT * u[4,t])
            @NLconstraint(model, x[9,t] == x[9,t-1])
        end
    end
    optimize!(model)
    return value.(x), value.(u), value.(θ), model
end

inv_sol = KKT_highway_inverse_game_solve(obs_x_FB[:,2:end], obs_u_FB, 3*ones(4), x0);


anim1 = @animate for i in 1:game_horizon
    plot( [inv_sol[1][1,i], inv_sol[1][1,i]], [inv_sol[1][2,i], inv_sol[1][2,i]], markershape = :square, label = "player 1, JuMP", xlims = (-0.5, 1.5), ylims = (0, 6))
    plot!([inv_sol[1][5,i], inv_sol[1][5,i]], [inv_sol[1][6,i], inv_sol[1][6,i]], markershape = :square, label = "player 2, JuMP", xlims = (-0.5, 1.5), ylims = (0, 6))
    plot!([0], seriestype = "vline", color = "black", label = "")
    plot!([1], seriestype = "vline", color = "black", label = "") 
end
gif(anim1, "lane_guiding_inv_JuMP.gif", fps = 10)

anim2 = @animate for i in 1:game_horizon
    plot( [obs_x_OL[1,i], obs_x_OL[1,i]], [obs_x_OL[2,i], obs_x_OL[2,i]], markershape = :square, label = "player 1, iLQ OLNE", xlims = (-0.5, 1.5), ylims = (0, 6))
    plot!([obs_x_OL[5,i], obs_x_OL[5,i]], [obs_x_OL[6,i], obs_x_OL[6,i]], markershape = :square, label = "player 2, iLQ OLNE", xlims = (-0.5, 1.5), ylims = (0, 6))    
    plot!([0], seriestype = "vline", color = "black", label = "")
    plot!([1], seriestype = "vline", color = "black", label = "")
end
gif(anim2, "lane_guiding_OL_iLQ.gif", fps = 10)




# 1. generate noisy observation, 10 for each noise level. If 10 noise level, then we generate 10x10 observations ✓
# 2. run inverse_KKT for every observation and record the state loss

num_clean_traj = 1
x0_set = [x0 for ii in 1:num_clean_traj]
expert_traj_list, c_expert = generate_expert_traj(g, solver1, x0_set, num_clean_traj)
if sum([c_expert[ii]==false for ii in 1:length(c_expert)]) >0
    @warn "regenerate expert demonstrations because some of the expert demonstration not converged!!!"
end

game = g
solver = solver1

# The below: generate random expert trajectories
num_obs = 10
noise_level_list = 0.005:0.005:0.03
num_noise_level = length(noise_level_list)
num_noise_level = length(noise_level_list)
noisy_expert_traj_list = [[[zero(SystemTrajectory, game) for kk in 1:num_obs] for jj in 1:num_noise_level] for ii in 1:num_clean_traj];

for ii in 1:num_clean_traj
    for jj in 1:num_noise_level
        tmp = generate_noisy_observation(nx, nu, game, expert_traj_list[ii], noise_level_list[jj], num_obs)
        for kk in 1:num_obs
            for t in 1:game_horizon
                noisy_expert_traj_list[ii][jj][kk].x[t] = tmp[kk].x[t]
                noisy_expert_traj_list[ii][jj][kk].u[t] = tmp[kk].u[t]
            end
        end
    end
end

# conv_table_list = [[[] for jj in 1:num_noise_level] for ii in 1:num_clean_traj];
# sol_table_list = deepcopy(conv_table_list);
# loss_table_list = deepcopy(conv_table_list);
# θ_list_list = [[[] for jj in 1:num_noise_level] for ii in 1:num_clean_traj];
# optim_loss_list_list = [[[] for jj in 1:num_noise_level] for ii in 1:num_clean_traj];
# generalization_error_list = [[[] for jj in 1:num_noise_level] for ii in 1:num_clean_traj];
# ground_truth_loss_list = [[[] for jj in 1:num_noise_level] for ii in 1:num_clean_traj];

θ₀ = 3*ones(4);
inv_traj_x_list = [[[] for jj in 1:num_obs] for ii in 1:length(noise_level_list)];
inv_traj_u_list = [[[] for jj in 1:num_obs] for ii in 1:length(noise_level_list)];
inv_sol_list = [[[] for jj in 1:num_obs] for ii in 1:length(noise_level_list)];
inv_loss_list = [[[] for jj in 1:num_obs] for ii in 1:length(noise_level_list)];
inv_mean_generalization_loss_list = [[[] for jj in 1:num_obs] for ii in 1:length(noise_level_list)];
inv_var_generalization_loss_list = [[[] for jj in 1:num_obs] for ii in 1:length(noise_level_list)];
inv_model_list = [[[] for jj in 1:num_obs] for ii in 1:length(noise_level_list)];
inv_ground_truth_loss_list = [[[] for jj in 1:num_obs] for ii in 1:length(noise_level_list)];
inv_ground_truth_computed_traj_list = [[[] for jj in 1:num_obs] for ii in 1:length(noise_level_list)];

num_test = 10
test_x0_set = [x0 + 0.5*[0,0,0,0,0,0,0,0,rand(1)[1]] for ii in 1:num_test];
test_expert_traj_list, c_test_expert = generate_expert_traj(game, solver, test_x0_set, num_test);


obs_time_list = 1:game_horizon-1
obs_state_list = 1:nx
obs_control_list = 1:nu
index = 1
for noise in 1:length(noise_level_list)
    for ii in 1:num_obs
        tmp_expert_traj_x = noisy_expert_traj_list[index][noise][ii].x
        tmp_expert_traj_u = noisy_expert_traj_list[index][noise][ii].u
        
        tmp_obs_x = transpose(mapreduce(permutedims, vcat, Vector([Vector(tmp_expert_traj_x[t]) for t in 1:game.h])))
        tmp_obs_u = transpose(mapreduce(permutedims, vcat, Vector([Vector(tmp_expert_traj_u[t]) for t in 1:game.h])))
        tmp_inv_traj_x, tmp_inv_traj_u, tmp_inv_sol, tmp_inv_model = KKT_highway_inverse_game_solve(tmp_obs_x[:,2:end], tmp_obs_u, θ₀, x0_set[index])
        tmp_inv_loss = objective_value(tmp_inv_model)
        println("The $(ii)-th observation of $(noise)-th noise level")
        # solution_summary(tmp_inv_model)
        tmp_ground_truth_loss_value, tmp_ground_truth_computed_traj, _, _=loss(tmp_inv_sol, iLQGames.dynamics(game), "FBNE_costate", expert_traj_list[index], false, false, [], [], obs_time_list, obs_state_list, obs_control_list) 
        # @infiltrate
        # tmp_test_sol = [[] for jj in 1:num_test]
        tmp_test_loss_value = zeros(num_test)
        for jj in 1:num_test
            # @infiltrate
            tmp_test_loss_value[jj], _,_,_ = loss(tmp_inv_sol, iLQGames.dynamics(game), "FBNE_costate", test_expert_traj_list[jj], false, false, [],[],obs_time_list, obs_state_list, obs_control_list)
        end
        @infiltrate
        push!(inv_mean_generalization_loss_list[noise][ii], mean(tmp_test_loss_value))
        push!(inv_var_generalization_loss_list[noise][ii], var(tmp_test_loss_value))
        push!(inv_sol_list[noise][ii], tmp_inv_sol)
        push!(inv_loss_list[noise][ii], objective_value(tmp_inv_model))
        push!(inv_traj_x_list[noise][ii], tmp_inv_traj_x)
        push!(inv_traj_u_list[noise][ii], tmp_inv_traj_u)
        push!(inv_ground_truth_loss_list[noise][ii], tmp_ground_truth_loss_value)
        push!(inv_ground_truth_computed_traj_list[noise][ii], tmp_ground_truth_computed_traj)
        push!(inv_model_list[noise][ii], tmp_inv_model)
    end
end




for ii in 1:num_clean_traj
    for jj in 1:num_noise_level
        conv_table,sol_table,loss_table,grad_table,equi_table,iter_table,ground_truth_loss = run_experiment(game,θ₀,[x0_set[ii] for kk in 1:num_obs], 
                                                                                                noisy_expert_traj_list[ii][jj], parameterized_cost, GD_iter_num, 20, 1e-8, 
                                                                                                1:game_horizon-1,1:nx, 1:nu, "FBNE_costate", 0.00000000001, false, 10.0, expert_traj_list[ii])
        θ_list, index_list, optim_loss_list = get_the_best_possible_reward_estimate_single([x0_set[ii] for kk in 1:num_obs], ["FBNE_costate","FBNE_costate"], sol_table, loss_table, equi_table)
        # generalization_error = generalization_loss(games[ii], θ_list[1], [x0+0.5*(rand(4)-0.5*ones(4)) for ii in 1:num_generalization], 
        #                             expert_traj_list, parameterized_cost, equilibrium_type_list) #problem
        push!(θ_list_list[ii][jj], θ_list)
        push!(optim_loss_list_list[ii][jj], optim_loss_list)
        push!(ground_truth_loss_list[ii][jj], ground_truth_loss)
        push!(generalization_error_list[ii][jj], generalization_error)
    end
end





## Plot state_prediction_loss vs. noise_variance level








