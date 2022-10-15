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

num_players=3
nx, nu, ΔT, game_horizon = 4*num_players+1, 2*num_players, 0.1, 40
struct ThreeCar <: ControlSystem{ΔT,nx,nu} end
dx(cs::ThreeCar, x, u, t) = SVector(x[4]cos(x[3]),   x[4]sin(x[3]),   u[1], u[2], 
                                        x[8]cos(x[7]),   x[8]sin(x[7]),   u[3], u[4],
                                        x[12]cos(x[11]), x[12]sin(x[11]), u[5], u[6],
                                        0
                                        )
dynamics = ThreeCar()

# platonning
x0 = SVector(0.0, 3, pi/2, 2,       0.3, 0, pi/2, 2,      0.7, 2,pi/2,1,                   0.2)
costs = (FunctionPlayerCost((g,x,u,t) -> ( 10*(x[5]-x[13])^2  + 4*(x[3]-pi/2)^2   +8*(x[4]-2)^2       +2*(u[1]^2 + u[2]^2)    )),
         FunctionPlayerCost((g,x,u,t) -> ( 10*(x[5]-x[1])^2   + 8*(x[8]-2)^2      +4*(x[7]-pi/2)^2     -log((x[5]-x[9])^2+(x[6]-x[10])^2)    +2*(u[3]^2+u[4]^2)    )),
         FunctionPlayerCost((g,x,u,t) -> ( 2*(x[9]-x0[9])^2   + 2*(u[5]^2+u[6]^2)  ))
    )
player_inputs = (SVector(1,2), SVector(3,4), SVector(5,6))
g = GeneralGame(game_horizon, player_inputs, dynamics, costs)
solver1 = iLQSolver(g, max_scale_backtrack=5, max_elwise_diff_step=Inf, equilibrium_type="OLNE_costate")
c1, expert_traj1, strategies1 = solve(g, solver1, x0)
solver2 = iLQSolver(g, max_scale_backtrack=5, max_elwise_diff_step=Inf, equilibrium_type="FBNE_costate")
c2, expert_traj2, strategies2 = solve(g, solver2, x0)

θ_true = [0, 10, 0, 10]
obs_x_FB = transpose(mapreduce(permutedims, vcat, Vector([Vector(expert_traj2.x[t]) for t in 1:g.h])))
obs_u_FB = transpose(mapreduce(permutedims, vcat, Vector([Vector(expert_traj2.u[t]) for t in 1:g.h])))
obs_x_OL = transpose(mapreduce(permutedims, vcat, Vector([Vector(expert_traj1.x[t]) for t in 1:g.h])))
obs_u_OL = transpose(mapreduce(permutedims, vcat, Vector([Vector(expert_traj1.u[t]) for t in 1:g.h])))

noisy_obs_x_OL = transpose(mapreduce(permutedims, vcat, Vector([Vector(noisy_expert_traj_list[1][1][1].x[t]) for t in 1:g.h])))
noisy_obs_u_OL = transpose(mapreduce(permutedims, vcat, Vector([Vector(noisy_expert_traj_list[1][1][1].u[t]) for t in 1:g.h])))



# ------------------------------------ Optimization problem begin ------------------------------------------- #

# function KKT_highway_forward_game_solve( x0,g)
#     model = Model(Ipopt.Optimizer)
#     @variable(model, x[1:nx, 1:g.h+1])
#     @variable(model, u[1:nu, 1:g.h])
#     @variable(model, θ[1:4])
#     set_start_value.(θ, θ_true)
#     set_start_value.(x[:,1:g.h], noisy_obs_x_OL)
#     set_start_value.(u[:,1:g.h], noisy_obs_u_OL)
#     @variable(model, λ[1:3, 1:nx, 1:g.h])
#     @objective(model, Min, 0)
#     ΔT = 0.1
#     for t in 1:g.h # for each time t within the game horizon
#         if t != g.h # dJ1/dx
#             @constraint(model, λ[1,1,t] +θ[1]*x[1,t]              - λ[1,1,t+1] == 0)
#             @constraint(model, λ[1,2,t]               - λ[1,2,t+1] == 0)
#             @NLconstraint(model, λ[1,3,t] + 8*(x[3,t]-pi/2)              - λ[1,3,t+1] + λ[1,1,t+1]*ΔT*x[4,t]*sin(x[3,t]) - λ[1,2,t+1]*ΔT*x[4,t]*cos(x[3,t])  == 0)
#             @NLconstraint(model, λ[1,4,t] + 16*(x[4,t]-2)                - λ[1,4,t+1] - λ[1,1,t+1]*ΔT*cos(x[3,t]) - λ[1,2,t+1]*ΔT*sin(x[3,t]) == 0)
#             @constraint(model, λ[1,5,t] + 2*θ[2]*(x[5,t]-x[13,t])          - λ[1,5,t+1] == 0)
#             @constraint(model, λ[1,6,t]               - λ[1,6,t+1] == 0)
#             @NLconstraint(model, λ[1,7,t]               - λ[1,7,t+1] + λ[1,5,t+1]*ΔT*x[8,t]*sin(x[7,t]) - λ[1,6,t+1]*ΔT*x[8,t]*cos(x[7,t]) == 0)
#             @NLconstraint(model, λ[1,8,t]               - λ[1,8,t+1] - λ[1,5,t+1]*ΔT*cos(x[7,t]) - λ[1,6,t+1]*ΔT*sin(x[7,t]) == 0)
#             @constraint(model, λ[1,9,t]               - λ[1,9,t+1] == 0)
#             @constraint(model, λ[1,10,t]              - λ[1,10,t+1] == 0)
#             @NLconstraint(model, λ[1,11,t]              - λ[1,11,t+1] + λ[1,9,t+1]*ΔT*x[12,t]*sin(x[11,t]) - λ[1,10,t+1]*ΔT*x[12,t]*cos(x[11,t]) == 0)
#             @NLconstraint(model, λ[1,12,t]              - λ[1,12,t+1] - λ[1,9,t+1]*ΔT*cos(x[11,t]) - λ[1,10,t+1]*ΔT*sin(x[11,t]) == 0)
#             @NLconstraint(model, λ[1,13,t] + 2*θ[2]*(x[13,t]-x[5,t])              - λ[1,13,t+1] == 0)
#         else
#             @constraint(model, λ[1,1,t] +θ[1]*x[1,t]== 0)
#             @constraint(model, λ[1,2,t] == 0)
#             @NLconstraint(model, λ[1,3,t] + 8*(x[3,t]-pi/2)== 0)
#             @NLconstraint(model, λ[1,4,t] + 16*(x[4,t]-2)== 0)
#             @constraint(model, λ[1,5,t] + 2*θ[2]*(x[5,t]-x[13,t]) == 0)
#             @constraint(model, λ[1,6,t] == 0)
#             @NLconstraint(model, λ[1,7,t] == 0)
#             @NLconstraint(model, λ[1,8,t] == 0)
#             @constraint(model, λ[1,9,t] == 0)
#             @constraint(model, λ[1,10,t] == 0)
#             @NLconstraint(model, λ[1,11,t] == 0)
#             @NLconstraint(model, λ[1,12,t] == 0)
#             @NLconstraint(model, λ[1,13,t] + 2*θ[2]*(x[13,t]-x[5,t]) == 0)
#         end
#         if t != g.h # dJ2/dx
#             @constraint(model, λ[2,1,t] +2*θ[4]*(x[1,t]-x[5,t])              - λ[2,1,t+1] == 0)
#             @constraint(model, λ[2,2,t]               - λ[2,2,t+1] == 0)
#             @NLconstraint(model, λ[2,3,t]               - λ[2,3,t+1] + λ[2,1,t+1]*ΔT*x[4,t]*sin(x[3,t]) - λ[2,2,t+1]*ΔT*x[4,t]*cos(x[3,t])  == 0)
#             @NLconstraint(model, λ[2,4,t]                 - λ[2,4,t+1] - λ[2,1,t+1]*ΔT*cos(x[3,t]) - λ[2,2,t+1]*ΔT*sin(x[3,t]) == 0)
#             @NLconstraint(model, λ[2,5,t] +θ[3]*x[5,t] +2*θ[4]*(x[5,t]-x[1,t]) -2*(x[5,t]-x[9,t])/((x[5,t]-x[9,t])^2+(x[6,t]-x[10,t])^2)          - λ[2,5,t+1] == 0)
#             @NLconstraint(model, λ[2,6,t] -2*(x[6,t]-x[10,t])/((x[5,t]-x[9,t])^2 + (x[6,t]-x[10,t])^2)              - λ[2,6,t+1] == 0)
#             @NLconstraint(model, λ[2,7,t] +8*(x[7,t]-pi/2)              - λ[2,7,t+1] + λ[2,5,t+1]*ΔT*x[8,t]*sin(x[7,t]) - λ[2,6,t+1]*ΔT*x[8,t]*cos(x[7,t]) == 0)
#             @NLconstraint(model, λ[2,8,t] +16*(x[8,t]-2)              - λ[2,8,t+1] - λ[2,5,t+1]*ΔT*cos(x[7,t]) - λ[2,6,t+1]*ΔT*sin(x[7,t]) == 0)
#             @NLconstraint(model, λ[2,9,t] -2*(x[9,t]-x[5,t])/((x[5,t]-x[9,t])^2+(x[6,t]-x[10,t])^2)              - λ[2,9,t+1] == 0)
#             @NLconstraint(model, λ[2,10,t] -2*(x[10,t]-x[6,t])/((x[5,t]-x[9,t])^2+(x[6,t]-x[10,t])^2)             - λ[2,10,t+1] == 0)
#             @NLconstraint(model, λ[2,11,t]              - λ[2,11,t+1] + λ[2,9,t+1]*ΔT*x[12,t]*sin(x[11,t]) - λ[2,10,t+1]*ΔT*x[12,t]*cos(x[11,t]) == 0)
#             @NLconstraint(model, λ[2,12,t]              - λ[2,12,t+1] - λ[2,9,t+1]*ΔT*cos(x[11,t]) - λ[2,10,t+1]*ΔT*sin(x[11,t]) == 0)
#             @NLconstraint(model, λ[2,13,t]              - λ[2,13,t+1] == 0)
#         else
#             @constraint(model, λ[2,1,t] +2*θ[4]*(x[1,t]-x[5,t])== 0)
#             @constraint(model, λ[2,2,t] == 0)
#             @NLconstraint(model, λ[2,3,t] == 0)
#             @NLconstraint(model, λ[2,4,t] == 0)
#             @NLconstraint(model, λ[2,5,t] +θ[3]*x[5,t] +2*θ[4]*(x[5,t]-x[1,t]) -2*(x[5,t]-x[9,t])/((x[5,t]-x[9,t])^2+(x[6,t]-x[10,t])^2) == 0)
#             @NLconstraint(model, λ[2,6,t] -2*(x[6,t]-x[10,t])/((x[5,t]-x[9,t])^2 + (x[6,t]-x[10,t])^2)== 0)
#             @NLconstraint(model, λ[2,7,t] +8*(x[7,t]-pi/2)== 0)
#             @NLconstraint(model, λ[2,8,t] +16*(x[8,t]-2)== 0)
#             @NLconstraint(model, λ[2,9,t] -2*(x[9,t]-x[5,t])/((x[5,t]-x[9,t])^2+(x[6,t]-x[10,t])^2)== 0)
#             @NLconstraint(model, λ[2,10,t] -2*(x[10,t]-x[6,t])/((x[5,t]-x[9,t])^2+(x[6,t]-x[10,t])^2)== 0)
#             @NLconstraint(model, λ[2,11,t] == 0)
#             @NLconstraint(model, λ[2,12,t] == 0)
#             @NLconstraint(model, λ[2,13,t] == 0)
#         end
#         if t != g.h # dJ3/dx
#             @constraint(model,   λ[3,1,t]               - λ[3,1,t+1] == 0)
#             @constraint(model,   λ[3,2,t]               - λ[3,2,t+1] == 0)
#             @NLconstraint(model, λ[3,3,t]               - λ[3,3,t+1] + λ[3,1,t+1]*ΔT*x[4,t]*sin(x[3,t]) - λ[3,2,t+1]*ΔT*x[4,t]*cos(x[3,t])  == 0)
#             @NLconstraint(model, λ[3,4,t]               - λ[3,4,t+1] - λ[3,1,t+1]*ΔT*cos(x[3,t]) - λ[3,2,t+1]*ΔT*sin(x[3,t]) == 0)
#             @constraint(model,   λ[3,5,t]               - λ[3,5,t+1] == 0)
#             @constraint(model,   λ[3,6,t]               - λ[3,6,t+1] == 0)
#             @NLconstraint(model, λ[3,7,t]               - λ[3,7,t+1] + λ[3,5,t+1]*ΔT*x[8,t]*sin(x[7,t]) - λ[3,6,t+1]*ΔT*x[8,t]*cos(x[7,t]) == 0)
#             @NLconstraint(model, λ[3,8,t]               - λ[3,8,t+1] - λ[3,5,t+1]*ΔT*cos(x[7,t]) - λ[3,6,t+1]*ΔT*sin(x[7,t]) == 0)
#             @constraint(model,   λ[3,9,t] +4*(x[9,t]-x0[9])              - λ[3,9,t+1] == 0)
#             @constraint(model,   λ[3,10,t]              - λ[3,10,t+1] == 0)
#             @NLconstraint(model, λ[3,11,t]              - λ[3,11,t+1] + λ[3,9,t+1]*ΔT*x[12,t]*sin(x[11,t]) - λ[3,10,t+1]*ΔT*x[12,t]*cos(x[11,t]) == 0)
#             @NLconstraint(model, λ[3,12,t]              - λ[3,12,t+1] - λ[3,9,t+1]*ΔT*cos(x[11,t]) - λ[3,10,t+1]*ΔT*sin(x[11,t]) == 0)
#             @NLconstraint(model, λ[3,13,t]              - λ[3,13,t+1] == 0)
#         else
#             @constraint(model,   λ[3,1,t] == 0)
#             @constraint(model,   λ[3,2,t] == 0)
#             @NLconstraint(model, λ[3,3,t] == 0)
#             @NLconstraint(model, λ[3,4,t] == 0)
#             @constraint(model,   λ[3,5,t] == 0)
#             @constraint(model,   λ[3,6,t] == 0)
#             @NLconstraint(model, λ[3,7,t] == 0)
#             @NLconstraint(model, λ[3,8,t] == 0)
#             @constraint(model,   λ[3,9,t] +4*(x[9,t]-x0[9])== 0)
#             @constraint(model,   λ[3,10,t] == 0)
#             @NLconstraint(model, λ[3,11,t] == 0)
#             @NLconstraint(model, λ[3,12,t] == 0)
#             @NLconstraint(model, λ[3,13,t] == 0)
#         end
#         # dJ1/du, dJ2/du and dJ3/du
#         @constraint(model, 4*u[1,t] - λ[1,3,t]*ΔT == 0)
#         @constraint(model, 4*u[2,t] - λ[1,4,t]*ΔT == 0)
#         @constraint(model, 4*u[3,t] - λ[2,7,t]*ΔT == 0)
#         @constraint(model, 4*u[4,t] - λ[2,8,t]*ΔT == 0)
#         @constraint(model, 4*u[5,t] - λ[3,11,t]*ΔT == 0)
#         @constraint(model, 4*u[6,t] - λ[3,12,t]*ΔT == 0)
#         if t == 1
#             @NLconstraint(model, x[1,1] == x0[1] + ΔT * x0[4]*cos(x0[3]))
#             @NLconstraint(model, x[2,1] == x0[2] + ΔT * x0[4]*sin(x0[3]))
#             @NLconstraint(model, x[3,1] == x0[3] + ΔT * u[1,t])
#             @NLconstraint(model, x[4,1] == x0[4] + ΔT * u[2,t])
#             @NLconstraint(model, x[5,1] == x0[5] + ΔT * x0[8]*cos(x0[7]))
#             @NLconstraint(model, x[6,1] == x0[6] + ΔT * x0[8]*sin(x0[7]))
#             @NLconstraint(model, x[7,1] == x0[7] + ΔT * u[3,t])
#             @NLconstraint(model, x[8,1] == x0[8] + ΔT * u[4,t])
#             @NLconstraint(model, x[9,1] == x0[9] + ΔT * x0[12]*cos(x0[11]))
#             @NLconstraint(model, x[10,1] == x0[10] + ΔT * x0[12]*sin(x0[11]))
#             @NLconstraint(model, x[11,1] == x0[11] + ΔT * u[5,t])
#             @NLconstraint(model, x[12,1] == x0[12] + ΔT * u[6,t])
#             @NLconstraint(model, x[13,1] == x0[13])
#         else
#             @NLconstraint(model, x[1,t] == x[1,t-1] + ΔT * x[4,t-1]*cos(x[3,t-1]))
#             @NLconstraint(model, x[2,t] == x[2,t-1] + ΔT * x[4,t-1]*sin(x[3,t-1]))
#             @NLconstraint(model, x[3,t] == x[3,t-1] + ΔT * u[1,t])
#             @NLconstraint(model, x[4,t] == x[4,t-1] + ΔT * u[2,t])
#             @NLconstraint(model, x[5,t] == x[5,t-1] + ΔT * x[8,t-1]*cos(x[7,t-1]))
#             @NLconstraint(model, x[6,t] == x[6,t-1] + ΔT * x[8,t-1]*sin(x[7,t-1]))
#             @NLconstraint(model, x[7,t] == x[7,t-1] + ΔT * u[3,t])
#             @NLconstraint(model, x[8,t] == x[8,t-1] + ΔT * u[4,t])
#             @NLconstraint(model, x[9,t] == x[9,t-1] + ΔT * x[12,t-1]*cos(x[11,t-1]))
#             @NLconstraint(model, x[10,t] == x[10,t-1] + ΔT * x[12,t-1]*sin(x[11,t-1]))
#             @NLconstraint(model, x[11,t] == x[11,t-1] + ΔT * u[5,t])
#             @NLconstraint(model, x[12,t] == x[12,t-1] + ΔT * u[6,t])
#             @NLconstraint(model, x[13,t] == x[13,t-1])
#         end
#     end
#     # @constraint(model, θ .>= -0.1*ones(3))
#     optimize!(model)
#     return value.(x), value.(u), value.(θ), model
# end

# for_sol=KKT_highway_forward_game_solve(x0, g)

include("../examples/cars3_def_2_KKT_x0.jl")
function two_level_inv_KKT(obs_x, θ₀, obs_time_list, obs_state_list)
    # first level, solve a feasible dynamics point
    feasible_sol = level_1_KKT_x0(obs_x, obs_time_list, obs_state_list);
    # second level, solver a good θ
    overall_sol = level_2_KKT_x0(feasible_sol[1],feasible_sol[2], obs_x, θ₀, obs_time_list, obs_state_list)
    return overall_sol
end
inv_sol=two_level_inv_KKT(obs_x_OL, 5*ones(4), 1:game_horizon-1, 1:nx)

# anim1 = @animate for i in 1:game_horizon
#     plot( [for_sol[1][1,i], for_sol[1][1,i]], [for_sol[1][2,i], for_sol[1][2,i]], markershape = :square, label = "player 1, JuMP", xlims = (-0.5, 1.5), ylims = (0, 6))
#     plot!([for_sol[1][5,i], for_sol[1][5,i]], [for_sol[1][6,i], for_sol[1][6,i]], markershape = :square, label = "player 2, JuMP", xlims = (-0.5, 1.5), ylims = (0, 6))
#     plot!([0], seriestype = "vline", color = "black", label = "")
#     plot!([1], seriestype = "vline", color = "black", label = "") 
# end
# gif(anim1, "lane_guiding_for_JuMP.gif", fps = 10)

# anim2 = @animate for i in 1:game_horizon
#     plot( [obs_x_OL[1,i], obs_x_OL[1,i]], [obs_x_OL[2,i], obs_x_OL[2,i]], markershape = :square, label = "player 1, iLQ OLNE", xlims = (-0.5, 1.5), ylims = (0, 6))
#     plot!([obs_x_OL[5,i], obs_x_OL[5,i]], [obs_x_OL[6,i], obs_x_OL[6,i]], markershape = :square, label = "player 2, iLQ OLNE", xlims = (-0.5, 1.5), ylims = (0, 6))    
#     plot!([0], seriestype = "vline", color = "black", label = "")
#     plot!([1], seriestype = "vline", color = "black", label = "")
# end
# gif(anim2, "lane_guiding_OL_iLQ.gif", fps = 10)



# ------------------------------------ Optimization problem end ------------------------------------------- #

# function KKT_highway_inverse_game_solve(obs_x, init_θ, x0, obs_time_list = 1:game_horizon-1, obs_state_index_list = [1,3,4,5,7,8], 
#     obs_control_index_list = [1,2,3,4])
#     # θ=[4,0,4]
#     # if no_control==true
#     #     ctrl_coeff=0
#     # else
#     #     ctrl_coeff=1
#     # end
#     model = Model(Ipopt.Optimizer)
#     JuMP.set_silent(model)
#     @variable(model, x[1:nx, 1:g.h])
#     @variable(model, u[1:nu, 1:g.h])
#     @variable(model, λ[1:2, 1:nx, 1:g.h])
#     @variable(model, θ[1:4])
#     set_start_value.(θ, init_θ)
#     set_start_value.(x[1:nx, 1:g.h-1], obs_x)
#     # set_start_value.(u, obs_u)
#     # @objective(model, Min, 0)
#     @constraint(model, θ[1] + θ[2] == 8 )
#     @constraint(model, θ[3] + θ[4] == 8 )
#     @constraint(model, θ.>=0)
#     @objective(model, Min, sum(sum((x[ii,t] - obs_x[ii,t])^2 for ii in obs_state_index_list ) for t in obs_time_list))# + ctrl_coeff*sum(sum((u[ii,t] - obs_u[ii,t])^2 for ii in obs_control_index_list) for t in obs_time_list) )
#     for t in 1:g.h # for each time t within the game horizon
#         if t != g.h # dJ1/dx
#             @constraint(model,   λ[1,1,t] + 2*θ[2]*x[1,t]             - λ[1,1,t+1] == 0)
#             @constraint(model,   λ[1,2,t]               - λ[1,2,t+1] == 0)
#             @NLconstraint(model, λ[1,3,t]               - λ[1,3,t+1] + λ[1,1,t+1]*ΔT*x[4,t]*sin(x[3,t]) - λ[1,2,t+1]*ΔT*x[4,t]*cos(x[3,t])  == 0)
#             @NLconstraint(model, λ[1,4,t]               - λ[1,4,t+1] - λ[1,1,t+1]*ΔT*cos(x[3,t]) - λ[1,2,t+1]*ΔT*sin(x[3,t]) == 0)
#             @constraint(model,   λ[1,5,t] + 2*θ[1]*(x[5,t]-x[9,t])           - λ[1,5,t+1] == 0)
#             @constraint(model,   λ[1,6,t]               - λ[1,6,t+1] == 0)
#             @NLconstraint(model, λ[1,7,t]               - λ[1,7,t+1] + λ[1,5,t+1]*ΔT*x[8,t]*sin(x[7,t]) - λ[1,6,t+1]*ΔT*x[8,t]*cos(x[7,t]) == 0)
#             @NLconstraint(model, λ[1,8,t]               - λ[1,8,t+1] - λ[1,5,t+1]*ΔT*cos(x[7,t]) - λ[1,6,t+1]*ΔT*sin(x[7,t]) == 0)
#             @NLconstraint(model, λ[1,9,t] + 2*θ[1]*(x[9,t]-x[5,t])              - λ[1,9,t+1] == 0)
#         else
#             @constraint(model,   λ[1,1,t] + 2*θ[2]*x[1,t] == 0)
#             @constraint(model,   λ[1,2,t] == 0)
#             @NLconstraint(model, λ[1,3,t] == 0)
#             @NLconstraint(model, λ[1,4,t] == 0)
#             @constraint(model,   λ[1,5,t] + 2*θ[1]*(x[5,t]-x[9,t]) == 0)
#             @constraint(model,   λ[1,6,t] == 0)
#             @NLconstraint(model, λ[1,7,t] == 0)
#             @NLconstraint(model, λ[1,8,t] == 0)
#             @NLconstraint(model, λ[1,9,t] + 2*θ[1]*(x[9,t]-x[5,t]) == 0)
#         end

#         if t != g.h # dJ2/dx
#             @constraint(model,   λ[2,1,t] + 2*θ[3]*(x[1,t]-x[5,t])                   - λ[2,1,t+1] == 0)
#             @constraint(model,   λ[2,2,t]                   - λ[2,2,t+1] == 0)
#             @NLconstraint(model, λ[2,3,t]                   - λ[2,3,t+1] + λ[2,1,t+1]*ΔT*x[4,t]*sin(x[3,t]) - λ[2,2,t+1]*ΔT*x[4,t]*cos(x[3,t])  == 0)
#             @NLconstraint(model, λ[2,4,t]                   - λ[2,4,t+1] - λ[2,1,t+1]*ΔT*cos(x[3,t]) - λ[2,2,t+1]*ΔT*sin(x[3,t]) == 0)
#             @constraint(model,   λ[2,5,t] + 2*θ[3]*(x[5,t]-x[1,t])                    - λ[2,5,t+1] == 0)
#             @constraint(model,   λ[2,6,t]                   - λ[2,6,t+1] == 0)
#             @NLconstraint(model, λ[2,7,t]                   - λ[2,7,t+1] + λ[2,5,t+1]*ΔT*x[8,t]*sin(x[7,t]) - λ[2,6,t+1]*ΔT*x[8,t]*cos(x[7,t]) == 0)
#             @NLconstraint(model, λ[2,8,t] + 2*θ[4]*(x[8,t]-1)                  - λ[2,8,t+1] - λ[2,5,t+1]*ΔT*cos(x[7,t]) - λ[2,6,t+1]*ΔT*sin(x[7,t]) == 0)
#             @NLconstraint(model, λ[2,9,t]                   -λ[2,9,t+1] == 0)
#         else
#             @constraint(model,   λ[2,1,t] + 2*θ[3]*(x[1,t]-x[5,t]) == 0)
#             @constraint(model,   λ[2,2,t]  == 0)
#             @NLconstraint(model, λ[2,3,t]  == 0)
#             @NLconstraint(model, λ[2,4,t]  == 0)
#             @constraint(model,   λ[2,5,t] + 2*θ[3]*(x[5,t]-x[1,t]) == 0)
#             @constraint(model,   λ[2,6,t]  == 0)
#             @NLconstraint(model, λ[2,7,t]  == 0)
#             @NLconstraint(model, λ[2,8,t] + 2*θ[4]*(x[8,t]-1) == 0)
#             @NLconstraint(model, λ[2,9,t] == 0)
#         end

#         # dJ1/du and dJ2/du
#         @constraint(model, 4*u[1,t] - λ[1,3,t]*ΔT == 0)
#         @constraint(model, 4*u[2,t] - λ[1,4,t]*ΔT == 0)
#         @constraint(model, 4*u[3,t] - λ[2,7,t]*ΔT == 0)
#         @constraint(model, 4*u[4,t] - λ[2,8,t]*ΔT == 0)
#         if t == 1
#             @NLconstraint(model, x[1,1] == x0[1] + ΔT * x0[4]*cos(x0[3]))
#             @NLconstraint(model, x[2,1] == x0[2] + ΔT * x0[4]*sin(x0[3]))
#             @NLconstraint(model, x[3,1] == x0[3] + ΔT * u[1,t])
#             @NLconstraint(model, x[4,1] == x0[4] + ΔT * u[2,t])
#             @NLconstraint(model, x[5,1] == x0[5] + ΔT * x0[8]*cos(x0[7]))
#             @NLconstraint(model, x[6,1] == x0[6] + ΔT * x0[8]*sin(x0[7]))
#             @NLconstraint(model, x[7,1] == x0[7] + ΔT * u[3,t])
#             @NLconstraint(model, x[8,1] == x0[8] + ΔT * u[4,t])
#             @NLconstraint(model, x[9,1] == x0[9])
#         else
#             @NLconstraint(model, x[1,t] == x[1,t-1] + ΔT * x[4,t-1]*cos(x[3,t-1]))
#             @NLconstraint(model, x[2,t] == x[2,t-1] + ΔT * x[4,t-1]*sin(x[3,t-1]))
#             @NLconstraint(model, x[3,t] == x[3,t-1] + ΔT * u[1,t])
#             @NLconstraint(model, x[4,t] == x[4,t-1] + ΔT * u[2,t])
#             @NLconstraint(model, x[5,t] == x[5,t-1] + ΔT * x[8,t-1]*cos(x[7,t-1]))
#             @NLconstraint(model, x[6,t] == x[6,t-1] + ΔT * x[8,t-1]*sin(x[7,t-1]))
#             @NLconstraint(model, x[7,t] == x[7,t-1] + ΔT * u[3,t])
#             @NLconstraint(model, x[8,t] == x[8,t-1] + ΔT * u[4,t])
#             @NLconstraint(model, x[9,t] == x[9,t-1])
#         end
#     end
#     optimize!(model)
#     return value.(x), value.(u), value.(θ), model
# end

# # inv_sol = KKT_highway_inverse_game_solve(obs_x_FB[:,2:end], obs_u_FB, 4*ones(4), x0);

# inv_sol = KKT_highway_inverse_game_solve(obs_x_OL[:,2:end],  4*ones(4), x0, 1:game_horizon-1, 1:nx, 1:nu, true)



# anim1 = @animate for i in 1:game_horizon
#     plot( [inv_sol[1][1,i], inv_sol[1][1,i]], [inv_sol[1][2,i], inv_sol[1][2,i]], markershape = :square, label = "player 1, JuMP", xlims = (-0.5, 1.5), ylims = (0, 6))
#     plot!([inv_sol[1][5,i], inv_sol[1][5,i]], [inv_sol[1][6,i], inv_sol[1][6,i]], markershape = :square, label = "player 2, JuMP", xlims = (-0.5, 1.5), ylims = (0, 6))
#     plot!([0], seriestype = "vline", color = "black", label = "")
#     plot!([1], seriestype = "vline", color = "black", label = "") 
# end
# gif(anim1, "lane_guiding_inv_JuMP.gif", fps = 10)

# anim2 = @animate for i in 1:game_horizon
#     plot( [obs_x_OL[1,i], obs_x_OL[1,i]], [obs_x_OL[2,i], obs_x_OL[2,i]], markershape = :square, label = "player 1, iLQ OLNE", xlims = (-0.5, 1.5), ylims = (0, 6))
#     plot!([obs_x_OL[5,i], obs_x_OL[5,i]], [obs_x_OL[6,i], obs_x_OL[6,i]], markershape = :square, label = "player 2, iLQ OLNE", xlims = (-0.5, 1.5), ylims = (0, 6))    
#     plot!([0], seriestype = "vline", color = "black", label = "")
#     plot!([1], seriestype = "vline", color = "black", label = "")
# end
# gif(anim2, "lane_guiding_OL_iLQ.gif", fps = 10)




# 1. generate noisy observation, 10 for each noise level. If 10 noise level, then we generate 10x10 observations ✓
# 2. run inverse_KKT for every observation and record the state loss

num_clean_traj = 1
x0_set = [x0 for ii in 1:num_clean_traj]
expert_traj_list, c_expert = generate_expert_traj(g, solver2, x0_set, num_clean_traj)
if sum([c_expert[ii]==false for ii in 1:length(c_expert)]) >0
    @warn "regenerate expert demonstrations because some of the expert demonstration not converged!!!"
end

game = g
solver = solver2

# The below: generate random expert trajectories
num_obs = 6
# noise_level_list = 0.005:0.005:0.05
noise_level_list = 0.004:0.004:0.04
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

θ₀ = 5*ones(4);
inv_traj_x_list = [[[] for jj in 1:num_obs] for ii in 1:length(noise_level_list)];
inv_traj_u_list = [[[] for jj in 1:num_obs] for ii in 1:length(noise_level_list)];
inv_sol_list = [[[] for jj in 1:num_obs] for ii in 1:length(noise_level_list)];
inv_loss_list = [[[] for jj in 1:num_obs] for ii in 1:length(noise_level_list)];
inv_mean_generalization_loss_list = [[[] for jj in 1:num_obs] for ii in 1:length(noise_level_list)];
inv_var_generalization_loss_list = [[[] for jj in 1:num_obs] for ii in 1:length(noise_level_list)];
inv_model_list = [[[] for jj in 1:num_obs] for ii in 1:length(noise_level_list)];
inv_ground_truth_loss_list = [[[] for jj in 1:num_obs] for ii in 1:length(noise_level_list)];
inv_ground_truth_computed_traj_list = [[[] for jj in 1:num_obs] for ii in 1:length(noise_level_list)];

num_test = 6
test_noise_level=1.0
test_x0_set = [x0 - [zeros(12);x0[13]] + test_noise_level*[zeros(12);rand(1)[1]] for ii in 1:num_test];
test_expert_traj_list, c_test_expert = generate_expert_traj(game, solver, test_x0_set, num_test);


# obs_time_list= [1,2,3,4,5,6,11,12,13,14,15,16,21,22,23,24,25,26,31,32,33,34,35,36]
obs_time_list = [1,2,3,4,5,6,7,8,9,10,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39]
obs_state_list = [1,2,3,5,6,7, 9, 10, 11]
obs_control_list=[]
# obs_time_list = 1:game_horizon-1
# obs_state_list = 1:nx
# obs_control_list = 1:nu
index = 1
Threads.@threads for noise in 1:length(noise_level_list)
    for ii in 1:num_obs
        tmp_expert_traj_x = noisy_expert_traj_list[index][noise][ii].x
        tmp_expert_traj_u = noisy_expert_traj_list[index][noise][ii].u
        
        tmp_obs_x = transpose(mapreduce(permutedims, vcat, Vector([Vector(tmp_expert_traj_x[t]) for t in 1:game.h])))
        # tmp_obs_u = transpose(mapreduce(permutedims, vcat, Vector([Vector(tmp_expert_traj_u[t]) for t in 1:game.h])))
        # tmp_inv_traj_x, tmp_inv_traj_u, tmp_inv_sol, tmp_inv_model = KKT_highway_inverse_game_solve(tmp_obs_x[:,2:end], tmp_obs_u, θ₀, x0_set[index], obs_time_list, obs_state_list, obs_control_list)
        tmp_sol = two_level_inv_KKT(tmp_obs_x, θ₀, obs_time_list, obs_state_list)
        tmp_inv_traj_x, tmp_inv_traj_u, tmp_inv_sol, tmp_inv_model = tmp_sol[1], tmp_sol[2], tmp_sol[3], tmp_sol[4];
        tmp_inv_loss = objective_value(tmp_inv_model)
        println("The $(ii)-th observation of $(noise)-th noise level")
        # solution_summary(tmp_inv_model)
        tmp_ground_truth_loss_value, tmp_ground_truth_computed_traj, _, _=loss(tmp_inv_sol, iLQGames.dynamics(game), "FBNE_costate", expert_traj_list[index], false, false, [], [], 1:game_horizon-1, 1:12, 1:nu, false) 
        # @infiltrate
        # tmp_test_sol = [[] for jj in 1:num_test]
        tmp_test_loss_value = zeros(num_test)
        for jj in 1:num_test
            # @infiltrate
            tmp_test_loss_value[jj], _,_,_ = loss(tmp_inv_sol, iLQGames.dynamics(game), "FBNE_costate", test_expert_traj_list[jj], false, false, [],[],1:game_horizon-1, 1:nx, 1:nu,false)
        end
        # @infiltrate
        push!(inv_mean_generalization_loss_list[noise][ii], mean(tmp_test_loss_value))
        # println("$(inv_mean_generalization_loss_list[noise][ii])")
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

jldsave("KKT_inverse_x0_full$(Dates.now())"; inv_traj_x_list, inv_traj_u_list, inv_sol_list, inv_loss_list, inv_mean_generalization_loss_list, inv_var_generalization_loss_list, inv_model_list, inv_ground_truth_loss_list,
    inv_ground_truth_computed_traj_list, obs_time_list, obs_state_list, obs_control_list, num_test, test_x0_set, 
    test_expert_traj_list, c_test_expert, noise_level_list, expert_traj_list, dynamics, nx, nu, game_horizon, g, solver1, costs,)


jldsave("KKT_x0_full_20_ill$(Dates.now())"; game_horizon, inv_mean_generalization_loss_list, inv_var_generalization_loss_list, inv_sol_list,
    inv_loss_list,  inv_ground_truth_loss_list,
    obs_time_list, obs_state_list)

jldsave("1008_baobei_KKT_x0_full_20_ill$(Dates.now())"; game_horizon, inv_mean_generalization_loss_list, inv_var_generalization_loss_list, inv_sol_list,
    inv_loss_list,  inv_ground_truth_loss_list,inv_traj_x_list, inv_traj_u_list,
    obs_time_list, obs_state_list, test_noise_level, x0, noise_level_list, num_test, test_expert_traj_list, expert_traj_list,
    obs_x_OL, obs_x_FB)


jldsave("1008_baobei_KKT_x0_partial_20_ill$(Dates.now())"; game_horizon, inv_mean_generalization_loss_list, inv_var_generalization_loss_list, inv_sol_list,
    inv_loss_list,  inv_ground_truth_loss_list,inv_traj_x_list, inv_traj_u_list,
    obs_time_list, obs_state_list, test_noise_level, x0, noise_level_list, num_test, test_expert_traj_list, expert_traj_list,
    obs_x_OL, obs_x_FB)





jldsave("KKT_inverse_$(Dates.now())"; inv_traj_x_list, inv_traj_u_list, inv_sol_list, inv_loss_list, inv_mean_generalization_loss_list, inv_var_generalization_loss_list, inv_model_list, inv_ground_truth_loss_list,
    inv_ground_truth_computed_traj_list, obs_time_list, obs_state_list, obs_control_list, num_test, test_x0_set, 
    test_expert_traj_list, c_test_expert, noise_level_list, expert_traj_list, KKT_highway_forward_game_solve, KKT_highway_inverse_game_solve, dynamics, nx, nu, game_horizon, g, solver1, costs,)

jldsave("KKT_inverse_compact_20_no_control_partial$(Dates.now())"; inv_traj_x_list, inv_traj_u_list, inv_sol_list, inv_loss_list, inv_mean_generalization_loss_list, inv_var_generalization_loss_list, inv_ground_truth_loss_list,
    inv_ground_truth_computed_traj_list, obs_time_list, obs_state_list, obs_control_list, num_test, test_x0_set, 
    test_expert_traj_list, c_test_expert, noise_level_list, expert_traj_list, dynamics, nx, nu, game_horizon, g, solver1, costs)
# for ii in 1:num_clean_traj
#     for jj in 1:num_noise_level
#         conv_table,sol_table,loss_table,grad_table,equi_table,iter_table,ground_truth_loss = run_experiment(game,θ₀,[x0_set[ii] for kk in 1:num_obs], 
#                                                                                                 noisy_expert_traj_list[ii][jj], parameterized_cost, GD_iter_num, 20, 1e-8, 
#                                                                                                 1:game_horizon-1,1:nx, 1:nu, "FBNE_costate", 0.00000000001, false, 10.0, expert_traj_list[ii])
#         θ_list, index_list, optim_loss_list = get_the_best_possible_reward_estimate_single([x0_set[ii] for kk in 1:num_obs], ["FBNE_costate","FBNE_costate"], sol_table, loss_table, equi_table)
#         # generalization_error = generalization_loss(games[ii], θ_list[1], [x0+0.5*(rand(4)-0.5*ones(4)) for ii in 1:num_generalization], 
#         #                             expert_traj_list, parameterized_cost, equilibrium_type_list) #problem
#         push!(θ_list_list[ii][jj], θ_list)
#         push!(optim_loss_list_list[ii][jj], optim_loss_list)
#         push!(ground_truth_loss_list[ii][jj], ground_truth_loss)
#         push!(generalization_error_list[ii][jj], generalization_error)
#     end
# end





## Plot state_prediction_loss vs. noise_variance level
inv_mean_generalization_loss_list=t1["inv_mean_generalization_loss_list"]
inv_loss_list = t1["inv_loss_list"]
inv_ground_truth_loss_list = t1["inv_ground_truth_loss_list"]

var1=[var(inv_mean_generalization_loss_list[ii][jj][1] for jj in 1:num_obs)[1] for ii in 1:num_noise_level]
var2=[var(inv_loss_list[ii][jj][1] for jj in 1:num_obs)[1] for ii in 1:num_noise_level]
var3=[var(inv_ground_truth_loss_list[ii][jj][1] for jj in 1:num_obs)[1] for ii in 1:num_noise_level]

plot(noise_level_list, [mean(inv_mean_generalization_loss_list[ii][jj][1] for jj in 1:num_obs)[1] for ii in 1:num_noise_level], label="generalization error")
plot!(noise_level_list, [mean(inv_loss_list[ii][jj][1] for jj in 1:num_obs)[1] for ii in 1:num_noise_level], label = "loss")
plot!(noise_level_list, [mean(inv_ground_truth_loss_list[ii][jj][1] for jj in 1:num_obs)[1] for ii in 1:num_noise_level], label="ground truth")



t1=load("KKT_dubins_0.5_gen_x0_compact")
noise_level_list = 0.004:0.002:0.04
num_obs=10
num_noise_level = length(noise_level_list)
inv_mean_generalization_loss_list = t1["inv_mean_generalization_loss_list"]
inv_loss_list = t1["inv_loss_list"]
inv_ground_truth_loss_list = t1["inv_ground_truth_loss_list"]

var1 = [var(inv_mean_generalization_loss_list[ii][jj][1] for jj in 1:num_obs)[1] for ii in 1:num_noise_level]
var2 = [var(inv_loss_list[ii][jj][1] for jj in 1:num_obs)[1] for ii in 1:num_noise_level]
var3 = [var(inv_ground_truth_loss_list[ii][jj][1] for jj in 1:num_obs)[1] for ii in 1:num_noise_level]

plot(noise_level_list, [mean(inv_mean_generalization_loss_list[ii][jj][1] for jj in 1:num_obs)[1] for ii in 1:num_noise_level], label="generalization error")
plot!(noise_level_list, [mean(inv_loss_list[ii][jj][1] for jj in 1:num_obs)[1] for ii in 1:num_noise_level], ribbons=(var2,var2), label = "loss")
plot!(noise_level_list, [mean(inv_ground_truth_loss_list[ii][jj][1] for jj in 1:num_obs)[1] for ii in 1:num_noise_level], ribbons=(var3,var3), label="ground truth")


t1=load("KKT_partial_0.3")
noise_level_list = 0.004:0.004:0.04
num_obs=10
num_noise_level = length(noise_level_list)
inv_mean_generalization_loss_list = t1["inv_mean_generalization_loss_list"]
inv_loss_list = t1["inv_loss_list"]
inv_ground_truth_loss_list = t1["inv_ground_truth_loss_list"]

var1 = [var(inv_mean_generalization_loss_list[ii][jj][1] for jj in 1:num_obs)[1] for ii in 1:num_noise_level]
var2 = [var(inv_loss_list[ii][jj][1] for jj in 1:num_obs)[1] for ii in 1:num_noise_level]
var3 = [var(inv_ground_truth_loss_list[ii][jj][1] for jj in 1:num_obs)[1] for ii in 1:num_noise_level]

plot(noise_level_list, [mean(inv_mean_generalization_loss_list[ii][jj][1] for jj in 1:num_obs)[1] for ii in 1:num_noise_level],ribbons=(var1,var1), label="generalization error", xlabel="noise level")
plot!(noise_level_list, [mean(inv_loss_list[ii][jj][1] for jj in 1:num_obs)[1] for ii in 1:num_noise_level], ribbons=(var2,var2), label = "loss")
plot!(noise_level_list, [mean(inv_ground_truth_loss_list[ii][jj][1] for jj in 1:num_obs)[1] for ii in 1:num_noise_level], ribbons=(var3,var3), label="ground truth error")


# below is to plot the predicted trajectory from KKT baseline
noise=1
ii=1
tt=loss(t1["inv_sol_list"][noise][ii][1], dynamics, "FBNE_costate", expert_traj2, false )
plt = plot([tt[2].x[t][1] for t in 1:39], [tt[2].x[t][2] for t in 1:39], color="blue")
plt = plot!([tt[2].x[t][5] for t in 1:39], [tt[2].x[t][6] for t in 1:39], color="red")
plt = plot!([obs_x_FB[1,t] for t in 1:39], [obs_x_FB[2,t] for t in 1:39], color="blue", linestyle=:dash, label="ground truth")
plt = plot!([obs_x_FB[5,t] for t in 1:39], [obs_x_FB[6,t] for t in 1:39], color = "red", linestyle=:dash, label="ground truth")


ttt = t1["inv_traj_x_list"][1][1][1]
plt = plot([ttt[1,t] for t in 1:39], [ttt[2,t] for t in 1:39], color="blue",linestyle=:dash, linewidth=3,label="OLNE under the cost learned by KKT",title="trajectories comparison")
plt = plot!([ttt[5,t] for t in 1:39], [ttt[6,t] for t in 1:39], color="red",linestyle=:dash,linewidth=3, label="OLNE under the cost learned by KKT")
plt = plot!([tt[2].x[t][1] for t in 1:39], [tt[2].x[t][2] for t in 1:39], color="blue", linestyle=:dot,linewidth=3, label="FBNE under the cost learned by KKT")
plt = plot!([tt[2].x[t][5] for t in 1:39], [tt[2].x[t][6] for t in 1:39], color="red",linestyle=:dot,linewidth=3, label = "FBNE under the cost learned by kKT")
plt = plot!([obs_x_FB[1,t] for t in 1:39], [obs_x_FB[2,t] for t in 1:39], color="blue", linewidth=3, label="ground truth FBNE data")
plt = plot!([obs_x_FB[5,t] for t in 1:39], [obs_x_FB[6,t] for t in 1:39], color = "red", linewidth=3, label="ground truth FBNE data")
savefig("traj_compare_OLNE.pdf")



# Oct. 5th
t1=load("KKT_partial_2cars_x0_baobei")













