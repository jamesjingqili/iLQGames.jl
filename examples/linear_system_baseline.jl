using iLQGames
import iLQGames: dx
import BenchmarkTools
using Plots
using ForwardDiff
using iLQGames:
    SystemTrajectory
using Infiltrator
using Optim
using LinearAlgebra
using Distributed
include("diff_solver.jl")
include("inverse_game_solver.jl")
include("experiment_utils.jl") # NOTICE!! Many functions are defined there.

# parametes: number of states, number of inputs, sampling time, horizon
nx, nu, ΔT, game_horizon = 4, 4, 0.1, 40
# setup the dynamics
struct LinearSystem <: ControlSystem{ΔT,nx,nu} end
# state: (px, py, phi, v)
dx(cs::LinearSystem, x, u, t) = SVector(u[1],u[2],u[3],u[4])
dynamics = LinearSystem()
# costs = (FunctionPlayerCost((g, x, u, t) -> (10*(x[1]-1)^2 + 0.1*(x[3]-pi/2)^2 + (x[4]-1)^2 + u[1]^2 + u[2]^2 - 0.1*((x[1]-x[5])^2 + (x[2]-x[6])^2))),
         # FunctionPlayerCost((g, x, u, t) -> ((x[5]-1)^2 + 0.1*(x[7]-pi/2)^2 + (x[8]-1)^2 + u[3]^2 + u[4]^2- 0.1*((x[1]-x[5])^2 + (x[2]-x[6])^2))))
costs = (FunctionPlayerCost((g, x, u, t) -> ( 2*(x[3])^2 + 2*(x[4])^2 + u[1]^2 + u[2]^2)),
         FunctionPlayerCost((g, x, u, t) -> ( 2*(x[1]-x[3])^2 + 2*(x[2]-x[4])^2 + u[3]^2 + u[4]^2)))
# indices of inputs that each player controls
player_inputs = (SVector(1,2), SVector(3,4))
# the horizon of the game
g = GeneralGame(game_horizon, player_inputs, dynamics, costs)
# get a solver, choose initial conditions and solve (in about 9 ms with AD)
solver1 = iLQSolver(g, max_scale_backtrack=10, max_elwise_diff_step=Inf, equilibrium_type="OLNE_costate")
x0 = SVector(0, 1, 1,1)
c1, expert_traj1, strategies1 = solve(g, solver1, x0)

x1_OL, y1_OL = [expert_traj1.x[i][1] for i in 1:game_horizon], [expert_traj1.x[i][2] for i in 1:game_horizon];
x2_OL, y2_OL = [expert_traj1.x[i][3] for i in 1:game_horizon], [expert_traj1.x[i][4] for i in 1:game_horizon];
anim1 = @animate for i in 1:game_horizon
    plot([x1_OL[i], x1_OL[i]], [y1_OL[i], y1_OL[i]], markershape = :square, label = "player 1, OL", xlims = (-1, 2), ylims = (-1, 2))
    plot!([x2_OL[i], x2_OL[i]], [y2_OL[i], y2_OL[i]], markershape = :square, label = "player 2, OL", xlims = (-1, 2), ylims = (-1, 2))
    plot!([0], seriestype = "vline", color = "black", label = "")
    plot!([1], seriestype = "vline", color = "black", label = "") 
end
gif(anim1, "LQ_OL.gif", fps = 10)


# get a solver, choose initial conditions and solve (in about 9 ms with AD)
solver2 = iLQSolver(g, max_scale_backtrack=5, max_elwise_diff_step=Inf, equilibrium_type="FBNE_costate")
c2, expert_traj2, strategies2 = solve(g, solver2, x0)

x1_FB, y1_FB = [expert_traj2.x[i][1] for i in 1:game_horizon], [expert_traj2.x[i][2] for i in 1:game_horizon];
x2_FB, y2_FB = [expert_traj2.x[i][3] for i in 1:game_horizon], [expert_traj2.x[i][4] for i in 1:game_horizon];
anim2 = @animate for i in 1:game_horizon
    plot([x1_FB[i], x1_FB[i]], [y1_FB[i], y1_FB[i]], markershape = :square, label = "player 1, FB", xlims = (-1, 2), ylims = (-1, 2))
    plot!([x2_FB[i], x2_FB[i]], [y2_FB[i], y2_FB[i]], markershape = :square, label = "player 2, FB", xlims = (-1, 2), ylims = (-1, 2))    
    plot!([0], seriestype = "vline", color = "black", label = "")
    plot!([1], seriestype = "vline", color = "black", label = "")
end

gif(anim2, "LQ_FB.gif", fps = 10)

#-----------------------------------------------------------------------------------------------------------------------------------

function parameterized_cost(θ::Vector)
    costs = (FunctionPlayerCost((g, x, u, t) -> ( θ[7]*x[1]^2 + θ[8]*x[2]^2 + θ[1]*x[3]^2 + θ[2]*x[4]^2 + θ[3]*(u[1]^2 + u[2]^2))),
             FunctionPlayerCost((g, x, u, t) -> ( θ[4]*(x[1]-x[3])^2 + θ[5]*(x[2]-x[4])^2 + θ[6]*(u[3]^2 + u[4]^2))))
    return costs
end

#-----------------------------------------------------------------------------------------------------------------------------------

# max_GD_iteration_num =200

# # θ = [9.0; 1.0; 2.0;1.0]
# θ = ones(8)
# θ_dim = length(θ)
# sol = [zeros(θ_dim) for iter in 1:max_GD_iteration_num+1]
# sol[1] = θ
# loss_values = zeros(max_GD_iteration_num+1)
# loss_values[1],_,_ = inverse_game_loss(sol[1], g, expert_traj1, x0, parameterized_cost, "OLNE_costate")
# gradient = [zeros(θ_dim) for iter in 1:max_GD_iteration_num]
# equilibrium_type = ["" for iter in 1:max_GD_iteration_num]
# traj_list = [zero(SystemTrajectory, g) for iter in 1:max_GD_iteration_num]
# solver_list = [[] for iter in 1:max_GD_iteration_num]
# for iter in 1:max_GD_iteration_num
#     sol[iter+1], loss_values[iter+1], gradient[iter], equilibrium_type[iter], _, _ = inverse_game_gradient_descent(sol[iter], 
#                                             g, expert_traj1, x0, 10, parameterized_cost, [],true)
#     println("Current solution: ", sol[iter+1])
#     if loss_values[iter+1]<0.1
#         break
#     end
# end

#----------------------------------------------------------------------------------------------------------------------------------
# max_GD_iteration_num = 40

θ_true = [2.0;2.0;1.0;2.0;2.0;1.0;0.0;0.0]

θ₀ = ones(8)

x0_set = [x0+0.1*rand(Normal(0,1),4) for ii in 1:2]
c_expert,expert_traj_list,expert_equi_list=generate_traj(g,θ_true,x0_set,parameterized_cost,["FBNE_costate","OLNE_costate"])

run_experiments_with_baselines(g, θ₀, x0_set, expert_traj_list, parameterized_cost, 50)


conv, rew, loss_lists, grad, equi, _ = test_experiments(g, θ₀, [x0], [expert_traj1], parameterized_cost,200)
iterations_FB = sum(equi[1][1][ii]!="" for ii in 1:length(equi[1][1]))
iterations_OL = sum(equi[2][1][ii]!="" for ii in 1:length(equi[2][1]))
iterations_BA = sum(equi[3][1][ii]!="" for ii in 1:length(equi[3][1]))



# 1. test robustness to observation noise
#  X: 
# Y1: state prediction loss. Code: loss(θ, equilibrium_type, expert_traj, false)
# Y2: generalization loss.   Code: generalization_loss()


num_test = 20
sampled_initial_states = [x0+[0.1;0.1;0.1;0.1; 0.1;0.1;0.1;0.1].*rand(Normal(0,1),nx) for ii in 1:num_test]
recorded_loss

for item in 1:num_test


end

# generalization to unseen initial state




# average computation time: FB, OL, Joint




