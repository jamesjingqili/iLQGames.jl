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
using Dates
include("../src/diff_solver.jl")
include("../src/inverse_game_solver.jl")
include("../src/experiment_utils.jl") # NOTICE!! Many functions are defined there.

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

#----------------------------------------------------------------------------------------------------------------------------------

include("../src/experiment_utils.jl") # NOTICE!! Many functions are defined there.


GD_iter_num = 200
n_data = 100
θ_true = [2.0;2.0;1.0;2.0;2.0;1.0;0.0;0.0]

θ₀ = ones(8)
# 
x0_set = [x0+rand(4)-0.5*ones(4) for ii in 1:n_data]
c_expert,expert_traj_list,expert_equi_list=generate_traj(g,θ_true,x0_set,parameterized_cost,["FBNE_costate","OLNE_costate"])


conv_table, sol_table, loss_table, grad_table, equi_table, iter_table,comp_time_table=run_experiments_with_baselines(g, θ₀, x0_set, expert_traj_list, 
                                                                                                                        parameterized_cost, GD_iter_num)


θ_list, index_list, optim_loss_list = get_the_best_possible_reward_estimate(x0_set, ["FBNE_costate","OLNE_costate"], sol_table, loss_table, equi_table)


iterations_BA,iterations_FB,iteration_OL=iterations_taken_to_converge(equi[1][1]),iterations_taken_to_converge(equi[2][1]),iterations_taken_to_converge(equi[3][1])



jldsave("LQ_data_$(Dates.now())"; nx, nu, ΔT, g,dynamics, costs, player_inputs, solver1, x0, c1, expert_traj1, strategies1, 
    solver2, c2, expert_traj2, strategies2, parameterized_cost, GD_iter_num, n_data, θ_true, θ₀, 
    c_expert, expert_traj_list, expert_equi_list, conv_table, sol_table, loss_table, grad_table, 
    equi_table, iter_table, comp_time_table, θ_list, index_list, optim_loss_list)

"Experiment 1: Without noitse. Histogram"
#  X: number of cases
# Y1: state prediction loss. Code: loss(θ, equilibrium_type, expert_traj, false)
# Y2: generalization loss.   Code: generalization_loss()
# Y3: computation time/GD iterations taken to converge.



"Experiment 2: With noise. Scatter plot"
# X: noise variance
# Y1: state prediction loss, mean and variance
# Y2: generalization loss, mean and variance



# generalization to unseen initial state




# average computation time: FB, OL, Joint



"Accelerated GD?"
