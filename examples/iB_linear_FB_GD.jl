using Distributed
@everywhere begin
    using Pkg
    Pkg.activate("../")
    Pkg.instantiate
    using iLQGames
    import iLQGames: dx
    import BenchmarkTools
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
    include("../src/diff_solver.jl")
    include("../src/inverse_game_solver.jl")
    include("../src/experiment_utils.jl") # NOTICE!! Many functions are defined there.

end


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
using Statistics
include("../src/diff_solver.jl")
include("../src/inverse_game_solver.jl")
include("../src/experiment_utils.jl") # NOTICE!! Many functions are defined there.

@everywhere begin
# parametes: number of states, number of inputs, sampling time, horizon
nx, nu, ΔT, game_horizon = 4, 4, 0.1, 100
# setup the dynamics
struct LinearSystem <: ControlSystem{ΔT,nx,nu} end
# state: (px, py, phi, v)
dx(cs::LinearSystem, x, u, t) = SVector(u[1],u[2],u[3],u[4])

dynamics = LinearSystem()

# costs = (FunctionPlayerCost((g, x, u, t) -> (10*(x[1]-1)^2 + 0.1*(x[3]-pi/2)^2 + (x[4]-1)^2 + u[1]^2 + u[2]^2 - 0.1*((x[1]-x[5])^2 + (x[2]-x[6])^2))),
         # FunctionPlayerCost((g, x, u, t) -> ((x[5]-1)^2 + 0.1*(x[7]-pi/2)^2 + (x[8]-1)^2 + u[3]^2 + u[4]^2- 0.1*((x[1]-x[5])^2 + (x[2]-x[6])^2))))
costs = (FunctionPlayerCost((g, x, u, t) -> ( 2*(x[3])^2 + 2*(x[4])^2 + u[1]^2 + u[2]^2)),
         FunctionPlayerCost((g, x, u, t) -> ( 2*(x[1]-x[3])^2 + 2*(x[2]-x[4])^2 + u[3]^2 + u[4]^2)))
# costs = (FunctionPlayerCost((g, x, u, t) -> ( 2*(x[1]-2*x[3])^2 + 2*(x[2]-2*x[4])^2 + u[1]^2 + u[2]^2)),
#          FunctionPlayerCost((g, x, u, t) -> ( 2*(x[1]-x[3])^2 + 2*(x[2]-x[4])^2 + u[3]^2 + u[4]^2)))


# indices of inputs that each player controls
player_inputs = (SVector(1,2), SVector(3,4))
# the horizon of the game
g = GeneralGame(game_horizon, player_inputs, dynamics, costs)
# get a solver, choose initial conditions and solve (in about 9 ms with AD)
solver1 = iLQSolver(g, max_scale_backtrack=10, max_elwise_diff_step=Inf, equilibrium_type="OLNE_costate")
x0 = SVector(0, 1, 1,1)
c1, expert_traj1, strategies1 = solve(g, solver1, x0)
# get a solver, choose initial conditions and solve (in about 9 ms with AD)
solver2 = iLQSolver(g, max_scale_backtrack=5, max_elwise_diff_step=Inf, equilibrium_type="FBNE_costate")
c2, expert_traj2, strategies2 = solve(g, solver2, x0)

end

x1_OL, y1_OL = [expert_traj1.x[i][1] for i in 1:game_horizon], [expert_traj1.x[i][2] for i in 1:game_horizon];
x2_OL, y2_OL = [expert_traj1.x[i][3] for i in 1:game_horizon], [expert_traj1.x[i][4] for i in 1:game_horizon];
anim1 = @animate for i in 1:game_horizon
    plot([x1_OL[i], x1_OL[i]], [y1_OL[i], y1_OL[i]], markershape = :square, label = "player 1, OL", xlims = (-1, 2), ylims = (-1, 2))
    plot!([x2_OL[i], x2_OL[i]], [y2_OL[i], y2_OL[i]], markershape = :square, label = "player 2, OL", xlims = (-1, 2), ylims = (-1, 2))
    plot!([0], seriestype = "vline", color = "black", label = "")
    plot!([1], seriestype = "vline", color = "black", label = "") 
end
gif(anim1, "LQ_OL_cycle.gif", fps = 10)




x1_FB, y1_FB = [expert_traj2.x[i][1] for i in 1:game_horizon], [expert_traj2.x[i][2] for i in 1:game_horizon];
x2_FB, y2_FB = [expert_traj2.x[i][3] for i in 1:game_horizon], [expert_traj2.x[i][4] for i in 1:game_horizon];
anim2 = @animate for i in 1:game_horizon
    plot([x1_FB[i], x1_FB[i]], [y1_FB[i], y1_FB[i]], markershape = :square, label = "player 1, FB", xlims = (-1, 2), ylims = (-1, 2))
    plot!([x2_FB[i], x2_FB[i]], [y2_FB[i], y2_FB[i]], markershape = :square, label = "player 2, FB", xlims = (-1, 2), ylims = (-1, 2))    
    plot!([0], seriestype = "vline", color = "black", label = "")
    plot!([1], seriestype = "vline", color = "black", label = "")
end

gif(anim2, "LQ_FB_cycle.gif", fps = 10)


function plot_traj(expert_traj1, expert_traj2)
    x1_OL, y1_OL = [expert_traj1.x[i][1] for i in 1:game_horizon], [expert_traj1.x[i][2] for i in 1:game_horizon];
    x2_OL, y2_OL = [expert_traj1.x[i][3] for i in 1:game_horizon], [expert_traj1.x[i][4] for i in 1:game_horizon];
    anim1 = @animate for i in 1:game_horizon
    plot([x1_OL[i], x1_OL[i]], [y1_OL[i], y1_OL[i]], markershape = :square, label = "player 1, OL", xlims = (-1, 2), ylims = (-1, 2))
    plot!([x2_OL[i], x2_OL[i]], [y2_OL[i], y2_OL[i]], markershape = :square, label = "player 2, OL", xlims = (-1, 2), ylims = (-1, 2))
    plot!([0], seriestype = "vline", color = "black", label = "")
    plot!([1], seriestype = "vline", color = "black", label = "") 
    end
    gif(anim1, "LQ_OL_cycle_test.gif", fps = 10)

    x1_FB, y1_FB = [expert_traj2.x[i][1] for i in 1:game_horizon], [expert_traj2.x[i][2] for i in 1:game_horizon];
    x2_FB, y2_FB = [expert_traj2.x[i][3] for i in 1:game_horizon], [expert_traj2.x[i][4] for i in 1:game_horizon];
    anim2 = @animate for i in 1:game_horizon
    plot([x1_FB[i], x1_FB[i]], [y1_FB[i], y1_FB[i]], markershape = :square, label = "player 1, FB", xlims = (-1, 2), ylims = (-1, 2))
    plot!([x2_FB[i], x2_FB[i]], [y2_FB[i], y2_FB[i]], markershape = :square, label = "player 2, FB", xlims = (-1, 2), ylims = (-1, 2))    
    plot!([0], seriestype = "vline", color = "black", label = "")
    plot!([1], seriestype = "vline", color = "black", label = "")
    end

    gif(anim2, "LQ_FB_cycle_test.gif", fps = 10)
end

function scatter_plot_traj(expert_traj1, traj1, traj2)
    x1_OL, y1_OL = [expert_traj1.x[i][1] for i in 1:game_horizon], [expert_traj1.x[i][2] for i in 1:game_horizon];
    x2_OL, y2_OL = [expert_traj1.x[i][3] for i in 1:game_horizon], [expert_traj1.x[i][4] for i in 1:game_horizon];
    plot(1:length(x1_OL), x1_OL, color=:red, label="x1, OL, expert", linewidth = 5)
    plot!(1:length(y1_OL), y1_OL, color=:blue, label="y1, OL, expert", linewidth = 5)
    plot!(1:length(x2_OL), x2_OL, color=:orange, label="x2, OL, expert", linewidth = 5)
    plot!(1:length(y2_OL), y2_OL, color=:green, label="x1, OL, expert", linewidth = 5)
    x1_OL, y1_OL = [traj1.x[i][1] for i in 1:game_horizon], [traj1.x[i][2] for i in 1:game_horizon];
    x2_OL, y2_OL = [traj1.x[i][3] for i in 1:game_horizon], [traj1.x[i][4] for i in 1:game_horizon];
    plot!(1:length(x1_OL), x1_OL, color=:red, linestyle = :dot, label="x1, OL, inferred", linewidth = 5)
    plot!(1:length(y1_OL), y1_OL, color=:blue, linestyle = :dot, label="y1, OL, inferred", linewidth = 5)
    plot!(1:length(x2_OL), x2_OL, color=:orange, linestyle = :dot, label="x2, OL, inferred", linewidth = 5)
    plot!(1:length(y2_OL), y2_OL, color=:green, linestyle = :dot, label="x1, OL, inferred", linewidth = 5)

    # savefig("LQ_scatter_plot_OL.png")

    # x1_FB, y1_FB = [expert_traj2.x[i][1] for i in 1:game_horizon], [expert_traj2.x[i][2] for i in 1:game_horizon];
    # x2_FB, y2_FB = [expert_traj2.x[i][3] for i in 1:game_horizon], [expert_traj2.x[i][4] for i in 1:game_horizon];
    # plot(1:length(x1_FB), x1_FB, label="x1, FB, expert")
    # plot!(1:length(y1_FB), y1_FB, label="y1, FB, expert")
    # plot!(1:length(x2_FB), x2_FB, label="x2, FB, expert")
    # plot!(1:length(y2_FB), y2_FB, label="x1, FB, expert")
    x1_FB, y1_FB = [traj2.x[i][1] for i in 1:game_horizon], [traj2.x[i][2] for i in 1:game_horizon];
    x2_FB, y2_FB = [traj2.x[i][3] for i in 1:game_horizon], [traj2.x[i][4] for i in 1:game_horizon];
    plot!(1:length(x1_FB), x1_FB, color=:red, linestyle = :dash, label="x1, FB, inferred", linewidth = 5)
    plot!(1:length(y1_FB), y1_FB, color=:blue, linestyle = :dash, label="y1, FB, inferred", linewidth = 5)
    plot!(1:length(x2_FB), x2_FB, color=:orange, linestyle = :dash, label="x2, FB, inferred", linewidth = 5)
    plot!(1:length(y2_FB), y2_FB, color=:green, linestyle = :dash, label="x1, FB, inferred", linewidth = 5)
    plot!(size=(1200,800))
    savefig("LQ_scatter_plot.pdf")
end

#-----------------------------------------------------------------------------------------------------------------------------------
@everywhere begin
function parameterized_cost(θ::Vector)
    costs = (FunctionPlayerCost((g, x, u, t) -> ( θ[1]*(x[1]^2 + x[2]^2) + θ[2]*(x[3]^2 + x[4]^2) + (u[1]^2 + u[2]^2))),
             FunctionPlayerCost((g, x, u, t) -> ( 0*(x[3]^2 + x[4]^2) + θ[3]*((x[1]-x[3])^2 + (x[2]-x[4])^2) + (u[3]^2 + u[4]^2))))
    return costs
end

# function parameterized_cost(θ::Vector)
#     costs = (FunctionPlayerCost((g, x, u, t) -> ( θ[1]*(x[1]-2*x[3])^2 + θ[2]*(x[2]-2*x[4])^2 + θ[3]*(u[1]^2 + u[2]^2))),
#             FunctionPlayerCost((g, x, u, t) -> ( θ[4]*((x[1]-x[3])^2 + (x[2]-x[4])^2) + θ[5]*(u[3]^2 + u[4]^2))))
# end


end

#----------------------------------------------------------------------------------------------------------------------------------








#----------------------------------------------------------------------------------------------------------------------------------

"Experiment 2: With noise. Scatter plot"
# X: noise variance
# Y1: state prediction loss, mean and variance
# Y2: generalization loss, mean and variance

include("experiment_utils.jl")

@everywhere begin

function parameterized_cost(θ::Vector)
    costs = (FunctionPlayerCost((g, x, u, t) -> ( θ[1]*(x[1]^2 + x[2]^2) + θ[2]*(x[3]^2 + x[4]^2) + (u[1]^2 + u[2]^2))),
             FunctionPlayerCost((g, x, u, t) -> ( 0*(x[3]^2 + x[4]^2) + θ[3]*((x[1]-x[3])^2 + (x[2]-x[4])^2) + (u[3]^2 + u[4]^2))))
    return costs
end
nx, nu, ΔT, game_horizon = 4, 4, 0.1, 40
costs = (FunctionPlayerCost((g, x, u, t) -> ( 2*(x[3])^2 + 2*(x[4])^2 + u[1]^2 + u[2]^2)),
         FunctionPlayerCost((g, x, u, t) -> ( 2*(x[1]-x[3])^2 + 2*(x[2]-x[4])^2 + u[3]^2 + u[4]^2)))
dynamics = LTISystem(LinearSystem{ΔT}(SMatrix{4,4}(Matrix(1.0*I,4,4)), SMatrix{4,4}(Matrix(1.0*I,4,4))), 
                SVector(1, 2, 3, 4))
player_inputs = (SVector(1,2), SVector(3,4))
game = GeneralGame(game_horizon, player_inputs, dynamics, costs)


solver = iLQSolver(game, max_scale_backtrack=10, max_elwise_diff_step=Inf, equilibrium_type="FBNE_costate")

GD_iter_num = 50
num_clean_traj = 10
noise_level_list = 0.0:0.02:0.1
num_noise_level = length(noise_level_list)
num_obs = 10
x0 = SVector(0, 1, 1,1)
x0_set = [x0+0.1*(rand(4)-0.5*ones(4)) for ii in 1:num_clean_traj]
# θ_true = [0.0;2.0;1.0;0.0; 2.0;1.0]

# nx, nu, ΔT, game_horizon = 4, 4, 0.1, 40
# costs = (FunctionPlayerCost((g, x, u, t) -> ( 2*(x[3])^2 + 2*(x[4])^2 + u[1]^2 + u[2]^2)),
#          FunctionPlayerCost((g, x, u, t) -> ( 2*(x[1]-x[3])^2 + 2*(x[2]-x[4])^2 + u[3]^2 + u[4]^2)))
# player_inputs = (SVector(1,2), SVector(3,4))
expert_traj_list, c_expert = generate_expert_traj(game, solver, x0_set, num_clean_traj)
if sum([c_expert[ii]==false for ii in 1:length(c_expert)]) >0
    @warn "regenerate expert demonstrations because some of the expert demonstration not converged!!!"
end
# c_expert,expert_traj_list,expert_equi_list=generate_traj(g,x0_set,parameterized_cost,["FBNE_costate","OLNE_costate"])
noisy_expert_traj_list = [[[zero(SystemTrajectory, game) for kk in 1:num_obs] for jj in 1:num_noise_level] for ii in 1:num_clean_traj];

end

Threads.@threads for ii in 1:num_clean_traj
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

@everywhere begin
conv_table_list = [[[] for jj in 1:num_noise_level] for ii in 1:num_clean_traj];
sol_table_list = deepcopy(conv_table_list);
loss_table_list = deepcopy(conv_table_list);
grad_table_list = deepcopy(conv_table_list);
equi_table_list = deepcopy(conv_table_list);
iter_table_list = deepcopy(conv_table_list);
comp_time_table_list = deepcopy(conv_table_list);

θ_list_list = deepcopy(conv_table_list);
index_list_list = deepcopy(conv_table_list);
optim_loss_list_list = deepcopy(conv_table_list);
state_prediction_error_list_list = deepcopy(conv_table_list);
generalization_error_list = deepcopy(conv_table_list);
θ₀ = ones(3);
end

num_generalization = 6
# test_x0_list = [x0+0.5*(rand(4)-0.5*ones(4)) for ii in 1:num_generalization]

# ---------------------------------------------------------------  (1)
Threads.@threads for ii in ProgressBar(1:num_clean_traj)
    for jj in 1:num_noise_level
        conv_table,sol_table,loss_table,grad_table,equi_table,iter_table,comp_time_table=run_experiment(game,θ₀,[x0_set[ii] for kk in 1:num_obs], 
                                                                                                noisy_expert_traj_list[ii][jj], parameterized_cost, GD_iter_num, 20, 1e-8, 
                                                                                                1:game_horizon-1,1:nx, 1:nu, "FBNE_costate", 0.001, false)
        @infiltrate
        θ_list, index_list, optim_loss_list = get_the_best_possible_reward_estimate_single([x0_set[ii] for kk in 1:num_obs], ["FBNE_costate","FBNE_costate"], sol_table, loss_table, equi_table)
        state_prediction_error_list = loss(θ_list[1], iLQGames.dynamics(game), "FBNE_costate", expert_traj_list[ii], true, false, [], [], 
                                            1:game_horizon-1, 1:nx, 1:nu) # the first true represents whether ignore outputing expert trajectories 
        # generalization_error = generalization_loss(games[ii], θ_list[1], [x0+0.5*(rand(4)-0.5*ones(4)) for ii in 1:num_generalization], 
        #                             expert_traj_list, parameterized_cost, equilibrium_type_list) #problem
        
        push!(state_prediction_error_list_list[ii][jj], state_prediction_error_list)
        push!(conv_table_list[ii][jj], conv_table)
        push!(sol_table_list[ii][jj], sol_table)
        push!(loss_table_list[ii][jj], loss_table)
        push!(grad_table_list[ii][jj], grad_table)
        push!(equi_table_list[ii][jj], equi_table)
        push!(iter_table_list[ii][jj], equi_table)
        push!(comp_time_table_list[ii][jj], comp_time_table)
        push!(θ_list_list[ii][jj], θ_list)
        push!(index_list_list[ii][jj], index_list)
        push!(optim_loss_list_list[ii][jj], optim_loss_list)
        # push!(generalization_error_list[ii][jj], generalization_error)
    end
end
# ---------------------------------------------------------------  (2)
using Random
num_time = 20
obs_time_list = sort!(shuffle(1:game_horizon-1)[1:num_time])
obs_state_list = [1,2,3]
obs_control_list = [1,2,3,4]

Threads.@threads for ii in ProgressBar(1:num_clean_traj)
    for jj in 1:num_noise_level
        conv_table,sol_table,loss_table,grad_table,equi_table,iter_table,comp_time_table=run_experiment(game,θ₀,[x0_set[ii] for kk in 1:num_obs], 
                                                                                                noisy_expert_traj_list[ii][jj], parameterized_cost, GD_iter_num, 20, 1e-8, 
                                                                                                obs_time_list,obs_state_list, obs_control_list, "FBNE_costate", 0.001, false)
        @infiltrate
        θ_list, index_list, optim_loss_list = get_the_best_possible_reward_estimate_single([x0_set[ii] for kk in 1:num_obs], ["FBNE_costate","FBNE_costate"], sol_table, loss_table, equi_table)
        state_prediction_error_list = loss(θ_list[1], iLQGames.dynamics(game), "FBNE_costate", expert_traj_list[ii], true, false, [], [], 
                                            1:game_horizon-1, 1:nx, 1:nu) # the first true represents whether ignore outputing expert trajectories 
        # generalization_error = generalization_loss(games[ii], θ_list[1], [x0+0.5*(rand(4)-0.5*ones(4)) for ii in 1:num_generalization], 
        #                             expert_traj_list, parameterized_cost, equilibrium_type_list) #problem
        
        push!(state_prediction_error_list_list[ii][jj], state_prediction_error_list)
        push!(conv_table_list[ii][jj], conv_table)
        push!(sol_table_list[ii][jj], sol_table)
        push!(loss_table_list[ii][jj], loss_table)
        push!(grad_table_list[ii][jj], grad_table)
        push!(equi_table_list[ii][jj], equi_table)
        push!(iter_table_list[ii][jj], equi_table)
        push!(comp_time_table_list[ii][jj], comp_time_table)
        push!(θ_list_list[ii][jj], θ_list)
        push!(index_list_list[ii][jj], index_list)
        push!(optim_loss_list_list[ii][jj], optim_loss_list)
        # push!(generalization_error_list[ii][jj], generalization_error)
    end
end

# ii -> nominal traj, jj -> noise level, index -> information pattern
mean_predictions = [zeros(num_noise_level) for index in 1:1]
variance_predictions = [zeros(num_noise_level) for index in 1:1]
for index in 1:1 # three information patterns
    for jj in 1:num_noise_level
        mean_predictions[index][jj] = mean(reduce(vcat,[state_prediction_error_list_list[ii][jj][index] for ii in 1:num_clean_traj]))
        variance_predictions[index][jj] = var(reduce(vcat,[state_prediction_error_list_list[ii][jj][index] for ii in 1:num_clean_traj]))
    end
end

plot(noise_level_list, mean_predictions, ribbons=(variance_predictions, variance_predictions))


for jj in num_noise_level
    plot(noise_level_list, state_prediction_error_list_list)
end

mean_predictions_loss = [zeros(num_noise_level) for index in 1:1]
variance_predictions_loss = [zeros(num_noise_level) for index in 1:1]
for index in 1:1 # three information patterns
    for jj in 1:num_noise_level
        mean_predictions_loss[index][jj] = mean(reduce(vcat,[optim_loss_list_list[ii][jj][index] for ii in 1:num_clean_traj]))
        variance_predictions_loss[index][jj] = var(reduce(vcat,[optim_loss_list_list[ii][jj][index] for ii in 1:num_clean_traj]))
    end
end

plot(noise_level_list, mean_predictions_loss, ribbons=(variance_predictions_loss, variance_predictions_loss))
plot!(xlabel="noise variance", ylabel="loss")



mean_GD = zeros((sum(iter_table_list[1][1][1][1][ii]!="" for ii in 1:length(iter_table_list[1][1][1][1]))))
var_GD = zeros((sum(iter_table_list[1][1][1][1][ii]!="" for ii in 1:length(iter_table_list[1][1][1][1]))))
for jj in 1:length(mean_GD)
    mean_GD[jj] = mean(reduce(vcat, loss_table_list[1][1][1][ii][jj] for ii in 1:num_obs))
    var_GD[jj] = var(reduce(vcat, loss_table_list[1][1][1][ii][jj] for ii in 1:num_obs))
end

plot(1:length(mean_GD), mean_GD, ribbons = (var_GD, var_GD), xlabel = "iterations", ylabel = "loss")




