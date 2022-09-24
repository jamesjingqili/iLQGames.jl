using Distributed
@everywhere using Pkg
@everywhere Pkg.activate("../")
@everywhere Pkg.instantiate()
@everywhere begin
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
    include("../src/diff_solver.jl")
    include("../src/inverse_game_solver.jl")
    include("../src/experiment_utils.jl") # NOTICE!! Many functions are defined there.
end

@everywhere begin
# parametes: number of states, number of inputs, sampling time, horizon
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
solver1 = iLQSolver(g, max_scale_backtrack=10, max_elwise_diff_step=Inf, equilibrium_type="OLNE_costate")
x0 = SVector(0, 0.5, pi/2, 1,       1, 0, pi/2, 1)
c1, expert_traj1, strategies1 = solve(g, solver1, x0)

solver2 = iLQSolver(g, max_scale_backtrack=5, max_elwise_diff_step=Inf, equilibrium_type="FBNE_costate")
c2, expert_traj2, strategies2 = solve(g, solver2, x0)

function parameterized_cost(θ::Vector)
    costs = (FunctionPlayerCost((g, x, u, t) -> ( θ[1]*(x[5]-1)^2  + (2*(x[4]-1)^2 + u[1]^2 + u[2]^2) - 0*((x[1]-x[5])^2 + (x[2]-x[6])^2))),
             FunctionPlayerCost((g, x, u, t) -> ( θ[2]*(x[5]-0)^2+θ[3]*(x[5] - x[1])^2 + (2*(x[8]-1)^2 + u[3]^2 + u[4]^2) - 0*((x[1]-x[5])^2 + (x[2]-x[6])^2))))
    return costs
end

# θ_true = [10, 1, 1, 4, 1]
θ_true = [4, 0, 4]

end

# ------------------------------------------------------------------------------------------------------------------------------------------
"Experiment 2: With noise. Scatter plot"
# X: noise variance
# Y1: state prediction loss, mean and variance
# Y2: generalization loss, mean and variance
@everywhere begin

GD_iter_num = 50
num_clean_traj = 1
noise_level_list = 0.01:0.01:0.05
num_noise_level = length(noise_level_list)
num_obs = 10
games = []
x0_set = [x0+0.0*rand(8).*[1;1;0;0;1;1;0;0] for ii in 1:num_clean_traj]
# θ_true = [2.0;2.0;1.0;2.0;2.0;1.0;0.0;0.0]

c_expert,expert_traj_list,expert_equi_list=generate_traj(g,x0_set,parameterized_cost,["FBNE_costate","OLNE_costate"])
noisy_expert_traj_list = [[[zero(SystemTrajectory, g) for kk in 1:num_obs] for jj in 1:num_noise_level] for ii in 1:num_clean_traj]

end
@sync @distributed for ii in 1:num_clean_traj
    for jj in 1:num_noise_level
        tmp = generate_noisy_observation(nx, nu, g, expert_traj_list[ii], noise_level_list[jj], num_obs);
        for kk in 1:num_obs
            for t in 1:g.h
                noisy_expert_traj_list[ii][jj][kk].x[t] = tmp[kk].x[t];
                noisy_expert_traj_list[ii][jj][kk].u[t] = tmp[kk].u[t];
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
# generalization_error_list = deepcopy(conv_table_list);
ground_truth_loss_list = deepcopy(conv_table_list);

θ₀ = 2*ones(3);

end

for ii in 1:num_clean_traj
    for jj in 1:num_noise_level
        conv_table,sol_table,loss_table,grad_table,equi_table,iter_table,ground_truth_loss = run_experiment(game,θ₀,[x0_set[ii] for kk in 1:num_obs], 
                                                                                                noisy_expert_traj_list[ii][jj], parameterized_cost, GD_iter_num, 20, 1e-8, 
                                                                                                1:game_horizon-1,1:nx, 1:nu, "FBNE_costate", 0.00000000001, false, 10.0, expert_traj_list[ii])
        θ_list, index_list, optim_loss_list = get_the_best_possible_reward_estimate_single([x0_set[ii] for kk in 1:num_obs], ["FBNE_costate","FBNE_costate"], sol_table, loss_table, equi_table)
        # state_prediction_error_list = loss(θ_list[1], iLQGames.dynamics(game), "FBNE_costate", expert_traj_list[ii], true, false, [], [], 
        #                                     1:game_horizon-1, 1:nx, 1:nu) # the first true represents whether ignore outputing expert trajectories 
        # generalization_error = generalization_loss(games[ii], θ_list[1], [x0+0.5*(rand(4)-0.5*ones(4)) for ii in 1:num_generalization], 
        #                             expert_traj_list, parameterized_cost, equilibrium_type_list) #problem
        
        # push!(state_prediction_error_list_list[ii][jj], state_prediction_error_list)
        push!(conv_table_list[ii][jj], conv_table)
        push!(sol_table_list[ii][jj], sol_table)
        push!(loss_table_list[ii][jj], loss_table)
        push!(grad_table_list[ii][jj], grad_table)
        push!(equi_table_list[ii][jj], equi_table)
        push!(iter_table_list[ii][jj], iter_table)
        # push!(comp_time_table_list[ii][jj], comp_time_table)
        push!(θ_list_list[ii][jj], θ_list)
        push!(index_list_list[ii][jj], index_list)
        push!(optim_loss_list_list[ii][jj], optim_loss_list)
        push!(ground_truth_loss_list[ii][jj], ground_truth_loss)
        # push!(generalization_error_list[ii][jj], generalization_error)
    end
end

using JLD2
jldsave("GD_full_10_$(Dates.now())"; noise_level_list, nx, nu, ΔT, game,dynamics, costs, player_inputs, solver, x0, parameterized_cost, GD_iter_num, num_clean_traj, θ_true, θ₀, 
    c_expert, expert_traj_list, conv_table_list, sol_table_list, loss_table_list, grad_table_list, 
    equi_table_list, iter_table_list, comp_time_table_list, θ_list_list, index_list_list, optim_loss_list_list, mean_GD,var_GD, 
    mean_predictions, variance_predictions, mean_predictions_loss, variance_predictions_loss)

# ii -> nominal traj, jj -> noise level, index -> information pattern
mean_predictions = [zeros(num_noise_level) for index in 1:3]
variance_predictions = [zeros(num_noise_level) for index in 1:3]
Threads.@threads for index in 1:3
    for jj in 1:num_noise_level
        mean_predictions[index][jj] = mean(reduce(vcat,[optim_loss_list_list[ii][jj][1][index] for ii in 1:4]))
        variance_predictions[index][jj] = var(reduce(vcat,[optim_loss_list_list[ii][jj][1][index] for ii in 1:4]))
    end
end


plot(noise_level_list, mean_predictions)


jldsave("highway_guiding_data_$(Dates.now())"; nx, nu, ΔT, g,dynamics, costs, player_inputs, x0, 
    parameterized_cost, GD_iter_num, noise_level_list, num_clean_traj, num_obs, θ_true, θ₀, 
    c_expert, expert_traj_list, expert_equi_list, conv_table_list, sol_table_list, loss_table_list, grad_table_list, 
    equi_table_list, iter_table_list, comp_time_table_list, θ_list_list, index_list_list, optim_loss_list_list,
    mean_predictions, variance_predictions)


# -----------------------------------------------------------------------------------------------------------------
using Random
num_time = 10
# obs_time_list = sort!(shuffle(1:game_horizon-1)[1:num_time])
obs_time_list = [1,2,3,4, 10,11,12,13]


conv_table_list1 = [[[] for jj in 1:num_noise_level] for ii in 1:num_clean_traj];
sol_table_list1 = deepcopy(conv_table_list1);
loss_table_list1 = deepcopy(conv_table_list1);
grad_table_list1 = deepcopy(conv_table_list1);
equi_table_list1 = deepcopy(conv_table_list1);
iter_table_list1 = deepcopy(conv_table_list1);
comp_time_table_list1 = deepcopy(conv_table_list1);

θ_list_list1 = deepcopy(conv_table_list1);
index_list_list1 = deepcopy(conv_table_list1);
optim_loss_list_list1 = deepcopy(conv_table_list1);
state_prediction_error_list_list1 = deepcopy(conv_table_list1);
generalization_error_list1 = deepcopy(conv_table_list1);
ground_truth_loss_list1 = deepcopy(conv_table_list1);

obs_state_list = [1,2,3]
obs_control_list = [1,2,3,4]


for ii in 1:num_clean_traj
    for jj in 1:num_noise_level
        conv_table1,sol_table1,loss_table1,grad_table1,equi_table1,iter_table1,ground_truth_loss1=run_experiment(game,θ₀,[x0_set[ii] for kk in 1:num_obs], 
                                                                                                noisy_expert_traj_list[ii][jj], parameterized_cost, GD_iter_num, 20, 1e-8, 
                                                                                                obs_time_list,obs_state_list, obs_control_list, "FBNE_costate", 0.00000001, false, 10, expert_traj_list[ii])
        θ_list1, index_list1, optim_loss_list1 = get_the_best_possible_reward_estimate_single([x0_set[ii] for kk in 1:num_obs], ["FBNE_costate","FBNE_costate"], sol_table1, loss_table1, equi_table1)
        # state_prediction_error_list1 = loss(θ_list1[1], iLQGames.dynamics(game), "FBNE_costate", expert_traj_list[ii], true, false, [], [], 
        #                                     1:game_horizon-1, 1:nx, 1:nu) # the first true represents whether ignore outputing expert trajectories 
        # generalization_error = generalization_loss(games[ii], θ_list[1], [x0+0.5*(rand(4)-0.5*ones(4)) for ii in 1:num_generalization], 
        #                             expert_traj_list, parameterized_cost, equilibrium_type_list) #problem
        
        # push!(state_prediction_error_list_list1[ii][jj], state_prediction_error_list1)
        push!(conv_table_list1[ii][jj], conv_table1)
        push!(sol_table_list1[ii][jj], sol_table1)
        push!(loss_table_list1[ii][jj], loss_table1)
        push!(grad_table_list1[ii][jj], grad_table1)
        push!(equi_table_list1[ii][jj], equi_table1)
        push!(iter_table_list1[ii][jj], iter_table1)
        # push!(comp_time_table_list1[ii][jj], comp_time_table1)
        push!(θ_list_list1[ii][jj], θ_list1)
        push!(index_list_list1[ii][jj], index_list1)
        push!(optim_loss_list_list1[ii][jj], optim_loss_list1)
        push!(ground_truth_loss_list1, ground_truth_loss1)
        # push!(generalization_error_list[ii][jj], generalization_error)
    end
end









#----------------------------------------------------------------------------------------------------------------
num_test
test_x0_set = [x0+0.2*rand(8).*[1;1;0;0;1;1;0;0] for ii in 1:num_test]



