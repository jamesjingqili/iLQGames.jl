


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

num_players=3
nx, nu, ΔT, game_horizon = 4*num_players+1+4, 2*num_players, 0.1, 30
struct ThreeCar <: ControlSystem{ΔT,nx,nu} end
dx(cs::ThreeCar, x, u, t) = SVector(x[4]cos(x[3]),   x[4]sin(x[3]),   u[1], u[2], 
                                        x[8]cos(x[7]),   x[8]sin(x[7]),   u[3], u[4],
                                        x[12]cos(x[11]), x[12]sin(x[11]), u[5], u[6],
                                        0,0,0,0,0
                                        )
dynamics = ThreeCar()
# platonning
# x0 = SVector(0.0, 3, pi/2, 2,       0.3, 0, pi/2, 2,      0.7, 2,pi/2,1,                   0.2,     0, 10, 0, 10  )
# x0 = SVector(0.4, 3, pi/2, 2,       0.4, 0, pi/2, 2,      0.7, 2,pi/2,1,                   0.0,     0, 10, 0, 10  )
# costs = (FunctionPlayerCost((g,x,u,t) -> ( x[14]*(x[1])^2  + x[15]*(x[5]-x[13])^2  + 4*(x[3]-pi/2)^2   +8*(x[4]-2)^2       +2*(u[1]^2 + u[2]^2)    )),
#          FunctionPlayerCost((g,x,u,t) -> ( x[16]*(x[5])^2  + x[17]*(x[5]-x[1])^2     +  8*(x[8]-2)^2  +4*(x[7]-pi/2)^2     -log((x[5]-x[9])^2+(x[6]-x[10])^2)  +2*(u[3]^2+u[4]^2)    )),
#          FunctionPlayerCost((g,x,u,t) -> ( 2*(x[9]-x0[9])^2   +2*(u[5]^2+u[6]^2)  ))
#     )
# x0 = SVector(0.0, 1, pi/2, 2,       1, 0, pi/2, 2,   0.5, 0.5,pi/2,2,                   0.2, 0, 8, 8, 0)
x0 = SVector(0.0, 1, pi/2, 2,       0.3, 0, pi/2, 2,   0.5, 0.5,pi/2,2,                   0.2, 0, 8, 8, 0)
costs = (FunctionPlayerCost((g,x,u,t) -> ( x[14]*x[1]^2 + x[15]*(x[5]-x[13])^2   +4*(x[3]-pi/2)^2  +2*(x[4]-2)^2       +2*(u[1]^2 + u[2]^2)    )),
         FunctionPlayerCost((g,x,u,t) -> ( x[16]*(x[5]-x[1])^2  +x[17]*x[5]^2  +4*(x[7]-pi/2)^2  +2*(x[8]-2)^2       -log((x[5]-x[9])^2+(x[6]-x[10])^2)    +2*(u[3]^2+u[4]^2)    )),
         FunctionPlayerCost((g,x,u,t) -> ( 2*(x[9]-x0[9])^2   + 2*(u[5]^2+u[6]^2)  ))
    )
player_inputs = (SVector(1,2), SVector(3,4), SVector(5,6))
g = GeneralGame(game_horizon, player_inputs, dynamics, costs)
# get a solver, choose initial conditions and solve (in about 9 ms with AD)

# x0 = SVector(0.0, 3, pi/2, 2,       0.3, 0, pi/2, 1.5,      0.5, 2,pi/2,1,                   1,     0, 10, 0, 10  )


solver1 = iLQSolver(g, max_scale_backtrack=5, max_elwise_diff_step=Inf, equilibrium_type="OLNE_costate")
c1, expert_traj1, strategies1 = solve(g, solver1, x0)
solver2 = iLQSolver(g, max_scale_backtrack=5, max_elwise_diff_step=Inf, equilibrium_type="FBNE_costate")
c2, expert_traj2, strategies2 = solve(g, solver2, x0)

# function parameterized_cost(θ::Vector)
# costs = (FunctionPlayerCost((g,x,u,t) -> ( x[14]*(x[1])^2  + x[15]*(x[5]-x[13])^2  + 4*(x[3]-pi/2)^2   +8*(x[4]-2)^2       +2*(u[1]^2 + u[2]^2)    )),
#          FunctionPlayerCost((g,x,u,t) -> ( x[16]*(x[5])^2  + x[17]*(x[5]-x[1])^2     +  8*(x[8]-2)^2  +4*(x[7]-pi/2)^2     -log((x[5]-x[9])^2+(x[6]-x[10])^2)  +2*(u[3]^2+u[4]^2)    )),
#          FunctionPlayerCost((g,x,u,t) -> ( 2*(x[9]-x0[9])^2   +2*(u[5]^2+u[6]^2)  ))
#     )
#     return costs
# end
function parameterized_cost(θ::Vector)
costs = (FunctionPlayerCost((g,x,u,t) -> ( x[14]*(x[1])^2  + x[15]*(x[5]-x[13])^2  + 4*(x[3]-pi/2)^2   +8*(x[4]-2)^2       +2*(u[1]^2 + u[2]^2)    )),
         FunctionPlayerCost((g,x,u,t) -> ( x[16]*(x[5])^2  + x[17]*(x[5]-x[1])^2     +  8*(x[8]-2)^2  +4*(x[7]-pi/2)^2     -log((x[5]-x[9])^2+(x[6]-x[10])^2)  +2*(u[3]^2+u[4]^2)    )),
         FunctionPlayerCost((g,x,u,t) -> ( 2*(x[9]-x0[9])^2   +2*(u[5]^2+u[6]^2)  ))
    )
    return costs
end
θ_true = [0, 8, 8, 0]


tmp = new_loss([1,1,1,1], dynamics, "FBNE_costate", expert_traj2, false, false, [], [], 1:game_horizon-1, 1:nx, 1:nu, false, false, [], static_game, static_solver, true_game_nx)


include("new_experiment_utils.jl")
static_game = g
static_solver = solver2
true_game_nx = 13
ForwardDiff.gradient(x -> new_loss(x, dynamics, "FBNE_costate", expert_traj2, true, false, [], [], 1:game_horizon-1, 1:nx, 1:nu, false, false, [], 
                    static_game, static_solver, true_game_nx), [1,1,1,1])

ForwardDiff.gradient(x -> new_loss([1,1,1,1], dynamics, "FBNE_costate", expert_traj2, true, false, [], [], 1:game_horizon-1, 1:nx, 1:nu, false, true, x, static_game, 
                    static_solver,  true_game_nx), [0.5, 3, pi/2, 2,       0.5, 0, pi/2, 1.5,      0.5, 2,pi/2,1,                   1,     0, 10, 0, 10  ])

# ------------------------------------------------------------------------------------------------------------------------------------------
"Experiment 2: With noise. Scatter plot"
# X: noise variance
# Y1: state prediction loss, mean and variance
# Y2: generalization loss, mean and variance

GD_iter_num = 80
num_clean_traj = 1
noise_level_list = 0.004:0.008:0.04
# noise_level_list=[0.0]
num_noise_level = length(noise_level_list)
num_obs = 6
games = []
x0_set = [x0 for ii in 1:num_clean_traj]
# θ_true = [2.0;2.0;1.0;2.0;2.0;1.0;0.0;0.0]

c_expert,expert_traj_list,expert_equi_list=generate_traj(g,x0_set,costs,["FBNE_costate","FBNE_costate"])
noisy_expert_traj_list = [[[zero(SystemTrajectory, g) for kk in 1:num_obs] for jj in 1:num_noise_level] for ii in 1:num_clean_traj]

for ii in 1:num_clean_traj
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


conv_table_list = [[[] for jj in 1:num_noise_level] for ii in 1:num_clean_traj];
sol_table_list = deepcopy(conv_table_list);
x0_table_list = deepcopy(conv_table_list);
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
ground_truth_loss_list = deepcopy(conv_table_list);
init_x0_list = deepcopy(conv_table_list);

θ₀ = 4*ones(4);

num_test=6
test_noise_level=1.0
test_x0_set = [x0 - [zeros(12);x0[13];zeros(4)] + test_noise_level*[zeros(12);rand(1)[1];zeros(4)] for ii in 1:num_test];

test_expert_traj_list, c_test_expert = generate_expert_traj(g, solver2, test_x0_set, num_test);


# solver_per_thread = [deepcopy(solver2) for _ in 1:Threads.nthreads()]

# obs_time_list= [1,2,3,4,5,6,11,12,13,14,15,16,21,22,23,24,25,26,31,32,33,34,35,36]
# obs_time_list= [1,2,3,4,5,6,7,8,9,10,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39]
obs_time_list = [1:10; 21:g.h-1]
obs_state_list = [1,2,3,5,6,7, 9,10,11,13]
obs_control_list = 1:nu
# obs_state_list = 1:nx

# obs_time_list = 1:game_horizon-1

regularization_size=1e-4
random_init_x0=false
if obs_state_list != 1:nx
    random_init_x0 = true
end
for ii in 1:num_clean_traj
    for jj in 1:num_noise_level
        if noise_level_list[jj] == 0.0
            tmp_num_obs = num_obs
        else
            tmp_num_obs = num_obs
        end
        if random_init_x0 == true
            init_x0 = [noisy_expert_traj_list[ii][jj][kk].x[1]-[0,0,0,noisy_expert_traj_list[ii][jj][kk].x[1][4],
                        0,0,0,noisy_expert_traj_list[ii][jj][kk].x[1][8],
                        0,0,0,noisy_expert_traj_list[ii][jj][kk].x[1][12],0,0,0,0,0] + (0.8*ones(nx)+0.4*rand(nx)).*[0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,0,0]  for kk in 1:tmp_num_obs]
        else
            init_x0 = [noisy_expert_traj_list[ii][jj][kk].x[1]  for kk in 1:tmp_num_obs]
        end
        println("Now the $(jj)-th noise level")
        conv_table,x0_table,sol_table,loss_table,grad_table,equi_table,iter_table,ground_truth_loss = new_run_experiment_x0(g,θ₀,init_x0, 
                                                                                                noisy_expert_traj_list[ii][jj], parameterized_cost, GD_iter_num, 15, 1e-4, 
                                                                                                obs_time_list,obs_state_list, obs_control_list, "FBNE_costate", 0.001, 
                                                                                                true, 10.0,expert_traj_list[ii],    true   ,false,[],true,
                                                                                                10, 0.1, 0.1, static_game, static_solver, true_game_nx )
        θ_list, index_list, optim_loss_list = get_the_best_possible_reward_estimate_single(init_x0, ["FBNE_costate","FBNE_costate"], sol_table, loss_table, equi_table)
        push!(conv_table_list[ii][jj], conv_table)
        push!(x0_table_list[ii][jj], x0_table)
        push!(init_x0_list[ii][jj], init_x0)
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



list_ground_truth_loss = [[[] for jj in 1:num_noise_level] for ii in 1:num_clean_traj]
list_generalization_loss = [[[] for jj in 1:num_noise_level] for ii in 1:num_clean_traj]
for ii in 1:num_clean_traj
    for jj in 1:num_noise_level
        tmp_ground_truth_loss = zeros(num_obs)
        tmp_generalization_loss = zeros(num_obs)
        for kk in 1:num_obs
            tmp = zeros(num_test)
            tmp_ground_truth_loss[kk] = new_loss(θ_list_list[ii][jj][1][kk], iLQGames.dynamics(g), "FBNE_costate", expert_traj_list[ii], true,false, [],[],1:g.h-1, 1:12, 1:nu, true, false, [], g, solver2, 13)
            for kkk in 1:num_test
                tmp[kkk] = new_loss(θ_list_list[ii][jj][1][kk], iLQGames.dynamics(g), "FBNE_costate", test_expert_traj_list[kkk], true, false,[],[],1:g.h-1, 1:12, 1:nu, true, false, [], g, solver2, 13)
            end
            tmp_generalization_loss[kk] = mean(tmp)
            println("Current iteration: $(jj), $(kk)")

        end
        # for kk in 1:num_test
        #     tmp_generalization_loss[kk] = loss(θ_list_list[ii][jj][1][1], iLQGames.dynamics(g), "FBNE_costate", test_expert_traj_list[kk], true, false,[],[],1:g.h-1, 1:nx, 1:nu)
        # end
        push!(list_ground_truth_loss[ii][jj], tmp_ground_truth_loss)
        push!(list_generalization_loss[ii][jj], tmp_generalization_loss)
        println(tmp_ground_truth_loss)
        println(tmp_generalization_loss)
    end
end


# each solver for each thread




using JLD2
jldsave("GD_no_control_2car_0.1_20_partial_x0$(Dates.now())"; noise_level_list, nx, nu, ΔT, g, dynamics, costs, player_inputs, solver1, solver2, x0, parameterized_cost, GD_iter_num, num_clean_traj, θ_true, θ₀, 
    c_expert, expert_traj_list, conv_table_list, sol_table_list, loss_table_list, grad_table_list, noisy_expert_traj_list,x0_set, test_x0_set,test_expert_traj_list,
    equi_table_list, iter_table_list, comp_time_table_list, θ_list_list, index_list_list, optim_loss_list_list, ground_truth_loss_list, generalization_error_list,
    # mean_prediction_loss, var_prediction_loss, mean_gen_loss, var_gen_loss, 
    list_generalization_loss, list_ground_truth_loss, tmp1,tmp1_var, tmp2, tmp2_var, tmp3, tmp3_var,
    tmp14,tmp14_var,tmp15,tmp15_var, tmp16, tmp16_var)

jldsave("GD_no_control_2car_0.1_20_full_x0$(Dates.now())"; noise_level_list, nx, nu, ΔT, g, dynamics, costs, player_inputs, solver1, solver2, x0, parameterized_cost, GD_iter_num, num_clean_traj, θ_true, θ₀, 
    c_expert, expert_traj_list, conv_table_list, sol_table_list, loss_table_list, grad_table_list, noisy_expert_traj_list,x0_set, test_x0_set,test_expert_traj_list,
    equi_table_list, iter_table_list, comp_time_table_list, θ_list_list, index_list_list, optim_loss_list_list, ground_truth_loss_list, generalization_error_list,
    # mean_prediction_loss, var_prediction_loss, mean_gen_loss, var_gen_loss, 
    list_generalization_loss, list_ground_truth_loss, tmp1,tmp1_var, tmp2, tmp2_var, tmp3, tmp3_var,
    tmp14,tmp14_var,tmp15,tmp15_var, tmp16, tmp16_var)

jldsave("1008_baobei_GD_2car_full_x0$(Dates.now())"; noise_level_list, nx, nu, ΔT, g, dynamics, costs, player_inputs, solver1, solver2, x0, parameterized_cost, GD_iter_num, num_clean_traj, θ_true, θ₀, 
    c_expert, expert_traj_list, conv_table_list, sol_table_list, loss_table_list, grad_table_list, noisy_expert_traj_list,x0_set, test_x0_set,test_expert_traj_list,
    equi_table_list, iter_table_list, comp_time_table_list, θ_list_list, index_list_list, optim_loss_list_list, ground_truth_loss_list, generalization_error_list,
    # mean_prediction_loss, var_prediction_loss, mean_gen_loss, var_gen_loss, 
    list_generalization_loss, list_ground_truth_loss, tmp14,tmp14_var,tmp15,tmp15_var, tmp16, tmp16_var)

jldsave("1008_baobei_GD_2car_partial_x0$(Dates.now())"; noise_level_list, nx, nu, ΔT, g, dynamics, costs, player_inputs, solver1, solver2, x0, parameterized_cost, GD_iter_num, num_clean_traj, θ_true, θ₀, 
    c_expert, expert_traj_list, conv_table_list, sol_table_list, loss_table_list, grad_table_list, noisy_expert_traj_list,x0_set, test_x0_set,test_expert_traj_list,
    equi_table_list, iter_table_list, comp_time_table_list, θ_list_list, index_list_list, optim_loss_list_list, ground_truth_loss_list, generalization_error_list,
    # mean_prediction_loss, var_prediction_loss, mean_gen_loss, var_gen_loss, 
    list_generalization_loss, list_ground_truth_loss, tmp14,tmp14_var,tmp15,tmp15_var, tmp16, tmp16_var)

jldsave("1014_baobei_GD_3car_full_x0$(Dates.now())"; noise_level_list, nx, nu, ΔT, g, dynamics, costs, player_inputs, solver1, solver2, x0, parameterized_cost, GD_iter_num, num_clean_traj, θ_true, θ₀, 
    c_expert, expert_traj_list, conv_table_list, sol_table_list, loss_table_list, grad_table_list, noisy_expert_traj_list,x0_set, test_x0_set,test_expert_traj_list,
    equi_table_list, iter_table_list, comp_time_table_list, θ_list_list, index_list_list, optim_loss_list_list, ground_truth_loss_list, generalization_error_list,
    # mean_prediction_loss, var_prediction_loss, mean_gen_loss, var_gen_loss, 
    list_generalization_loss, list_ground_truth_loss, tmp14,tmp14_var,tmp15,tmp15_var, tmp16, tmp16_var)


jldsave("Indy_baobei_GD_3car_full_x0$(Dates.now())"; noise_level_list, nx, nu, ΔT, g, dynamics, costs, player_inputs, solver1, solver2, x0, parameterized_cost, GD_iter_num, num_clean_traj, θ_true, θ₀, 
    c_expert, expert_traj_list, conv_table_list, sol_table_list, loss_table_list, grad_table_list, noisy_expert_traj_list,x0_set, test_x0_set,test_expert_traj_list,
    equi_table_list, iter_table_list, comp_time_table_list, θ_list_list, index_list_list, optim_loss_list_list, ground_truth_loss_list, generalization_error_list,
    # mean_prediction_loss, var_prediction_loss, mean_gen_loss, var_gen_loss, 
    list_generalization_loss, list_ground_truth_loss, tmp14,tmp14_var,tmp15,tmp15_var, tmp16, tmp16_var)

jldsave("Indy_baobei_GD_3car_partial_x0$(Dates.now())"; noise_level_list, nx, nu, ΔT, g, dynamics, costs, player_inputs, solver1, solver2, x0, parameterized_cost, GD_iter_num, num_clean_traj, θ_true, θ₀, 
    c_expert, expert_traj_list, conv_table_list, sol_table_list, loss_table_list, grad_table_list, noisy_expert_traj_list,x0_set, test_x0_set,test_expert_traj_list,
    equi_table_list, iter_table_list, comp_time_table_list, θ_list_list, index_list_list, optim_loss_list_list, ground_truth_loss_list, generalization_error_list,
    # mean_prediction_loss, var_prediction_loss, mean_gen_loss, var_gen_loss, 
    list_generalization_loss, list_ground_truth_loss, tmp14,tmp14_var,tmp15,tmp15_var, tmp16, tmp16_var)

jldsave("Walnut_Creek_Indy_baobei_GD_3car_full_x0$(Dates.now())"; noise_level_list, nx, nu, ΔT, g, dynamics, costs, player_inputs, solver1, solver2, x0, parameterized_cost, GD_iter_num, num_clean_traj, θ_true, θ₀, 
    c_expert, expert_traj_list, conv_table_list, sol_table_list, loss_table_list, grad_table_list, noisy_expert_traj_list,x0_set, test_x0_set,test_expert_traj_list,
    equi_table_list, iter_table_list, comp_time_table_list, θ_list_list, index_list_list, optim_loss_list_list, ground_truth_loss_list, generalization_error_list,
    # mean_prediction_loss, var_prediction_loss, mean_gen_loss, var_gen_loss, 
    list_generalization_loss, list_ground_truth_loss, tmp14,tmp14_var,tmp15,tmp15_var, tmp16, tmp16_var)

jldsave("Walnut_Creek_Indy_baobei_GD_3car_partial_x0$(Dates.now())"; noise_level_list, nx, nu, ΔT, g, dynamics, costs, player_inputs, solver1, solver2, x0, parameterized_cost, GD_iter_num, num_clean_traj, θ_true, θ₀, 
    c_expert, expert_traj_list, conv_table_list, sol_table_list, loss_table_list, grad_table_list, noisy_expert_traj_list,x0_set, test_x0_set,test_expert_traj_list,
    equi_table_list, iter_table_list, comp_time_table_list, θ_list_list, index_list_list, optim_loss_list_list, ground_truth_loss_list, generalization_error_list,
    # mean_prediction_loss, var_prediction_loss, mean_gen_loss, var_gen_loss, 
    list_generalization_loss, list_ground_truth_loss, tmp14,tmp14_var,tmp15,tmp15_var, tmp16, tmp16_var)



# ii -> nominal traj, jj -> noise level, index -> information pattern
# mean_predictions = [zeros(num_noise_level) for index in 1:3]
# variance_predictions = [zeros(num_noise_level) for index in 1:3]
# Threads.@threads for index in 1:3
#     for jj in 1:num_noise_level
#         mean_predictions[index][jj] = mean(reduce(vcat,[optim_loss_list_list[ii][jj][1][index] for ii in 1:4]))
#         variance_predictions[index][jj] = var(reduce(vcat,[optim_loss_list_list[ii][jj][1][index] for ii in 1:4]))
#     end
# end
# plot(noise_level_list, mean_predictions)


index=1
mean_GD_list = []
var_GD_list = []
for noise in 1:length(noise_level_list)
    mean_GD_local = zeros(GD_iter_num)
    var_GD_local = zeros(GD_iter_num)
    for jj in 1:length(mean_GD_local)
        mean_GD_local[jj] = mean(reduce(vcat, log(ground_truth_loss_list[index][noise][1][ii][jj]) for ii in 1:num_obs))
        var_GD_local[jj] = var(reduce(vcat, log(ground_truth_loss_list[index][noise][1][ii][jj]) for ii in 1:num_obs))
    end
    push!(mean_GD_list, mean_GD_local)
    push!(var_GD_list, var_GD_local)
end

plt=plot()
color_list = ["blue", "green", "orange"]
for noise in 1:3
    plt=plot!(1:length(mean_GD_list[noise]), mean_GD_list[noise],ribbons=(var_GD_list[noise], var_GD_list[noise]),color=color_list[noise],alpha=1, title="Cost Inference under Full observation", xlabel = "iterations", ylabel = "log(||X̂ - X||₂² + ||Û - U ||₂²)", label="σ = $(noise_level_list[noise])")
    for ii in 1:num_obs
        plt=scatter!(1:length(mean_GD_list[noise]), log.(ground_truth_loss_list[1][noise][1][ii]),color=color_list[noise], alpha=0.5, markershape=:x, label="")
    end
end
display(plt)
savefig("full_GD_log_var.pdf")


jldsave("highway_guiding_data_$(Dates.now())"; nx, nu, ΔT, g,dynamics, costs, player_inputs, x0, 
    parameterized_cost, GD_iter_num, noise_level_list, num_clean_traj, num_obs, θ_true, θ₀, 
    c_expert, expert_traj_list, expert_equi_list, conv_table_list, sol_table_list, loss_table_list, grad_table_list, 
    equi_table_list, iter_table_list, comp_time_table_list, θ_list_list, index_list_list, optim_loss_list_list,
    mean_predictions, variance_predictions)


# -----------------------------------------------------------------------------------------------------------------
using Random
num_time = 10
# obs_time_list = sort!(shuffle(1:game_horizon-1)[1:num_time])
obs_time_list = [1,2,3,4,5,6,7,8,9,10, 30,32,33,34,35,36,37,38,39]


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

obs_state_list = [1,3,5,7,8]
obs_control_list = [1,2,3,4]


for ii in 1:num_clean_traj
    for jj in 1:num_noise_level
        conv_table1,sol_table1,loss_table1,grad_table1,equi_table1,iter_table1,ground_truth_loss1=run_experiment(g,θ₀,[x0_set[ii] for kk in 1:num_obs], 
                                                                                                noisy_expert_traj_list[ii][jj], parameterized_cost, GD_iter_num, 20, 1e-4, 
                                                                                                obs_time_list,obs_state_list, obs_control_list, "FBNE_costate", , 0.0001, true, 2.0,[],true)
        θ_list1, index_list1, optim_loss_list1 = get_the_best_possible_reward_estimate_single([x0_set[ii] for kk in 1:num_obs], ["FBNE_costate","FBNE_costate"], sol_table1, ground_truth_loss1, equi_table1)
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


index=1
mean_GD_list1 = []
var_GD_list1 = []
for noise in 1:length(noise_level_list)
    mean_GD_local = zeros(GD_iter_num)
    var_GD_local = zeros(GD_iter_num)
    for jj in 1:length(mean_GD_local)
        mean_GD_local[jj] = mean(reduce(vcat, log(ground_truth_loss_list1[index][noise][1][ii][jj]) for ii in 1:num_obs))
        var_GD_local[jj] = var(reduce(vcat, log(ground_truth_loss_list1[index][noise][1][ii][jj]) for ii in 1:num_obs))
    end
    push!(mean_GD_list1, mean_GD_local)
    push!(var_GD_list1, var_GD_local)
end

# below is full observation
mean_prediction_loss = zeros(length(noise_level_list))
var_prediction_loss = zeros(length(noise_level_list))
index=1
for noise in 1:length(noise_level_list)
    mean_prediction_loss[noise] = mean(reduce(vcat, optim_loss_list_list[index][noise][1][ii] for ii in 1:num_obs))
    var_prediction_loss[noise] = var(reduce(vcat, optim_loss_list_list[index][noise][1][ii] for ii in 1:num_obs))
end

plt = plot()
plt = plot!(noise_level_list[1:9], mean_prediction_loss[1:9], ribbons=(var_prediction_loss[1:9], var_prediction_loss[1:9]))
display(plt)


# mean_loss = zeros(length(noise_level_list))
# var_loss = zeros(length(noise_level_list))
# index=1
# for noise in 1:length(noise_level_list)
#     mean_loss[noise] = mean(reduce(vcat, loss_table_list[index][noise][1][ii] for ii in 1:num_obs))
#     var_loss[noise] = var(reduce(vcat, loss_table_list[index][noise][1][ii] for ii in 1:num_obs))
# end

plt = plot()
plt = plot!(noise_level_list[1:9], mean_loss[1:9], ribbons=(var_loss[1:9], var_loss[1:9]))
display(plt)


mean_gen_loss = zeros(length(noise_level_list))
var_gen_loss = zeros(length(noise_level_list))
index=1
for noise in 1:length(noise_level_list)
    mean_gen_loss[noise] = mean(reduce(vcat, generalization_error_list[index][noise][1][ii] for ii in 1:num_obs))
    var_gen_loss[noise] = var(reduce(vcat, generalization_error_list[index][noise][1][ii] for ii in 1:num_obs))
end
plt = plot(noise_level_list[1:9], mean_gen_loss[1:9], ribbons=(var_gen_loss[1:9], var_gen_loss[1:9]))
display(plt)

#----------------------------------------------------------------------------------------------------------------
# plot
index=1
noise=9
ii = 8
plot([expert_traj_list[index].x[t][1] for t in 1:g.h], [expert_traj_list[index].x[t][2] for t in 1:g.h], color="red", label="player 1, ground truth")
plot!([expert_traj_list[index].x[t][5] for t in 1:g.h], [expert_traj_list[index].x[t][6] for t in 1:g.h], color="blue", label = "player 2, ground truth")
scatter!([noisy_expert_traj_list[index][noise][ii].x[t][1] for t in 1:g.h], [noisy_expert_traj_list[index][noise][ii].x[t][2] for t in 1:g.h], color="red", label = "player 1, noisy observation σ = $(noise_level_list[noise])")
scatter!([noisy_expert_traj_list[index][noise][ii].x[t][5] for t in 1:g.h], [noisy_expert_traj_list[index][noise][ii].x[t][6] for t in 1:g.h], color="blue",label = "player 2, noisy observation σ = $(noise_level_list[noise])")
savefig("sigam_$(noise_level_list[noise])_$(ii).pdf")




index=1
ii = 1
tmp=loss(θ_list_list[1][1][1][1], dynamics, "FBNE_costate", expert_traj2, false, false, [],[],obs_time_list, obs_state_list, obs_control_list, true, true, x0_table_list[1][1][1][1][ index_list_list[1][1][1] ][1])
plot([expert_traj_list[index].x[t][1] for t in 1:g.h], [expert_traj_list[index].x[t][2] for t in 1:g.h], label="ground truth, player 1")
plot!([expert_traj_list[index].x[t][5] for t in 1:g.h], [expert_traj_list[index].x[t][6] for t in 1:g.h], label="ground truth, player 2")
plot!([tmp[2].x[t][1] for t in 1:g.h], [tmp[2].x[t][2] for t in 1:g.h], label="predicted, player 1")
plot!([tmp[2].x[t][5] for t in 1:g.h], [tmp[2].x[t][6] for t in 1:g.h], label="predicted, player 2")
scatter!([expert_traj_list[index].x[t][1] for t in obs_time_list], [expert_traj_list[index].x[t][2] for t in obs_time_list], label="ground truth observation, player 1")
scatter!([expert_traj_list[index].x[t][5] for t in obs_time_list], [expert_traj_list[index].x[t][6] for t in obs_time_list], label="ground truth observation, player 2")



index=1
ii = 8
test_loss, test_traj, _, _ = loss(θ_list_list[1][noise][1][ii], iLQGames.dynamics(g), "FBNE_costate", test_expert_traj_list[index], false, false, 
                [], [], 1:game_horizon-1, 1:nx, 1:nu) 
scatter([test_traj.x[t][1] for t in 1:g.h], [test_traj.x[t][2] for t in 1:g.h], color="red",label="player 1, predicted",legend=:topleft)
scatter!([test_traj.x[t][5] for t in 1:g.h], [test_traj.x[t][6] for t in 1:g.h], color = "blue",label="player 2, predicted")
plot!([test_expert_traj_list[index].x[t][1] for t in 1:g.h], [test_expert_traj_list[index].x[t][2] for t in 1:g.h], color="red", label = "player 1, ground truth")
plot!([test_expert_traj_list[index].x[t][5] for t in 1:g.h], [test_expert_traj_list[index].x[t][6] for t in 1:g.h], color="blue", label="player 2, ground truth")
savefig("generalization_test_$(noise_level_list[noise])_$(ii).pdf")



tmp = load("KKT_compact_data")

#----------------------------------------------------

tmp1 = [mean(tmp["inv_loss_list"][ii])[1] for ii in 1:num_noise_level]
tmp2 = [mean(tmp["inv_ground_truth_loss_list"][ii])[1] for ii in 1:num_noise_level]
tmp3 = [mean(tmp["inv_mean_generalization_loss_list"][ii])[1] for ii in 1:num_noise_level]
tmp1_var = [var(tmp["inv_loss_list"][ii])[1] for ii in 1:num_noise_level]
tmp2_var = [var(tmp["inv_ground_truth_loss_list"][ii])[1] for ii in 1:num_noise_level]
tmp3_var = [var(tmp["inv_mean_generalization_loss_list"][ii])[1] for ii in 1:num_noise_level]

tmp4 = [mean(optim_loss_list_list[1][ii])[1] for ii in 1:num_noise_level]
tmp5 = [ground_truth_loss_list[1][ii][1] for ii in 1:num_noise_level]
tmp6 = [mean(generalization_error_list[1][ii])[1] for ii in 1:num_noise_level]

plot(noise_level_list, tmp1, ribbons=(tmp1_var, tmp1_var), line=:dash, color="red", label = "inverse KKT OLNE, distance to noisy observation data", xlabel="noise variance", size = (900,500),legend = :outerleft)
plot!(noise_level_list, tmp2,ribbons=(tmp2_var, tmp2_var),line=:dash, color="blue", label = "inverse KKT OLNE, distance to ground truth data")
plot!(noise_level_list, tmp3,ribbons=(tmp3_var, tmp3_var),line=:dash, color="orange", label = "inverse KKT OLNE, generalization loss")

plot!(noise_level_list, tmp4, color="red", label="Inverse FBNE, distance to observation data")
plot!(noise_level_list, tmp5, color="blue", label = "Inverse FBNE, ground truth loss")
plot!(noise_level_list, tmp6, color="orange", label="Inverse FBNE, generalization loss")


tmp14 = [mean(optim_loss_list_list[1][ii][1])[1] for ii in 1:num_noise_level]
tmp15 = [mean(list_ground_truth_loss[1][jj][1]) for jj in 1:num_noise_level]
tmp16 = [mean(list_generalization_loss[1][jj][1]) for jj in 1:num_noise_level]
tmp14_var = [var(optim_loss_list_list[1][ii][1]) for ii in 1:num_noise_level]
tmp15_var = [var(list_ground_truth_loss[1][jj][1]) for jj in 1:num_noise_level]
tmp16_var = [var(list_generalization_loss[1][jj][1]) for jj in 1:num_noise_level]
plot!(noise_level_list, tmp14,ribbons=(tmp14_var, tmp14_var), color="red", label="Inverse FBNE, distance to noisy observation data")
plot!(noise_level_list, tmp15,ribbons=(tmp15_var, tmp15_var), color="blue", label = "Inverse FBNE, distance to ground truth data")
plot!(noise_level_list, tmp16,ribbons=(tmp16_var, tmp16_var), color="orange", label="Inverse FBNE, generalization loss")

savefig("10_8_all.pdf")



plot(noise_level_list, tmp14,ribbons=(tmp14_var, tmp14_var), color="red", label="Inverse FBNE, distance to noisy observation data")
plot!(noise_level_list, tmp15,ribbons=(tmp15_var, tmp15_var), color="blue", label = "Inverse FBNE, distance to ground truth data")
plot!(noise_level_list, tmp16,ribbons=(tmp16_var, tmp16_var), color="orange", label="Inverse FBNE, generalization loss")
plot!(ylims=(0,10))


ttt = load("inv_dubins_0.1_server")
plot(noise_level_list, ttt["tmp14"], ribbons = (ttt["tmp14_var"], ttt["tmp14_var"]))
plot!(noise_level_list, ttt["tmp15"], ribbons = (ttt["tmp15_var"], ttt["tmp15_var"]))
# plot!(noise_level_list, ttt["tmp16"], ribbons = (ttt["tmp16_var"], ttt["tmp16_var"]))
plot!(noise_level_list, ttt["tmp16"])


# ----------------------------------------------------------------------------------------------------
# plot
t1 = load("GD_2_cars_no_control")
t2 = load("KKT_2_cars_no_control")

tmp1 = [mean(t1["inv_loss_list"][ii])[1] for ii in 1:num_noise_level]
tmp2 = [mean(t1["inv_ground_truth_loss_list"][ii])[1] for ii in 1:num_noise_level]
tmp3 = [mean(t1["inv_mean_generalization_loss_list"][ii])[1] for ii in 1:num_noise_level]
tmp1_var = [var(t1["inv_loss_list"][ii])[1] for ii in 1:num_noise_level]
tmp2_var = [var(t1["inv_ground_truth_loss_list"][ii])[1] for ii in 1:num_noise_level]
tmp3_var = [var(t1["inv_mean_generalization_loss_list"][ii])[1] for ii in 1:num_noise_level]
tmp14, tmp15, tmp16, tmp14_var, tmp15_var, tmp16_var = t2["tmp14"], t2["tmp15"], t2["tmp16"], t2["tmp14_var"], t2["tmp15_var"], t2["tmp16_var"]
plt=plot()
plt=plot(noise_level_list, tmp1, ribbons=(tmp1_var, tmp1_var),alpha=1, line=:dash,linewidth=3, color="red", label = "inverse KKT OLNE, distance to noisy observation data", xlabel="noise variance", size = (700,500),legend = :topright,ylims=(0,10))
plt=plot!(noise_level_list, tmp2,ribbons=(tmp2_var, tmp2_var),alpha=1,line=:dash,linewidth=3, color="blue", label = "inverse KKT OLNE, distance to ground truth data")
plt=plot!(noise_level_list, tmp3,ribbons=(tmp3_var, tmp3_var),alpha=1,line=:dash,linewidth=3, color="orange", label = "inverse KKT OLNE, generalization loss")
plt=plot!(noise_level_list, tmp14,ribbons=(tmp14_var, tmp14_var),alpha=1, color="red",linewidth=3, label="Inverse FBNE, distance to noisy observation data")
plt=plot!(noise_level_list, tmp15,ribbons=(tmp15_var, tmp15_var),alpha=1, color="blue",linewidth=3, label = "Inverse FBNE, distance to ground truth data")
plt=plot!(noise_level_list, tmp16,ribbons=(tmp16_var, tmp16_var),alpha=1, color="orange",linewidth=3, label="Inverse FBNE, generalization loss")

for ii in 1:10
    plt=scatter!(noise_level_list, [t1["inv_loss_list"][jj][ii][1] for jj in 1:num_noise_level], markershape=:x,alpha=0.1, color="red", label="")
    plt=scatter!(noise_level_list, [t1["inv_ground_truth_loss_list"][jj][ii][1] for jj in 1:num_noise_level], markershape=:x,alpha=0.1, color="blue", label="")
    plt=scatter!(noise_level_list, [t1["inv_mean_generalization_loss_list"][jj][ii][1] for jj in 1:num_noise_level], markershape=:x,alpha=0.1, color="orange", label="")
    plt=scatter!(noise_level_list, [t2["optim_loss_list_list"][1][jj][1][ii] for jj in 1:num_noise_level], markershape=:x,alpha=0.1, color="red", label="")
    plt=scatter!(noise_level_list, [t2["list_ground_truth_loss"][1][jj][1][ii] for jj in 1:num_noise_level], markershape=:x,alpha=0.1, color="red", label="")
    plt=scatter!(noise_level_list, [t2["list_generalization_loss"][1][jj][1][ii] for jj in 1:num_noise_level], markershape=:x,alpha=0.1, color="red", label="")
end
display(plt)

savefig("20_no_control_full.pdf")

savefig("20_no_control_partial.pdf")


# t1["noisy_expert_traj_list"][1][1][1].x
# obs_time_list = t2["obs_time_list"]
obs_time_list= [1,2,3,4,5,6,11,12,13,14,15,16,21,22,23,24,25,26,31,32,33,34,35,36]

obs_state_list = t2["obs_state_list"]

game_horizon = length(t1["expert_traj_list"][1].x)

plot_size = (1000,600)
noise = 9
subplt1 = plot([t1["expert_traj_list"][1].x[t][1] for t in 1:game_horizon], [t1["expert_traj_list"][1].x[t][2] for t in 1:game_horizon], color="red", label="player 1, ground truth, σ = $(t1["noise_level_list"][noise])", size = plot_size, xlabel="x", ylabel="y")
subplt1 = plot!([t1["expert_traj_list"][1].x[t][5] for t in 1:game_horizon], [t1["expert_traj_list"][1].x[t][6] for t in 1:game_horizon], color="blue",label="player 2, ground truth, σ = $(t1["noise_level_list"][noise])")
subplt1 = scatter!([t1["noisy_expert_traj_list"][1][noise][1].x[t][1] for t in 1:game_horizon], [t1["noisy_expert_traj_list"][1][noise][1].x[t][2] for t in 1:game_horizon], color="red",label="player 1, noisy observation full, σ = $(t1["noise_level_list"][noise])")
subplt1 = scatter!([t1["noisy_expert_traj_list"][1][noise][1].x[t][5] for t in 1:game_horizon], [t1["noisy_expert_traj_list"][1][noise][1].x[t][6] for t in 1:game_horizon], color="blue",label="player 2, noisy observation full, σ = $(t1["noise_level_list"][noise])")
display(subplt1)
noise = 9
subplt2 = plot([t1["expert_traj_list"][1].x[t][1] for t in 1:game_horizon], [t1["expert_traj_list"][1].x[t][2] for t in 1:game_horizon], color="red", label="player 1, ground truth, σ = $(t1["noise_level_list"][noise])", size = plot_size, xlabel="x", ylabel="y")
subplt2 = plot!([t1["expert_traj_list"][1].x[t][5] for t in 1:game_horizon], [t1["expert_traj_list"][1].x[t][6] for t in 1:game_horizon], color="blue",label="player 2, ground truth, σ = $(t1["noise_level_list"][noise])")
subplt2 = scatter!([t1["noisy_expert_traj_list"][1][noise][1].x[t][1] for t in obs_time_list], [t1["noisy_expert_traj_list"][1][noise][1].x[t][2] for t in obs_time_list], markershape=:x, color="red",label="player 1, noisy observation partial, σ = $(t1["noise_level_list"][noise])")
subplt2 = scatter!([t1["noisy_expert_traj_list"][1][noise][1].x[t][5] for t in obs_time_list], [t1["noisy_expert_traj_list"][1][noise][1].x[t][6] for t in obs_time_list], markershape=:x, color="blue",label="player 2, noisy observation partial, σ = $(t1["noise_level_list"][noise])")
display(subplt2)
noise = 19
subplt3 = plot([t1["expert_traj_list"][1].x[t][1] for t in 1:game_horizon], [t1["expert_traj_list"][1].x[t][2] for t in 1:game_horizon], color="red", label="player 1, ground truth, σ = $(t1["noise_level_list"][noise])", size = plot_size, xlabel="x", ylabel="y")
subplt3 = plot!([t1["expert_traj_list"][1].x[t][5] for t in 1:game_horizon], [t1["expert_traj_list"][1].x[t][6] for t in 1:game_horizon], color="blue",label="player 2, ground truth, σ = $(t1["noise_level_list"][noise])")
subplt3 = scatter!([t1["noisy_expert_traj_list"][1][noise][1].x[t][1] for t in 1:game_horizon], [t1["noisy_expert_traj_list"][1][noise][1].x[t][2] for t in 1:game_horizon], color="red",label="player 1, noisy observation full, σ = $(t1["noise_level_list"][noise])")
subplt3 = scatter!([t1["noisy_expert_traj_list"][1][noise][1].x[t][5] for t in 1:game_horizon], [t1["noisy_expert_traj_list"][1][noise][1].x[t][6] for t in 1:game_horizon], color="blue",label="player 2, noisy observation full, σ = $(t1["noise_level_list"][noise])")
display(subplt3)
noise=19
subplt4 = plot([t1["expert_traj_list"][1].x[t][1] for t in 1:game_horizon], [t1["expert_traj_list"][1].x[t][2] for t in 1:game_horizon], color="red", label="player 1, ground truth, σ = $(t1["noise_level_list"][noise])", size = plot_size, xlabel="x", ylabel="y")
subplt4 = plot!([t1["expert_traj_list"][1].x[t][5] for t in 1:game_horizon], [t1["expert_traj_list"][1].x[t][6] for t in 1:game_horizon], color="blue",label="player 2, ground truth, σ = $(t1["noise_level_list"][noise])")
subplt4 = scatter!([t1["noisy_expert_traj_list"][1][noise][1].x[t][1] for t in obs_time_list], [t1["noisy_expert_traj_list"][1][noise][1].x[t][2] for t in obs_time_list], markershape=:x, color="red",label="player 1, noisy observation partial, σ = $(t1["noise_level_list"][noise])")
subplt4 = scatter!([t1["noisy_expert_traj_list"][1][noise][1].x[t][5] for t in obs_time_list], [t1["noisy_expert_traj_list"][1][noise][1].x[t][6] for t in obs_time_list], markershape=:x, color="blue",label="player 2, noisy observation partial, σ = $(t1["noise_level_list"][noise])")
display(subplt4)

fullplt = plot(subplt1, subplt3, subplt2, subplt4, layout=(2,2))
display(fullplt)

savefig(fullplt, "visualization_noise_level.pdf")



t1 = load("GD_2cars_partial") # Oct. 1st
t2 = load("20_2cars_nocontrol_new") # Oct 1st




#---------------------------------------
# full:
t1=load("inv_dubins_0.5_gen_x0_compact")

plot(t1["noise_level_list"], [mean(t1["optim_loss_list_list"][1][noise][1][ii] for ii in 1:10) for noise in 1:length(t1["noise_level_list"])],color="blue", label="loss")
plot!(t1["noise_level_list"], [mean(t1["list_ground_truth_loss"][1][noise][1][ii] for ii in 1:10) for noise in 1:length(t1["noise_level_list"])], color="red", label="ground truth distance")
plot!(t1["noise_level_list"], [mean(t1["list_generalization_loss"][1][noise][1][ii] for ii in 1:10) for noise in 1:length(t1["noise_level_list"])], color="orange", label="generalization error")




t2=load("KKT_dubins_0.5_gen_x0_compact")
noise_level_list = 0.004:0.002:0.04
num_obs=10
num_noise_level = length(noise_level_list)
inv_mean_generalization_loss_list = t2["inv_mean_generalization_loss_list"]
inv_loss_list = t2["inv_loss_list"]
inv_ground_truth_loss_list = t2["inv_ground_truth_loss_list"]

plot_num_obs = 10
var1 = [var(inv_mean_generalization_loss_list[ii][jj][1] for jj in 1:plot_num_obs)[1] for ii in 1:num_noise_level]
var2 = [var(inv_loss_list[ii][jj][1] for jj in 1:plot_num_obs)[1] for ii in 1:num_noise_level]
var3 = [var(inv_ground_truth_loss_list[ii][jj][1] for jj in 1:plot_num_obs)[1] for ii in 1:num_noise_level]
plot(noise_level_list, [mean(inv_mean_generalization_loss_list[ii][jj][1] for jj in 1:plot_num_obs)[1] for ii in 1:num_noise_level], label="generalization error")
plot!(noise_level_list, [mean(inv_loss_list[ii][jj][1] for jj in 1:plot_num_obs)[1] for ii in 1:num_noise_level], ribbons=(var2,var2), label = "loss")
plot!(noise_level_list, [mean(inv_ground_truth_loss_list[ii][jj][1] for jj in 1:plot_num_obs)[1] for ii in 1:num_noise_level], ribbons=(var3,var3), label="ground truth")




# ------------------------------------------- Oct. 7th
t1=load("KKT_partial_2cars_x0_baobei")
t2 = load("GD_partial_2cars_x0_baobei")
tmp1 = [mean(t1["inv_loss_list"][ii])[1] for ii in 1:num_noise_level]
tmp2 = [mean(t1["inv_ground_truth_loss_list"][ii])[1] for ii in 1:num_noise_level]
tmp3 = [mean(t1["inv_mean_generalization_loss_list"][ii])[1] for ii in 1:num_noise_level]
tmp1_var = [var(t1["inv_loss_list"][ii])[1] for ii in 1:num_noise_level]
tmp2_var = [var(t1["inv_ground_truth_loss_list"][ii])[1] for ii in 1:num_noise_level]
tmp3_var = [var(t1["inv_mean_generalization_loss_list"][ii])[1] for ii in 1:num_noise_level]
tmp14, tmp15, tmp16, tmp14_var, tmp15_var, tmp16_var = t2["tmp14"], t2["tmp15"], t2["tmp16"], t2["tmp14_var"], t2["tmp15_var"], t2["tmp16_var"]

plt=plot()
plt=plot(noise_level_list, tmp1, ribbons=(tmp1_var, tmp1_var),alpha=1, line=:dash,linewidth=3, color="red", label = "inverse KKT OLNE, distance to noisy observation data", xlabel="noise variance", size = (700,500),legend = :topright,ylims=(0,5))
plt=plot!(noise_level_list, tmp2,ribbons=(tmp2_var, tmp2_var),alpha=1,line=:dash,linewidth=3, color="blue", label = "inverse KKT OLNE, distance to ground truth data")
plt=plot!(noise_level_list, tmp3,ribbons=(tmp3_var, tmp3_var),alpha=1,line=:dash,linewidth=3, color="orange", label = "inverse KKT OLNE, generalization loss")
plt=plot!(noise_level_list, tmp14,ribbons=(tmp14_var, tmp14_var),alpha=1, color="red",linewidth=3, label="Inverse FBNE, distance to noisy observation data")
plt=plot!(noise_level_list, tmp15,ribbons=(tmp15_var, tmp15_var),alpha=1, color="blue",linewidth=3, label = "Inverse FBNE, distance to ground truth data")
plt=plot!(noise_level_list, tmp16,ribbons=(tmp16_var, tmp16_var),alpha=1, color="orange",linewidth=3, label="Inverse FBNE, generalization loss")

savefig("latest_partial_x0.pdf")
savefig("latest_full_x0.pdf")



# ------------------------------------ Oct. 18
t1=load("Indi_var_KKT_clean_3cars_full2022-10-18T05:52:43.663")
t2=load("Indy_baobei_GD_3car_full_x02022-10-18T05:36:39.382")
noise_level_list=0.004:0.008:0.04
num_noise_level = length(noise_level_list)
tmp1 = [mean(t1["inv_loss_list"][ii])[1] for ii in 1:num_noise_level]
tmp2 = [mean(t1["inv_ground_truth_loss_list"][ii])[1] for ii in 1:num_noise_level]
tmp3 = [mean(t1["inv_mean_generalization_loss_list"][ii])[1] for ii in 1:num_noise_level]
tmp1_var = [var(t1["inv_loss_list"][ii])[1] for ii in 1:num_noise_level]
tmp2_var = [var(t1["inv_ground_truth_loss_list"][ii])[1] for ii in 1:num_noise_level]
tmp3_var = [var(t1["inv_mean_generalization_loss_list"][ii])[1] for ii in 1:num_noise_level]
tmp14, tmp15, tmp16, tmp14_var, tmp15_var, tmp16_var = t2["tmp14"], t2["tmp15"], t2["tmp16"], t2["tmp14_var"], t2["tmp15_var"], t2["tmp16_var"]

plt=plot()
plt=plot(noise_level_list, tmp1, ribbons=(tmp1_var, tmp1_var),alpha=1, line=:dash,linewidth=3, color="red", label = "inverse KKT OLNE, L2 regularized loss", xlabel="noise variance", size = (700,500),legend = :topright,ylims=(0,5))
plt=plot!(noise_level_list, tmp2,ribbons=(tmp2_var, tmp2_var),alpha=1,line=:dash,linewidth=3, color="blue", label = "inverse KKT OLNE, distance to ground truth data")
plt=plot!(noise_level_list, tmp3,ribbons=(tmp3_var, tmp3_var),alpha=1,line=:dash,linewidth=3, color="orange", label = "inverse KKT OLNE, generalization loss")
plt=plot!(noise_level_list, tmp14,ribbons=(tmp14_var, tmp14_var),alpha=1, color="red",linewidth=3, label="Inverse FBNE, L2 regularized loss")
plt=plot!(noise_level_list, tmp15,ribbons=(tmp15_var, tmp15_var),alpha=1, color="blue",linewidth=3, label = "Inverse FBNE, distance to ground truth data")
plt=plot!(noise_level_list, tmp16,ribbons=(tmp16_var, tmp16_var),alpha=1, color="orange",linewidth=3, label="Inverse FBNE, generalization loss")
plt=plot!(ylim=(0,8), size=(400,400))

savefig("Indy_partial_x0.pdf")
savefig("Infy_full_x0.pdf")


# -------------------------------------- Oct. 22
# 2-car
t1=load("1012_baobei_GD_2car_full_x02022-10-14T23:19:41.847")
t2=load("")
t3=load("")
t4=load("KKT_partial_2cars_x0_baobei")



