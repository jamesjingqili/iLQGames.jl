# full observation : obs_state_list=1:13

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
# x0 = SVector(0.0, 3, pi/2, 2,       0.3, 0, pi/2, 2,      0.7, 2,pi/2,1,                   0.2)
# platonning
x0 = SVector(0, 2, pi/2, 1,       1, 0, pi/2, 1,   0.5, 1,pi/2,1,                   0.2)
costs = (FunctionPlayerCost((g,x,u,t) -> ( 8*(x[5]-x[13])^2   +4*(x[3]-pi/2)^2  +2*(x[4]-1)^2       +2*(u[1]^2 + u[2]^2)    )),
         FunctionPlayerCost((g,x,u,t) -> ( 4*(x[5]-x[1])^2    +4*(x[7]-pi/2)^2  +4*(x[8]-1)^2       -log((x[5]-x[9])^2+(x[6]-x[10])^2)    +2*(u[3]^2+u[4]^2)    )),
         FunctionPlayerCost((g,x,u,t) -> ( 2*(x[9]-x0[9])^2   + 2*(u[5]^2+u[6]^2)  ))
    )
player_inputs = (SVector(1,2), SVector(3,4), SVector(5,6))
g = GeneralGame(game_horizon, player_inputs, dynamics, costs)
solver1 = iLQSolver(g, max_scale_backtrack=5, max_elwise_diff_step=Inf, equilibrium_type="OLNE_costate")
c1, expert_traj1, strategies1 = solve(g, solver1, x0)
solver2 = iLQSolver(g, max_scale_backtrack=5, max_elwise_diff_step=Inf, equilibrium_type="FBNE_costate")
c2, expert_traj2, strategies2 = solve(g, solver2, x0)

θ_true = [0, 8, 4, 4]
obs_x_FB = transpose(mapreduce(permutedims, vcat, Vector([Vector(expert_traj2.x[t]) for t in 1:g.h])))
obs_u_FB = transpose(mapreduce(permutedims, vcat, Vector([Vector(expert_traj2.u[t]) for t in 1:g.h])))
obs_x_OL = transpose(mapreduce(permutedims, vcat, Vector([Vector(expert_traj1.x[t]) for t in 1:g.h])))
obs_u_OL = transpose(mapreduce(permutedims, vcat, Vector([Vector(expert_traj1.u[t]) for t in 1:g.h])))

# noisy_obs_x_OL = transpose(mapreduce(permutedims, vcat, Vector([Vector(noisy_expert_traj_list[1][1][1].x[t]) for t in 1:g.h])))
# noisy_obs_u_OL = transpose(mapreduce(permutedims, vcat, Vector([Vector(noisy_expert_traj_list[1][1][1].u[t]) for t in 1:g.h])))

function parameterized_cost(θ::Vector)
costs = (FunctionPlayerCost((g,x,u,t) -> ( θ[1]*x[1]^2 +θ[2]*(x[5]-x[13])^2   +4*(x[3]-pi/2)^2  +2*(x[4]-1)^2       +2*(u[1]^2 + u[2]^2)    )),
         FunctionPlayerCost((g,x,u,t) -> ( θ[3]*(x[5]-x[1])^2    +4*(x[7]-pi/2)^2  +θ[4]*(x[8]-1)^2       -log((x[5]-x[9])^2+(x[6]-x[10])^2)    +2*(u[3]^2+u[4]^2)    )),
         FunctionPlayerCost((g,x,u,t) -> ( 2*(x[9]-x0[9])^2   + 2*(u[5]^2+u[6]^2)  ))
    )
    return costs
end

include("../examples/cars3_def_2_KKT_x0.jl")
function two_level_inv_KKT(obs_x, θ₀, obs_time_list, obs_state_list)
    # first level, solve a feasible dynamics point
    feasible_sol = level_1_KKT_x0(obs_x, obs_time_list, obs_state_list);
    # second level, solver a good θ
    overall_sol = level_2_KKT_x0(feasible_sol[1],feasible_sol[2], obs_x, θ₀, obs_time_list, obs_state_list)
    return overall_sol
end
inv_sol=two_level_inv_KKT(obs_x_FB, 4*ones(4), 1:game_horizon-1, 1:nx)

solution_summary(inv_sol[4])
num_clean_traj = 1
x0_set = [x0 for ii in 1:num_clean_traj]
expert_traj_list, c_expert = generate_expert_traj(g, solver2, x0_set, num_clean_traj)
if sum([c_expert[ii]==false for ii in 1:length(c_expert)]) >0
    @warn "regenerate expert demonstrations because some of the expert demonstration not converged!!!"
end
game = g
solver = solver2
# The below: generate random expert trajectories
num_obs = 10
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
θ₀ = 4*ones(4);
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
obs_time_list = [1,2,3,4,5,6,7,8,9,10,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39]
obs_state_list = [1,2,3,5,6,7, 9, 10, 11]
obs_control_list=[]
# obs_time_list = 1:game_horizon-1
# obs_state_list = 1:nx
# obs_control_list = 1:nu
index_value = 1
for noise in 1:length(noise_level_list)
    for ii in 1:num_obs
        tmp_expert_traj_x = noisy_expert_traj_list[index_value][noise][ii].x
        tmp_expert_traj_u = noisy_expert_traj_list[index_value][noise][ii].u
        
        tmp_obs_x = transpose(mapreduce(permutedims, vcat, Vector([Vector(tmp_expert_traj_x[t]) for t in 1:game.h])))
        # tmp_obs_u = transpose(mapreduce(permutedims, vcat, Vector([Vector(tmp_expert_traj_u[t]) for t in 1:game.h])))
        # tmp_inv_traj_x, tmp_inv_traj_u, tmp_inv_sol, tmp_inv_model = KKT_highway_inverse_game_solve(tmp_obs_x[:,2:end], tmp_obs_u, θ₀, x0_set[index], obs_time_list, obs_state_list, obs_control_list)
        tmp_sol = two_level_inv_KKT(tmp_obs_x, θ₀, obs_time_list, obs_state_list)
        tmp_inv_traj_x, tmp_inv_traj_u, tmp_inv_sol, tmp_inv_model = tmp_sol[1], tmp_sol[2], tmp_sol[3], tmp_sol[4];
        tmp_inv_loss = objective_value(tmp_inv_model)
        println("The $(ii)-th observation of $(noise)-th noise level")
        # solution_summary(tmp_inv_model)
        tmp_ground_truth_loss_value, tmp_ground_truth_computed_traj, _, _=loss(tmp_inv_sol, iLQGames.dynamics(game), "FBNE_costate", expert_traj_list[index_value], false, false, [], [], 1:game_horizon-1, 1:12, 1:nu, false) 
        # @infiltrate
        # tmp_test_sol = [[] for jj in 1:num_test]
        tmp_test_loss_value = zeros(num_test)
        for jj in 1:num_test
            # @infiltrate
            tmp_test_loss_value[jj], _,_,_ = loss(tmp_inv_sol, iLQGames.dynamics(game), "FBNE_costate", test_expert_traj_list[jj], false, false, [],[],1:game_horizon-1, 1:12, 1:nu,false)
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
        if termination_status(tmp_inv_model)==NUMERICAL_ERROR
            println("Failed at $(noise), $(ii)")
        end
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



jldsave("baobei_KKT_clean_3cars_partial$(Dates.now())"; game_horizon, inv_mean_generalization_loss_list, inv_var_generalization_loss_list, inv_sol_list,
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

plot(noise_level_list, [mean(inv_mean_generalization_loss_list[ii][jj][1] for jj in 1:num_obs)[1] for ii in 1:num_noise_level],ribbons=(var1,var1), label="generalization error")
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




#-------------------- Oct. 15
var1 = [var(inv_mean_generalization_loss_list[ii][jj][1] for jj in 1:num_obs)[1] for ii in 1:num_noise_level]
var2 = [var(inv_loss_list[ii][jj][1] for jj in 1:num_obs)[1] for ii in 1:num_noise_level]
var3 = [var(inv_ground_truth_loss_list[ii][jj][1] for jj in 1:num_obs)[1] for ii in 1:num_noise_level]

plot(noise_level_list, [mean(inv_mean_generalization_loss_list[ii][jj][1] for jj in 1:num_obs)[1] for ii in 1:num_noise_level],ribbons=(var1,var1), label="generalization error", xlabel="noise level")
plot!(noise_level_list, [mean(inv_loss_list[ii][jj][1] for jj in 1:num_obs)[1] for ii in 1:num_noise_level], ribbons=(var2,var2), label = "loss")
plot!(noise_level_list, [mean(inv_ground_truth_loss_list[ii][jj][1] for jj in 1:num_obs)[1] for ii in 1:num_noise_level], ribbons=(var3,var3), label="ground truth error")



plot(obs_x_OL[1,:], obs_x_OL[2,:],color="red")
plot!(obs_x_OL[5,:], obs_x_OL[6,:], color="blue")
plot!(obs_x_OL[9,:], obs_x_OL[10,:], color="black")
plot!(obs_x_FB[1,:], obs_x_FB[2,:],color="red", linestyle=:dash)
plot!(obs_x_FB[5,:], obs_x_FB[6,:], color="blue",linestyle=:dash)
plot!(obs_x_FB[9,:], obs_x_FB[10,:], color="black",linestyle=:dash)



