using Distributed
@everywhere begin
    using Pkg
    Pkg.activate("../")

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
gif(anim1, "LQ_OL.gif", fps = 10)




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
@everywhere begin
function parameterized_cost(θ::Vector)
    costs = (FunctionPlayerCost((g, x, u, t) -> ( θ[1]*(x[1]^2 + x[2]^2) + θ[2]*(x[3]^2 + x[4]^2) + θ[3]*(u[1]^2 + u[2]^2))),
             FunctionPlayerCost((g, x, u, t) -> ( 0*(x[3]^2 + x[4]^2) + θ[4]*((x[1]-x[3])^2 + (x[2]-x[4])^2) + θ[5]*(u[3]^2 + u[4]^2))))
    return costs
end
end

#----------------------------------------------------------------------------------------------------------------------------------

include("../src/experiment_utils.jl") # NOTICE!! Many functions are defined there.


GD_iter_num = 200
n_data = 1
θ_true = [0.0;2.0;1.0;0.0;2.0;1.0;]

θ₀ = θ_true
# 
x0_set = [x0+0*(rand(4)-0.5*ones(4)) for ii in 1:n_data]
c_expert,expert_traj_list,expert_equi_list=generate_traj(g,x0_set,parameterized_cost,["FBNE_costate","OLNE_costate"])


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

# (1) filter out the converged example here:
num_converged = sum(conv_table[1].*conv_table[2].*conv_table[3])
time_list = [[0.0 for ii in 1:num_converged] for index in 1:3]
time_list_index = zeros(num_converged)
jj = 1
for ii in 1:n_data
    if conv_table[1][ii].*conv_table[2][ii].*conv_table[3][ii] == 1
        time_list[1][jj] = comp_time_table[1][ii]
        time_list[2][jj] = comp_time_table[2][ii]
        time_list[3][jj] = comp_time_table[3][ii]
        time_list_index[jj] = ii
        jj = jj+1
    end
end

histogram(time_list[1], bins = (0:1:80), alpha=0.5, label = "Bayesian")
histogram!(time_list[2], bins = (0:1:80), alpha=0.5, label = "pure FB")
histogram!(time_list[3], bins = (0:1:80), alpha=0.5, label = "pure OL")
savefig("LQ_comp_time_table1.pdf")


bins = 0:2:100.0
transformed_comp_time_table = deepcopy(comp_time_table)
for index in 1:3
    for ii in n_data
        if comp_time_table[index][ii] > bins[end]
            transformed_comp_time_table[index][ii] = bins[end]-1e-6
        end
    end
end


histogram(transformed_comp_time_table[1], bins=bins, alpha=0.5, label = "Bayesian")
histogram!(transformed_comp_time_table[2], bins=bins, alpha=0.5, label = "pure FB")
histogram!(transformed_comp_time_table[3], bins=bins, alpha=0.5, label = "pure OL")
savefig("LQ_comp_time.pdf")

histogram(comp_time_table[1], bins = (0:10:300), alpha=0.5, label = "Bayesian")
histogram!(comp_time_table[2], bins = (0:10:300), alpha=0.5, label = "pure FB")
histogram!(comp_time_table[3], bins = (0:10:300), alpha=0.5, label = "pure OL")
savefig("LQ_comp_time_table.pdf")


#----------------------------------------------------------------------------------------------------------------------------------

"Experiment 2: With noise. Scatter plot"
# X: noise variance
# Y1: state prediction loss, mean and variance
# Y2: generalization loss, mean and variance

include("experiment_utils.jl")

GD_iter_num = 200
num_clean_traj = 10
noise_level_list = 0:0.01:0.04
num_noise_level = length(noise_level_list)
num_obs = 10
x0 = SVector(0, 1, 1,1)
x0_set = [x0+0.5*(rand(4)-0.5*ones(4)) for ii in 1:num_clean_traj]
θ_true = [0.0;2.0;1.0;0.0; 2.0;1.0]

nx, nu, ΔT, game_horizon = 4, 4, 0.1, 40
costs = (FunctionPlayerCost((g, x, u, t) -> ( 2*(x[3])^2 + 2*(x[4])^2 + u[1]^2 + u[2]^2)),
         FunctionPlayerCost((g, x, u, t) -> ( 2*(x[1]-x[3])^2 + 2*(x[2]-x[4])^2 + u[3]^2 + u[4]^2)))
player_inputs = (SVector(1,2), SVector(3,4))
games, expert_traj_list, expert_equi_list, solvers, c_expert = generate_LQ_problem_and_traj(game_horizon, ΔT, player_inputs, costs, 
    x0_set, ["FBNE_costate","OLNE_costate"], num_clean_traj)
if sum([c_expert[ii]==false for ii in 1:length(c_expert)]) >0
    @warn "regenerate expert demonstrations because some of the expert demonstration not converged!!!"
end
# c_expert,expert_traj_list,expert_equi_list=generate_traj(g,x0_set,parameterized_cost,["FBNE_costate","OLNE_costate"])
noisy_expert_traj_list = [[[zero(SystemTrajectory, games[1]) for kk in 1:num_obs] for jj in 1:num_noise_level] for ii in 1:num_clean_traj];

for ii in 1:num_clean_traj
    for jj in 1:num_noise_level
        tmp = generate_noisy_observation(nx, nu, games[ii], expert_traj_list[ii], noise_level_list[jj], num_obs)
        for kk in 1:num_obs
            for t in 1:game_horizon
                noisy_expert_traj_list[ii][jj][kk].x[t] = tmp[kk].x[t]
                noisy_expert_traj_list[ii][jj][kk].u[t] = tmp[kk].u[t]
            end
        end
    end
end


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


θ₀ = ones(5);

for ii in 1:num_clean_traj
    for jj in 1:num_noise_level
        conv_table,sol_table,loss_table,grad_table,equi_table,iter_table,comp_time_table=run_experiments_with_baselines(games[ii],θ₀,[x0_set[ii] for kk in 1:num_obs], 
                                                                                                noisy_expert_traj_list[ii][jj], parameterized_cost, GD_iter_num)
        θ_list, index_list, optim_loss_list = get_the_best_possible_reward_estimate([x0_set[ii] for kk in 1:num_obs], ["FBNE_costate","OLNE_costate"], sol_table, loss_table, equi_table)
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
    end
end

# ii -> nominal traj, jj -> noise level, index -> information pattern
mean_predictions = [zeros(num_noise_level) for index in 1:3]
variance_predictions = [zeros(num_noise_level) for index in 1:3]
for index in 1:3 # three information patterns
    for jj in 1:num_noise_level
        mean_predictions[index][jj] = mean(reduce(vcat,[optim_loss_list_list[ii][jj][1][index] for ii in 1:num_clean_traj]))
        variance_predictions[index][jj] = var(reduce(vcat,[optim_loss_list_list[ii][jj][1][index] for ii in 1:num_clean_traj]))
    end
end

# --------------------------------------------------------------------------------------------------------------------------------
# average computation time: FB, OL, Joint

nx, nu, ΔT, game_horizon = 4, 4, 0.1, 40
costs = (FunctionPlayerCost((g, x, u, t) -> ( 2*(x[3])^2 + 2*(x[4])^2 + u[1]^2 + u[2]^2)),
         FunctionPlayerCost((g, x, u, t) -> ( 2*(x[1]-x[3])^2 + 2*(x[2]-x[4])^2 + u[3]^2 + u[4]^2)))
# indices of inputs that each player controls
player_inputs = (SVector(1,2), SVector(3,4))

num_LQs = 10
num_obs = 10

games, expert_trajs, expert_equi, solvers, converged_expert = generate_LQ_problem_and_traj(game_horizon, ΔT, player_inputs, costs, [SVector{4}(0.0,1.0,1.0,1.0)], ["FBNE_costate","OLNE_costate"], 10, 10)



# x1_OL, y1_OL = [expert_trajs[1].x[i][1] for i in 1:game_horizon], [expert_trajs[1].x[i][2] for i in 1:game_horizon];
# x2_OL, y2_OL = [expert_trajs[1].x[i][3] for i in 1:game_horizon], [expert_trajs[1].x[i][4] for i in 1:game_horizon];
# anim1 = @animate for i in 1:game_horizon
#     plot([x1_OL[i], x1_OL[i]], [y1_OL[i], y1_OL[i]], markershape = :square, label = "player 1, OL", xlims = (-1, 2), ylims = (-1, 2))
#     plot!([x2_OL[i], x2_OL[i]], [y2_OL[i], y2_OL[i]], markershape = :square, label = "player 2, OL", xlims = (-1, 2), ylims = (-1, 2))
#     plot!([0], seriestype = "vline", color = "black", label = "")
#     plot!([1], seriestype = "vline", color = "black", label = "") 
# end
# gif(anim1, "test_LQ_OL.gif", fps = 10)















"Accelerated GD?"
