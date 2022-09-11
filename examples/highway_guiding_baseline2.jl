using Distributed
@everywhere begin
    using Pkg
    Pkg.activate("../")

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
costs = (FunctionPlayerCost((g, x, u, t) -> ( 10*(x[5]-1)^2 + 2*(x[4]-1)^2 + u[1]^2 + u[2]^2 - 0*((x[1]-x[5])^2 + (x[2]-x[6])^2))),
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
    costs = (FunctionPlayerCost((g, x, u, t) -> ( θ[1]*(x[5]-θ[4])^2 + θ[2]*(x[4]-1)^2 + θ[3]*(u[1]^2 + u[2]^2) - 0*((x[1]-x[5])^2 + (x[2]-x[6])^2))),
             FunctionPlayerCost((g, x, u, t) -> (  θ[5]*(x[5] - x[1])^2 + θ[6]*(x[8]-1)^2 + θ[7]*(u[3]^2 + u[4]^2) - 0*((x[1]-x[5])^2 + (x[2]-x[6])^2))))
    return costs
end

end

# ------------------------------------------------------------------------------------------------------------------------------------------
"Experiment 2: With noise. Scatter plot"
# X: noise variance
# Y1: state prediction loss, mean and variance
# Y2: generalization loss, mean and variance

GD_iter_num = 300
num_clean_traj = 1
noise_level_list = 0:0.005:0.0
num_noise_level = length(noise_level_list)
num_obs = 10
games = []
x0_set = [x0+0.5*(rand(8)-0.5*ones(8)) for ii in 1:num_clean_traj]
# θ_true = [2.0;2.0;1.0;2.0;2.0;1.0;0.0;0.0]

c_expert,expert_traj_list,expert_equi_list=generate_traj(g,x0_set,parameterized_cost,["FBNE_costate","OLNE_costate"])
noisy_expert_traj_list = [[[zero(SystemTrajectory, g) for kk in 1:num_obs] for jj in 1:num_noise_level] for ii in 1:num_clean_traj]

Threads.@threads for ii in 1:num_clean_traj
    for jj in 1:num_noise_level
        tmp = generate_noisy_observation(nx, nu, g, expert_traj_list[ii], noise_level_list[jj], num_obs)
        for kk in 1:num_obs
            for t in 1:g.h
                noisy_expert_traj_list[ii][jj][kk].x[t] = tmp[kk].x[t]
                noisy_expert_traj_list[ii][jj][kk].u[t] = tmp[kk].u[t]
            end
        end
    end
end


conv_table_list = [[[] for jj in 1:num_noise_level] for ii in 1:num_clean_traj]
sol_table_list = deepcopy(conv_table_list)
loss_table_list = deepcopy(conv_table_list)
grad_table_list = deepcopy(conv_table_list)
equi_table_list = deepcopy(conv_table_list)
iter_table_list = deepcopy(conv_table_list)
comp_time_table_list = deepcopy(conv_table_list)

θ_list_list = deepcopy(conv_table_list)
index_list_list = deepcopy(conv_table_list)
optim_loss_list_list = deepcopy(conv_table_list)


θ₀ = ones(7)

Threads.@threads for ii in 1:num_clean_traj
    for jj in 1:num_noise_level
        conv_table,sol_table,loss_table,grad_table,equi_table,iter_table,comp_time_table=run_experiments_with_baselines(g,θ₀,[x0_set[ii] for kk in 1:num_obs], 
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
for index in 1:3
    for jj in 1:num_noise_level
         
        mean_predictions[index][jj] = mean(reduce(vcat,[optim_loss_list_list[ii][jj][1][index] for ii in 1:num_clean_traj]))
        variance_predictions[index][jj] = var(reduce(vcat,[optim_loss_list_list[ii][jj][1][index] for ii in 1:num_clean_traj]))
    end
end




