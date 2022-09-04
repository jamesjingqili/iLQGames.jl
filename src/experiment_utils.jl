function loss(θ, equilibrium_type, expert_traj, gradient_mode = true, specified_solver_and_traj = false, 
                nominal_solver=[], nominal_traj=[]) 
    x0 = first(expert_traj.x)
    if gradient_mode == false
        nominal_game = GeneralGame(game_horizon, player_inputs, dynamics, parameterized_cost(ForwardDiff.value.(θ)))
        nominal_solver = iLQSolver(nominal_game, max_scale_backtrack=10, max_elwise_diff_step=Inf, equilibrium_type=equilibrium_type)
        nominal_converged, nominal_traj, nominal_strategies = solve(nominal_game, nominal_solver, x0)
        loss_value = norm(nominal_traj.x - expert_traj.x)^2 + norm(nominal_traj.u-expert_traj.u)^2
        # @infiltrate
        return loss_value, nominal_traj, nominal_strategies, nominal_solver
    else
        if specified_solver_and_traj == false
            nominal_game = GeneralGame(game_horizon, player_inputs, dynamics, parameterized_cost(ForwardDiff.value.(θ)))
            nominal_solver = iLQSolver(nominal_game, max_scale_backtrack=10, max_elwise_diff_step=Inf, equilibrium_type=equilibrium_type)
            nominal_converged, nominal_traj, nominal_strategies = solve(nominal_game, nominal_solver, x0)
        end
        costs = parameterized_cost(θ)
        game = GeneralGame(game_horizon, player_inputs, dynamics, costs)

        lqg = Differentiable_Solvers.lq_approximation(game, nominal_traj, nominal_solver)
        if equilibrium_type=="OLNE_KKT" || equilibrium_type=="OLNE_costate" || equilibrium_type=="OLNE"
            traj = Differentiable_Solvers.trajectory(x0, game, Differentiable_Solvers.solve_lq_game_OLNE(lqg), nominal_traj)
        elseif equilibrium_type=="FBNE_KKT" || equilibrium_type=="FBNE_costate" || equilibrium_type=="FBNE"
            traj = Differentiable_Solvers.trajectory(x0, game, Differentiable_Solvers.solve_lq_game_FBNE(lqg), nominal_traj)
        else
            @warn "equilibrium_type is wrong!"
        end
        loss_value = norm(traj.x - expert_traj.x)^2 + norm(traj.u - expert_traj.u)^2
        return loss_value
    end
end

function inverse_game_gradient_descent(θ::Vector, g::GeneralGame, expert_traj::SystemTrajectory, x0::SVector, 
                                        max_GD_iteration_num::Int, parameterized_cost, equilibrium_type=[], Bayesian_belief_update=false, 
                                        specify_current_loss_and_solver=false, current_loss=[], current_traj=[], current_solver=[])
    α, θ_next, new_loss, new_traj, new_solver = 1.0, θ, 0.0, zero(SystemTrajectory, g), current_solver
    if Bayesian_belief_update==true
        equilibrium_type = inverse_game_update_belief(θ, g, expert_traj, x0, parameterized_cost, "FBNE_costate", "OLNE_costate")
    end
    if specify_current_loss_and_solver == false
        current_loss, current_traj, current_str, current_solver = loss(θ, equilibrium_type, expert_traj, false)
    end
    # gradient_value = inverse_game_gradient(current_loss, θ, g, expert_traj, x0, parameterized_cost, equilibrium_type)
    gradient_value = ForwardDiff.gradient(x -> loss(x, equilibrium_type, expert_traj, true, true, current_solver, current_traj), θ)
    for iter in 1:max_GD_iteration_num
        θ_next = θ-α*gradient_value
        while minimum(θ_next)<=-0.1
            α = α*0.5^2
            θ_next = θ-α*gradient_value
        end
        new_loss, new_traj, new_str, new_solver = loss(θ_next, equilibrium_type, expert_traj, false)
        if new_loss < current_loss
            println("Inverse Game Line Search Step Size: ", α)
            return θ_next, new_loss, gradient_value, equilibrium_type, new_traj, new_solver
            break
        end
        α = α*0.5
        println("inverse game line search not well")
    end
    return θ_next, new_loss, gradient_value, equilibrium_type, new_traj, new_solver
end


function objective_inference(x0, θ, expert_traj, g, max_GD_iteration_num, equilibrium_type=[], 
                            Bayesian_update=false)
    θ_dim = length(θ)
    sol = [zeros(θ_dim) for iter in 1:max_GD_iteration_num+1]
    sol[1] = θ
    loss_values = zeros(max_GD_iteration_num+1)
    loss_values[1],_,_ = inverse_game_loss(sol[1], g, expert_traj, x0, parameterized_cost, equilibrium_type)
    gradient = [zeros(θ_dim) for iter in 1:max_GD_iteration_num]
    equilibrium_type_list = ["" for iter in 1:max_GD_iteration_num]
    converged = false
    for iter in 1:max_GD_iteration_num
        sol[iter+1], loss_values[iter+1], gradient[iter], equilibrium_type_list[iter] = inverse_game_gradient_descent(sol[iter], 
                                                                                g, expert_traj, x0, 10, 
                                                                                parameterized_cost, equilibrium_type, Bayesian_update)
        println("iteration: ", iter)
        println("current_loss: ", loss_values[iter+1])
        println("equilibrium_type: ", equilibrium_type_list[iter])
        println("Current solution: ", sol[iter+1])
        if loss_values[iter+1]<0.1
            converged = true
            break
        end
    end
    return converged, sol, loss_values, gradient, equilibrium_type_list
end


# θ represents the initialization in gradient descent
function run_experiments_with_baselines(g, θ, x0_set, expert_traj_list, parameterized_cost, 
                                                max_GD_iteration_num, Bayesian_update=true,
                                                all_equilibrium_types = ["FBNE_costate","OLNE_costate"])
    # In the returned table, the rows coresponding to Bayesian, FB, OL
    n_data = length(x0_set)
    n_equi_types = length(all_equilibrium_types)
    sol_table  = [[[] for jj in 1:n_data] for ii in 1:n_equi_types+1]
    grad_table = [[[] for jj in 1:n_data] for ii in 1:n_equi_types+1]
    equi_table = [[[] for jj in 1:n_data] for ii in 1:n_equi_types+1]
    comp_time_table = [[[] for jj in 1:n_data] for ii in 1:n_equi_types+1]
    conv_table = [[false for jj in 1:n_data] for ii in 1:n_equi_types+1] # converged_table
    loss_table = [[[] for jj in 1:n_data] for ii in 1:n_equi_types+1]
    total_iter_table = zeros(1+n_equi_types, n_data)
    @distributed for iter in 1:n_data
        x0 = x0_set[iter]
        expert_traj = expert_traj_list[iter]
        conv_table[1][iter], sol_table[1][iter], loss_table[1][iter], grad_table[1][iter], equi_table[1][iter]=objective_inference(x0,
                                                                        θ,expert_traj,g,max_GD_iteration_num,"FBNE_costate", true)
        total_iter_table[1,iter] = iterations_taken_to_converge(equi_table[1][iter])
        for index in 1:n_equi_types
            conv_table[1+index][iter], sol_table[1+index][iter], loss_table[1+index][iter], grad_table[1+index][iter], equi_table[1+index][iter]=objective_inference(x0,
                                                                        θ,expert_traj,g,max_GD_iteration_num, all_equilibrium_types[index], false)
            total_iter_table[1+index,iter] = iterations_taken_to_converge(equi_table[1+index][iter])
        end
    end
    return conv_table, sol_table, loss_table, grad_table, equi_table, total_iter_table, comp_time_table
end


# Given θ and equilibrium type, for each initial condition in x0_set, compute prediction loss
function generalization_loss(g, θ, x0_set, expert_traj_list, parameterized_cost, equilibrium_type_list)
    num_samples = length(x0_set)
    loss_list = zeros(num_samples)
    traj_list = [zero(SystemTrajectory, g) for ii in 1:num_samples]

    @distributed for iter in 1:length(x0_set)
        loss_list[iter], tmp_traj, _, _ = loss(θ, equilibrium_type_list[iter], expert_traj_list[iter], false)
        for t in 1:g.h
            traj_list[iter].x[t] = tmp_traj.x[t]
            traj_list[iter].u[t] = tmp_traj.u[t]
        end
    end
    return loss_list, traj_list
end


# generate expert trajectories for initial conditions in x0_set
function generate_traj(g, θ, x0_set, parameterized_cost, equilibrium_type_list )
    n_data = length(x0_set)
    conv = [false for ii in 1:n_data]
    expert_traj_list = [zero(SystemTrajectory, g) for ii in 1:n_data]
    expert_equi_list = ["" for ii in 1:n_data]
    @distributed for item in 1:n_data
        if rand(1)[1]>0.5
            expert_equi_list[item] = equilibrium_type_list[1]
        else
            expert_equi_list[item] = equilibrium_type_list[2]
        end
        solver = iLQSolver(g, max_scale_backtrack=10, max_elwise_diff_step=Inf, 
                            equilibrium_type=expert_equi_list[item])
        conv[item], tmp, _ = solve(g, solver, x0_set[item])
        for t in 1:g.h
            expert_traj_list[item].x[t] = tmp.x[t]
            expert_traj_list[item].u[t] = tmp.u[t]
        end
    end
    return conv, expert_traj_list, expert_equi_list
end

function generate_noisy_observation(nx, nu, g, expert_traj, noise_level, number_of_perturbed_traj_needed)
    # nx is the dimension of the states
    # nu is the dimension of the actions
    # g is the game
    # g.h represents the horizon
    perturbed_trajectories_list = [zero(SystemTrajectory, g) for ii in 1:number_of_perturbed_traj_needed]
    for ii in 1:number_of_perturbed_traj_needed
        for t in 1:g.h
            perturbed_trajectories_list[ii].x[t] = expert_traj.x[t] + rand(Normal(0, noise_level), nx)
            perturbed_trajectories_list[ii].u[t] = expert_traj.u[t] + rand(Normal(0, noise_level), nu)
        end
    end
    return perturbed_trajectories_list
end


function iterations_taken_to_converge(equi_list)
    return sum(equi_list[ii]!="" for ii in 1:length(equi_list))
end






# Get the best possible reward estimate
# function get_the_best_possible_reward_estimate(x0_set, all_equilibrium_types, sol_table, loss_table, equi_list)
#     n_data = length(x0_set)
#     n_equi_types = length(all_equilibrium_types)
#     for index in 1:n_equi_types+1
#         for ii in 1:n_data
#             if minimum(loss_table[index][ii])==0.0
#                 index_list[index][ii] = index[ii][iterations_taken_to_converge(equi_list)
#                 θ_list[index][ii] = sol[index_list[index][ii]]
#             else
#                 index_list[index][ii] = length(sol_table[index][ii])
#                 θ_list[index][ii] = sol[index][ii][end]
#             end
#         end
#     end
#     return θ_list, index_list
# end


# If the solution doesn't converge in run_experiments_with_baselines, then we can continue here
function continue_experiments_with_baseline(g, θ_list, x0_set, expert_traj_list, parameterized_cost, 
                                                max_GD_iteration_num, Bayesian_update=true,
                                                all_equilibrium_types = ["FBNE_costate","OLNE_costate"])
    n_data = length(x0_set)
    n_equi_types = length(all_equilibrium_types)
    sol_table  = [[[] for jj in 1:n_data] for ii in 1:n_equi_types+1]
    grad_table = [[[] for jj in 1:n_data] for ii in 1:n_equi_types+1]
    equi_table = [[[] for jj in 1:n_data] for ii in 1:n_equi_types+1]
    comp_time_table = [[[] for jj in 1:n_data] for ii in 1:n_equi_types+1]
    conv_table = [[false for jj in 1:n_data] for ii in 1:n_equi_types+1] # converged_table
    loss_table = [[[] for jj in 1:n_data] for ii in 1:n_equi_types+1]
    total_iter_table = zeros(1+n_equi_types, n_data)
    @distributed for iter in 1:n_data
        x0 = x0_set[iter]
        expert_traj = expert_traj_list[iter]
        conv_table[1][iter], sol_table[1][iter], loss_table[1][iter], grad_table[1][iter], equi_table[1][iter]=objective_inference(x0,
                                                                        θ_list[1][iter],expert_traj,g,max_GD_iteration_num,"FBNE_costate", true)
        total_iter_table[1,iter] = iterations_taken_to_converge(equi_table[1][iter])
        for index in 1:n_equi_types
            conv_table[1+index][iter], sol_table[1+index][iter], loss_table[1+index][iter], grad_table[1+index][iter], equi_table[1+index][iter]=objective_inference(x0,
                                                                        θ_list[1+index][iter],expert_traj,g,max_GD_iteration_num, all_equilibrium_types[index], false)
            total_iter_table[1+index,iter] = iterations_taken_to_converge(equi_table[1+index][iter])
        end
    end
    return conv_table, sol_table, loss_table, grad_table, equi_table, total_iter_table, comp_time_table
end

