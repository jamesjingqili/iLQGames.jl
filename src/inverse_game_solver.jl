using Infiltrator
using LinearAlgebra

# evaluate trajectory prediction loss
function inverse_game_loss(θ::Vector, g::GeneralGame, expert_traj::SystemTrajectory, x0::SVector, parameterized_cost, equilibrium_type)
	current_cost = parameterized_cost(θ) # modify the cost vector
	current_game = GeneralGame(g.h, g.uids, g.dyn, current_cost)
	solver = iLQSolver(current_game, max_scale_backtrack=10, max_elwise_diff_step=Inf,equilibrium_type=equilibrium_type)
	converged, trajectory, strategies = solve(current_game, solver, x0)
	
	if converged==true
		println("converged!")
	else
		println("not converge T_T")
	end

	return norm(trajectory.x - expert_traj.x)^2 + norm(trajectory.u - expert_traj.u)^2, trajectory, strategies
end

# get gradient
function inverse_game_gradient(current_loss::Float64, θ::Vector, g::GeneralGame, expert_traj::SystemTrajectory, x0::SVector, parameterized_cost, equilibrium_type)
	num_parameters = length(θ)
	gradient = zeros(num_parameters)
	Δ = 0.01
	for ii in 1:num_parameters
		θ_new = copy(θ)
		θ_new[ii] += Δ
		new_loss, tmp_traj, tmp_strategy = inverse_game_loss(θ_new, g, expert_traj, x0, parameterized_cost, equilibrium_type)
		gradient[ii] = (new_loss-current_loss)/Δ
		# @infiltrate
	end
	return gradient
end



function inverse_game_gradient_LQ(solver, traj, θ::Vector, g::GeneralGame, expert_traj::SystemTrajectory, x0::SVector, parameterized_cost, equilibrium_type)
	num_parameters = length(θ)
	gradient = zeros(num_parameters)
	lqg = lq_approximation(solver, g, traj)

	return gradient
end

# line search
function inverse_game_gradient_descent(θ::Vector, g::GeneralGame, expert_traj::SystemTrajectory, x0::SVector, max_GD_iteration_num::Int, parameterized_cost, equilibrium_type)
	α = 1.0
	current_loss, current_traj, current_str = inverse_game_loss(θ, g, expert_traj, x0, parameterized_cost, equilibrium_type)
	θ_next = θ
	new_loss = 0.0
	gradient = inverse_game_gradient(current_loss, θ, g, expert_traj, x0, parameterized_cost, equilibrium_type)
	for iter in 1:max_GD_iteration_num
		θ_next = θ-α*gradient
		new_loss, new_traj, new_str = inverse_game_loss(θ_next, g, expert_traj, x0, parameterized_cost, equilibrium_type)
		if new_loss < current_loss
			println("Inverse Game Line Search Step Size: ", α)
			return θ_next, new_loss, gradient
			@infiltrate
			break
		end
		α = α*0.5
		println("inverse game line search not well")
	end
	return θ_next, new_loss, gradient
end


function define_gradient_game(g, θ, x0)
	new_game = 
	return new_game
end


# 
function inverse_game_update_belief(θ::Vector, cost_basis::PlayerCost, g::GeneralGame,
						expert_traj::SystemTrajectory, x0::SVector)
	# return belief
end



# 1. how to define new game structure?
# 2. how to 