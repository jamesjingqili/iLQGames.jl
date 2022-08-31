using Infiltrator
using LinearAlgebra
using iLQGames:
	SystemTrajectory
	GeneralGame
using Distributions

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
function inverse_game_gradient(current_loss::Float64, θ::Vector, g::GeneralGame, expert_traj::SystemTrajectory, 
								x0::SVector, parameterized_cost, equilibrium_type)
	num_parameters = length(θ)
	gradient = zeros(num_parameters)
	Δ = 0.001
	for ii in 1:num_parameters
		θ_new = copy(θ)
		θ_new[ii] += Δ
		new_loss, tmp_traj, tmp_strategy = inverse_game_loss(θ_new, g, expert_traj, x0, parameterized_cost, equilibrium_type)
		gradient[ii] = (new_loss-current_loss)/Δ
	end
	return gradient
end



# function inverse_game_gradient_LQ(solver, traj, θ::Vector, g::GeneralGame, expert_traj::SystemTrajectory, x0::SVector, 
#		parameterized_cost, equilibrium_type)
# 	num_parameters = length(θ)
# 	gradient = zeros(num_parameters)
# 	lqg = lq_approximation(solver, g, traj)

# 	return gradient
# end

# line search
function inverse_game_gradient_descent(θ::Vector, g::GeneralGame, expert_traj::SystemTrajectory, x0::SVector, max_GD_iteration_num::Int, 
										parameterized_cost, equilibrium_type)
	α = 1.0
	current_loss, current_traj, current_str = inverse_game_loss(θ, g, expert_traj, x0, parameterized_cost, equilibrium_type)
	θ_next = θ
	new_loss = 0.0
	# gradient = inverse_game_gradient(current_loss, θ, g, expert_traj, x0, parameterized_cost, equilibrium_type)
	gradient = ForwardDiff.gradient(x -> loss(x, equilibrium_type, expert_traj), θ)
	for iter in 1:max_GD_iteration_num
		θ_next = θ-α*gradient
		new_loss, new_traj, new_str = inverse_game_loss(θ_next, g, expert_traj, x0, parameterized_cost, equilibrium_type)
		if new_loss < current_loss
			println("Inverse Game Line Search Step Size: ", α)
			return θ_next, new_loss, gradient
			
			break
		end
		α = α*0.5
		println("inverse game line search not well")
	end
	return θ_next, new_loss, gradient
end




function inverse_game_update_belief(θ::Vector, g::GeneralGame, expert_traj::SystemTrajectory, x0::SVector, 
						parameterized_cost, equilibrium_type1, equilibrium_type2)
	current_cost = parameterized_cost(θ) # modify the cost vector
	current_game = GeneralGame(g.h, g.uids, g.dyn, current_cost)

	solver1 = iLQSolver(current_game, max_scale_backtrack=10, max_elwise_diff_step=Inf,equilibrium_type=equilibrium_type1)
	converged1, trajectory1, strategies1 = solve(current_game, solver1, x0)
	solver2 = iLQSolver(current_game, max_scale_backtrack=10, max_elwise_diff_step=Inf,equilibrium_type=equilibrium_type2)
	converged2, trajectory2, strategies2 = solve(current_game, solver2, x0)
	prior1 = 0.5
	prior2 = 0.5
	x1_list, u1_list, x2_list, u2_list, expert_traj_x, expert_traj_u = [], [], [], [], [], []
	for ii in 1:g.h
		for jj in 1:length(x0)
			# @infiltrate
			push!(x1_list, trajectory1.x[ii][jj])
			push!(x2_list, trajectory2.x[ii][jj])
			push!(expert_traj_x, expert_traj.x[ii][jj])
			if jj <= g.uids[end][end]
				push!(u1_list, trajectory1.u[ii][jj])
				push!(u2_list, trajectory2.u[ii][jj])
				push!(expert_traj_u, expert_traj.u[ii][jj])
			end
		end
	end
	# @infiltrate
	hypothesis_traj1 = Vector{Float64}([x1_list; u1_list])
	hypothesis_traj2 = Vector{Float64}([x2_list; u2_list])
	expert_traj_list = Vector{Float64}([expert_traj_x; expert_traj_u])
	n_dim = length([x1_list; u1_list])
	probability1 = pdf(MvNormal(hypothesis_traj1, Matrix{Float64}(I, n_dim, n_dim)), expert_traj_list)
	probability2 = pdf(MvNormal(hypothesis_traj2, Matrix{Float64}(I, n_dim, n_dim)), expert_traj_list)
	belief1 = (probability1*prior1)/(probability1*prior1+probability2*prior2)
	belief2 = (probability2*prior2)/(probability1*prior1+probability2*prior2)
	if belief1 > belief2
		return equilibrium_type1
	else
		return equilibrium_type2
	end
end

