using LinearAlgebra


function loss(θ, current_traj, expert_traj)
	return norm(current_traj - expert_traj)
end

function OLNE_LQ_gradient(g, x0)
	T = horizon(g)

end

function update_belief(g, cost_estimate, current_traj, expert_traj)
	# Bayesian update
end

function inverse_game_line_search(g, cost_estimate, gradient, current_traj, expert_traj)
	# call 'loss()' and do the line search
end



for iteration in range(1000):
	# update belief
	# calculate gradient
	# line search
	# if converged, then output the cost estimate.
end

