
using ForwardDiff
using Optim
using LinearAlgebra
include("diff_solver.jl")

θ=[10.0;2.0]

function parameterized_cost(θ::Vector)
    costs=(FunctionPlayerCost((g, x, u, t) -> (θ[1]*(x[1]-1)^2 + (2*(x[4]-1)^2 + (u[1]^2+u[2]^2) - 0.2*((x[1]-x[5])^2 + (x[2]-x[6])^2)))),
             FunctionPlayerCost((g, x, u, t) -> (θ[2]*(x[5]-1)^2+(2*(x[8]-1)^2+u[3]^2+u[4]^2-0.2*((x[1]-x[5])^2 + (x[2]-x[6])^2)))))
    return costs
end


function loss(θ, equilibrium_type, expert_traj)	
	nominal_game = GeneralGame(game_horizon, player_inputs, dynamics, parameterized_cost(ForwardDiff.value.(θ)))

	nominal_solver = iLQSolver(nominal_game, max_scale_backtrack=10, max_elwise_diff_step=Inf, equilibrium_type=equilibrium_type)
	x0 = SVector(0, 0, pi/2, 1.2,       1, 0, pi/2, 1)
	nominal_converged, nominal_traj, nominal_strategies = solve(nominal_game, nominal_solver, x0)
	




	costs = parameterized_cost(θ)
	game = GeneralGame(game_horizon, player_inputs, dynamics, costs)





	lqg = Differentiable_Solvers.lq_approximation(game, nominal_traj, nominal_solver)
	if equilibrium_type=="OLNE_KKT" || equilibrium_type=="OLNE_costate"
		traj = Differentiable_Solvers.trajectory(x0, game, Differentiable_Solvers.solve_lq_game_OLNE(lqg),nominal_traj)
	elseif equilibrium_type=="FBNE_KKT" || equilibrium_type=="FBNE_costate"
		traj = Differentiable_Solvers.trajectory(x0, game, Differentiable_Solvers.solve_lq_game_FBNE(lqg),nominal_traj)
    else
    	@warn "equilibrium_type is wrong!"
    end
    loss = norm(traj.x - expert_traj.x)^2 + norm(traj.u - expert_traj.u)^2
    return loss
end
# tagging, forward the tags. 
# 



























ForwardDiff.gradient(θ -> loss(θ, "FBNE_KKT", expert_traj2), [12.0; 3.0])


ForwardDiff.gradient(θ -> loss(θ, "OLNE_KKT", expert_traj1), [12.0; 3.0])


result = Optim.optimize(
        θ -> loss(θ, "FBNE_KKT", expert_traj2),
        [11.0;3.0];
        method = Optim.BFGS(),
        autodiff = :forward,
        extended_trace = true,
        iterations = 20,
        store_trace = true,
        x_tol = 1e-3,
    )

Optim.x_trace(result) 
Optim.f_trace(result)

