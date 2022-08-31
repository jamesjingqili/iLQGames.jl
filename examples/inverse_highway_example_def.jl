using Infiltrator
using ForwardDiff
using Optim
using LinearAlgebra
include("diff_solver.jl")
include("inverse_game_solver.jl")

# θ=[10.0; 3.0;]


# function parameterized_cost(θ::Vector)
#     costs = (FunctionPlayerCost((g, x, u, t) -> ( θ[1]*(x[1]-1)^2 + θ[2]*(2*(x[4]-1)^2 + u[1]^2 + u[2]^2 - 0.2*((x[1]-x[5])^2 + (x[2]-x[6])^2)))),
#              FunctionPlayerCost((g, x, u, t) -> ( θ[3]*(x[5]-1)^2 + θ[4]*(2*(x[8]-1)^2 + u[3]^2 + u[4]^2 - 0.2*((x[1]-x[5])^2 + (x[2]-x[6])^2)))))
#     return costs
# end

function parameterized_cost(θ::Vector)
    costs = (FunctionPlayerCost((g, x, u, t) -> ( θ[1]*x[3]^2 + θ[2]*(x[4]^2 + 0.5*u[1]^2 + 0.5*u[2]^2))),
             FunctionPlayerCost((g, x, u, t) -> ( θ[3]*(x[1]-x[3])^2 + θ[4]*((x[2]-x[4])^2 + 0.5*u[3]^2 + 0.5*u[4]^2))))
    return costs
end


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


# tagging, forward the tags. 
# 
function inverse_game_gradient_descent(θ::Vector, g::GeneralGame, expert_traj::SystemTrajectory, x0::SVector, max_GD_iteration_num::Int, parameterized_cost, equilibrium_type)
    α = 1.0
    current_loss, current_traj, current_str = inverse_game_loss(θ, g, expert_traj, x0, parameterized_cost, equilibrium_type)
    θ_next = θ
    new_loss = 0.0
    # gradient = inverse_game_gradient(current_loss, θ, g, expert_traj, x0, parameterized_cost, equilibrium_type)
    gradient_value = ForwardDiff.gradient(x -> loss(x, equilibrium_type, expert_traj), θ)
    for iter in 1:max_GD_iteration_num
        θ_next = θ-α*gradient_value
        new_loss, new_traj, new_str = inverse_game_loss(θ_next, g, expert_traj, x0, parameterized_cost, equilibrium_type)
        if new_loss < current_loss
            println("Inverse Game Line Search Step Size: ", α)
            return θ_next, new_loss, gradient_value
            break
        end
        α = α*0.5
        println("inverse game line search not well")
    end
    return θ_next, new_loss, gradient_value
end



function compare_grad(θ, equilibrium_type1, equilibrium_type2)
    current_loss, _, _ = inverse_game_loss(θ, g, expert_traj2, x0, parameterized_cost, equilibrium_type1)
    gradient1 = inverse_game_gradient(current_loss, θ, g, expert_traj1, x0, parameterized_cost, equilibrium_type1)
    gradient2 = ForwardDiff.gradient(x -> loss(x, equilibrium_type2, expert_traj2), θ)
end


θ = [3.6;3.0;3.0;3.0]
current_loss, _, _ = inverse_game_loss(θ, g, expert_traj2, x0, parameterized_cost, "FBNE")
gradient1 = inverse_game_gradient(current_loss, θ, g, expert_traj2, x0, parameterized_cost, "FBNE")
gradient2 = ForwardDiff.gradient(x -> loss(x, "FBNE", expert_traj2), θ)




















ForwardDiff.gradient(θ -> loss(θ, "FBNE_costate", expert_traj2, true, false), [12.0])


ForwardDiff.gradient(θ -> loss(θ, "OLNE", expert_traj1, true, false), [9.0; 3.0])


result = Optim.optimize(
        θ -> loss(θ, "FBNE", expert_traj2),
        [8.0];
        method = Optim.AcceleratedGradientDescent(),
        autodiff = :forward,
        extended_trace = true,
        iterations = 10,
        store_trace = true,
        x_tol = 1e-3,
    )


result = Optim.optimize(
        θ -> loss(θ, "OLNE", expert_traj1),
        [11.0;2.0];
        method = Optim.GradientDescent(),
        autodiff = :forward,
        extended_trace = true,
        iterations = 20,
        store_trace = true,
        x_tol = 1e-3,
    )


Optim.x_trace(result) 
Optim.f_trace(result)

