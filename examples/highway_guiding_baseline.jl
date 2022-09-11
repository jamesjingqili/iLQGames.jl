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
using Distributions
include("diff_solver.jl")
include("inverse_game_solver.jl")
include("experiment_utils.jl")

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

x1_OL, y1_OL = [expert_traj1.x[i][1] for i in 1:game_horizon], [expert_traj1.x[i][2] for i in 1:game_horizon];
x2_OL, y2_OL = [expert_traj1.x[i][5] for i in 1:game_horizon], [expert_traj1.x[i][6] for i in 1:game_horizon];
anim1 = @animate for i in 1:game_horizon
    plot([x1_OL[i], x1_OL[i]], [y1_OL[i], y1_OL[i]], markershape = :square, label = "player 1, OL", xlims = (-0.5, 1.5), ylims = (0, 6))
    plot!([x2_OL[i], x2_OL[i]], [y2_OL[i], y2_OL[i]], markershape = :square, label = "player 2, OL", xlims = (-0.5, 1.5), ylims = (0, 6))
    plot!([0], seriestype = "vline", color = "black", label = "")
    plot!([1], seriestype = "vline", color = "black", label = "") 
end

gif(anim1, "lane_guiding_OL.gif", fps = 10)


# get a solver, choose initial conditions and solve (in about 9 ms with AD)
solver2 = iLQSolver(g, max_scale_backtrack=5, max_elwise_diff_step=Inf, equilibrium_type="FBNE_costate")
c2, expert_traj2, strategies2 = solve(g, solver2, x0)

x1_FB, y1_FB = [expert_traj2.x[i][1] for i in 1:game_horizon], [expert_traj2.x[i][2] for i in 1:game_horizon];
x2_FB, y2_FB = [expert_traj2.x[i][5] for i in 1:game_horizon], [expert_traj2.x[i][6] for i in 1:game_horizon];
anim2 = @animate for i in 1:game_horizon
    plot([x1_FB[i], x1_FB[i]], [y1_FB[i], y1_FB[i]], markershape = :square, label = "player 1, FB", xlims = (-0.5, 1.5), ylims = (0, 6))
    plot!([x2_FB[i], x2_FB[i]], [y2_FB[i], y2_FB[i]], markershape = :square, label = "player 2, FB", xlims = (-0.5, 1.5), ylims = (0, 6))    
    plot!([0], seriestype = "vline", color = "black", label = "")
    plot!([1], seriestype = "vline", color = "black", label = "")
end

gif(anim2, "lane_guiding_FB.gif", fps = 10)


#--------------------------------------------------------------------------------------------------------------------------------

function parameterized_cost(θ::Vector)
    costs = (FunctionPlayerCost((g, x, u, t) -> ( θ[1]*(x[5]-θ[4])^2 + θ[2]*(x[4]-1)^2 + θ[3]*(u[1]^2 + u[2]^2) - 0*((x[1]-x[5])^2 + (x[2]-x[6])^2))),
             FunctionPlayerCost((g, x, u, t) -> (  θ[5]*(x[5] - x[1])^2 + θ[6]*(x[8]-1)^2 + θ[7]*(u[3]^2 + u[4]^2) - 0*((x[1]-x[5])^2 + (x[2]-x[6])^2))))
    return costs
end
# function loss(θ, equilibrium_type, expert_traj, gradient_mode = true, specified_solver_and_traj = false, 
#                 nominal_solver=[], nominal_traj=[]) 
#     x0 = first(expert_traj.x)
#     if gradient_mode == false
#         nominal_game = GeneralGame(game_horizon, player_inputs, dynamics, parameterized_cost(ForwardDiff.value.(θ)))
#         nominal_solver = iLQSolver(nominal_game, max_scale_backtrack=10, max_elwise_diff_step=Inf, equilibrium_type=equilibrium_type)
#         nominal_converged, nominal_traj, nominal_strategies = solve(nominal_game, nominal_solver, x0)
#         loss_value = norm(nominal_traj.x - expert_traj.x)^2 + norm(nominal_traj.u-expert_traj.u)^2
#         # @infiltrate
#         return loss_value, nominal_traj, nominal_strategies, nominal_solver
#     else
#         if specified_solver_and_traj == false
#             nominal_game = GeneralGame(game_horizon, player_inputs, dynamics, parameterized_cost(ForwardDiff.value.(θ)))
#             nominal_solver = iLQSolver(nominal_game, max_scale_backtrack=10, max_elwise_diff_step=Inf, equilibrium_type=equilibrium_type)
#             nominal_converged, nominal_traj, nominal_strategies = solve(nominal_game, nominal_solver, x0)
#         end
#         costs = parameterized_cost(θ)
#         game = GeneralGame(game_horizon, player_inputs, dynamics, costs)

#         lqg = Differentiable_Solvers.lq_approximation(game, nominal_traj, nominal_solver)
#         if equilibrium_type=="OLNE_KKT" || equilibrium_type=="OLNE_costate" || equilibrium_type=="OLNE"
#             traj = Differentiable_Solvers.trajectory(x0, game, Differentiable_Solvers.solve_lq_game_OLNE(lqg), nominal_traj)
#         elseif equilibrium_type=="FBNE_KKT" || equilibrium_type=="FBNE_costate" || equilibrium_type=="FBNE"
#             traj = Differentiable_Solvers.trajectory(x0, game, Differentiable_Solvers.solve_lq_game_FBNE(lqg), nominal_traj)
#         else
#             @warn "equilibrium_type is wrong!"
#         end
#         loss_value = norm(traj.x - expert_traj.x)^2 + norm(traj.u - expert_traj.u)^2
#         return loss_value
#     end
# end

# θ = [10.0;2.0;1.0;1.0;4.0;2.0;1.0;]
# current_loss, _, _ = inverse_game_loss(θ, g, expert_traj2, x0, parameterized_cost, "FBNE_costate")
# gradient1 = inverse_game_gradient(current_loss, θ, g, expert_traj2, x0, parameterized_cost, "FBNE_costate")
# gradient2 = ForwardDiff.gradient(x -> loss(x, "FBNE_costate", expert_traj2), θ)


#--------------------------------------------------------------------------------------------------------------------------------


function inverse_game_gradient_descent(θ::Vector, g::GeneralGame, expert_traj::SystemTrajectory, x0::SVector, 
                                        max_GD_iteration_num::Int, parameterized_cost, equilibrium_type=[], Bayesian_belief_update=false)
    α, θ_next, new_loss = 2.0, θ, 0.0
    if Bayesian_belief_update==true
        equilibrium_type = inverse_game_update_belief(θ, g, expert_traj, x0, parameterized_cost, "FBNE_costate", "OLNE_costate")
    end
    current_loss, current_traj, current_str, current_solver = loss(θ, equilibrium_type, expert_traj, false)
    # gradient_value = inverse_game_gradient(current_loss, θ, g, expert_traj, x0, parameterized_cost, equilibrium_type)
    gradient_value = ForwardDiff.gradient(x -> loss(x, equilibrium_type, expert_traj, true, true, current_solver, current_traj), θ)
    for iter in 1:max_GD_iteration_num
        
        θ_next = θ-α*gradient_value
        while minimum(θ_next[[1,2,3,5,6,7]])<=0
            # @infiltrate
            α = α*0.5^4
            θ_next = θ-α*gradient_value
        end
        new_loss, new_traj, new_str, new_solver = loss(θ_next, equilibrium_type, expert_traj, false)
        if new_loss < current_loss
            println("Inverse Game Line Search Step Size: ", α)
            return θ_next, new_loss, gradient_value, equilibrium_type
            break
        end
        α = α*0.5
        println("inverse game line search not well")
    end
    return θ_next, new_loss, gradient_value, equilibrium_type
end

max_GD_iteration_num =200

θ = [2.0; 2.0; 2.0; 2.0; 2.0; 2.0; 2.0]
θ_dim = length(θ)
sol = [zeros(θ_dim) for iter in 1:max_GD_iteration_num+1]
sol[1] = θ
loss_values = zeros(max_GD_iteration_num+1)
loss_values[1],_,_ = inverse_game_loss(sol[1], g, expert_traj2, x0, parameterized_cost, "FBNE_costate")
gradient = [zeros(θ_dim) for iter in 1:max_GD_iteration_num]
equilibrium_type = ["" for iter in 1:max_GD_iteration_num]
for iter in 1:max_GD_iteration_num
    sol[iter+1], loss_values[iter+1], gradient[iter], equilibrium_type[iter] = inverse_game_gradient_descent(sol[iter], g, expert_traj2, x0, 10, 
                                                                            parameterized_cost, [],true)
    println("Current solution: ", sol[iter+1])
    if loss_values[iter+1]<0.1
        break
    end
end

#--------------------------------------------------------------------------------------------------------------------------------
"Robustness to observation noise"

noisy_expert_traj1 = deepcopy(expert_traj1)
for ii in 1:g.h
    noisy_expert_traj1.x[ii] = expert_traj1.x[ii] + 0.05*rand(Normal(0,1), nx)
    noisy_expert_traj1.u[ii] = expert_traj1.u[ii] + 0.05*rand(Normal(0,1), nu)
end
noisy_expert_traj2 = expert_traj2
for ii in 1:g.h
    noisy_expert_traj2.x[ii] = expert_traj2.x[ii] + 0.05*rand(Normal(0,1), nx)
    noisy_expert_traj2.u[ii] = expert_traj2.u[ii] + 0.05*rand(Normal(0,1), nu)
end


max_GD_iteration_num =40

θ = [2.0; 2.0; 2.0; 2.0; 2.0; 2.0]
θ_dim = length(θ)
sol = [zeros(θ_dim) for iter in 1:max_GD_iteration_num+1]
sol[1] = θ
loss_values = zeros(max_GD_iteration_num+1)
loss_values[1],_,_ = inverse_game_loss(sol[1], g, noisy_expert_traj2, x0, parameterized_cost, "OLNE_costate")
gradient = [zeros(θ_dim) for iter in 1:max_GD_iteration_num]
equilibrium_type = ["" for iter in 1:max_GD_iteration_num]
for iter in 1:max_GD_iteration_num
    sol[iter+1], loss_values[iter+1], gradient[iter], equilibrium_type[iter] = inverse_game_gradient_descent(sol[iter], g, noisy_expert_traj2, x0, 10, 
                                                                            parameterized_cost, [],true)
    println("Current solution: ", sol[iter+1])
    if loss_values[iter+1]<0.1
        break
    end
end

#--------------------------------------------------------------------------------------------------------------------------------
"plot the learned trajectory"

selected_item = 1
loss_test1, traj_test1, _, _=loss(sol[selected_item], equilibrium_type[selected_item], expert_traj2, false)

# g_test = GeneralGame(game_horizon, player_inputs, dynamics, parameterized_cost(sol[selected_item]))
# solver_test = iLQSolver(g_test, max_scale_backtrack=10, max_elwise_diff_step=Inf,max_elwise_diff_converged=0.05, equilibrium_type[selected_item])
# converged_test1, traj_test1, strategies_test1 = solve(g_test, solver_test, x0)


x1_test1, y1_test1 = [traj_test1.x[i][1] for i in 1:game_horizon], [traj_test1.x[i][2] for i in 1:game_horizon];
x2_test1, y2_test1 = [traj_test1.x[i][5] for i in 1:game_horizon], [traj_test1.x[i][6] for i in 1:game_horizon];
anim1_test = @animate for i in 1:game_horizon
    plot([x1_test1[i], x1_test1[i]], [y1_test1[i], y1_test1[i]], markershape = :square, label = "player 1, FB", 
        xlims = (-0.5, 1.5), ylims = (0, 6))
    plot!([x2_test1[i], x2_test1[i]], [y2_test1[i], y2_test1[i]], markershape = :square, label = "player 2, FB", 
        xlims = (-0.5, 1.5), ylims = (0, 6))
    plot!([0], seriestype = "vline", color = "black", label = "")
    plot!([1], seriestype = "vline", color = "black", label = "") 
end

gif(anim1_test, "test_noisy_highway.gif", fps = 10)


#----------------------------------------------------------------------------------------------------------------------------------
# max_GD_iteration_num = 40


function single_experiment(x0, θ, expert_traj, g, max_GD_iteration_num, equilibrium_type=[], Bayesian_update=false)
    θ_dim = length(θ)
    sol = [zeros(θ_dim) for iter in 1:max_GD_iteration_num+1]
    sol[1] = θ
    loss_values = zeros(max_GD_iteration_num+1)
    loss_values[1],_,_ = inverse_game_loss(sol[1], g, expert_traj, x0, parameterized_cost, equilibrium_type)
    gradient = [zeros(θ_dim) for iter in 1:max_GD_iteration_num]
    equilibrium_type_list = ["" for iter in 1:max_GD_iteration_num]
    converged = false
    for iter in 1:max_GD_iteration_num
        sol[iter+1], loss_values[iter+1], gradient[iter], equilibrium_type_list[iter] = inverse_game_gradient_descent(sol[iter], g, expert_traj, x0, 10, 
                                                                                parameterized_cost, equilibrium_type, Bayesian_update)
        println("Current solution: ", sol[iter+1])
        if loss_values[iter+1]<0.1
            converged = true
            break
        end
    end

    return converged, sol, loss_values, gradient, equilibrium_type_list
end

function test_experiments(g, θ, x0_set, expert_traj_set, parameterized_cost, max_GD_iteration_num, Bayesian_update=true)
    # In the returned table, the rows coresponding to FB, OL and Bayesian
    n_data = length(x0_set)
    sol_table  = [[[] for jj in 1:n_data] for ii in 1:3]
    loss_table = [[[] for jj in 1:n_data] for ii in 1:3]
    grad_table = [[[] for jj in 1:n_data] for ii in 1:3]
    equi_table = [[[] for jj in 1:n_data] for ii in 1:3]
    comp_time_table = [[[] for jj in 1:n_data] for ii in 1:3]
    conv_table = [[[] for jj in 1:n_data] for ii in 1:3] # converged_table
    for iter in 1:n_ndata
        x0 = x0_set[iter]
        expert_traj = expert_traj_set[iter]
        # FB
        conv_table[1][iter], sol_table[1][iter], loss_table[1][iter], grad_table[1][iter], equi_table[1][iter], comp_time_table[1][iter] = single_experiment(x0,θ,expert_traj,g,"FBNE_costate", false)
        # OL
        conv_table[2][iter], sol_table[2][iter], loss_table[2][iter], grad_table[2][iter], equi_table[2][iter], comp_time_table[2][iter] = single_experiment(x0,θ,expert_traj,g,"OLNE_costate", false)
        # Bayesian
        conv_table[3][iter], sol_table[3][iter], loss_table[3][iter], grad_table[3][iter], equi_table[3][iter], comp_time_table[3][iter] = single_experiment(x0,θ,expert_traj,g,"FBNE_costate", true)
    end

    return sol_table, loss_table, gradient_table, equilibrium_table, computation_time_table
end




# test robustness to observation noise




# generalization to unseen initial state




# average computation time: FB, OL, Joint






