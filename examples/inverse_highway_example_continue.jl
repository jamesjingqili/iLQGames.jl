function inverse_game_gradient_descent(θ::Vector, g::GeneralGame, expert_traj::SystemTrajectory, x0::SVector, 
                                        max_GD_iteration_num::Int, parameterized_cost, equilibrium_type)
    α = 1.0
    current_loss, current_traj, current_str, current_solver = loss(θ, equilibrium_type, expert_traj, false)
    θ_next = θ
    new_loss = 0.0
    gradient = inverse_game_gradient(current_loss, θ, g, expert_traj, x0, parameterized_cost, equilibrium_type)
    # gradient = ForwardDiff.gradient(x -> loss(x, equilibrium_type, expert_traj, true, true, current_solver, current_traj), θ)
    for iter in 1:max_GD_iteration_num
        θ_next = θ-α*gradient
        new_loss, new_traj, new_str, new_solver = loss(θ_next, equilibrium_type, expert_traj, false)
        if new_loss < current_loss
            println("Inverse Game Line Search Step Size: ", α)
            @infiltrate
            return θ_next, new_loss, gradient
            break
        end
        α = α*0.5
        println("inverse game line search not well")
    end
    return θ_next, new_loss, gradient
end


max_GD_iteration_num = 20


θ = [13.0;]
θ_dim = length(θ)
sol = [zeros(θ_dim) for iter in 1:max_GD_iteration_num+1]
sol[1] = θ
loss_values = zeros(max_GD_iteration_num)
gradient = [zeros(θ_dim) for iter in 1:max_GD_iteration_num]
for iter in 1:max_GD_iteration_num
    sol[iter+1], loss_values[iter], gradient[iter] = inverse_game_gradient_descent(sol[iter], g, expert_traj1, x0, 10, 
                                                                            parameterized_cost, "OLNE_costate")
    println("Current solution: ", sol[iter+1])
    if loss_values[iter]<0.1
        break
    end
end



#------------------------------------------------------------------------------------------------
g_test = GeneralGame(game_horizon, player_inputs, dynamics, parameterized_cost(sol[1]))
solver_test = iLQSolver(g_test, max_scale_backtrack=10, max_elwise_diff_step=Inf,max_elwise_diff_converged=0.05, equilibrium_type="OLNE_KKT")
converged_test1, traj_test1, strategies_test1 = solve(g_test, solver_test, x0)


x1_test1, y1_test1 = [traj_test1.x[i][1] for i in 1:game_horizon], [traj_test1.x[i][2] for i in 1:game_horizon];
x2_test1, y2_test1 = [traj_test1.x[i][5] for i in 1:game_horizon], [traj_test1.x[i][6] for i in 1:game_horizon];
anim1_test = @animate for i in 1:game_horizon
    plot([x1_test1[i], x1_test1[i]], [y1_test1[i], y1_test1[i]], markershape = :square, label = "player 1, OL", 
        xlims = (-0.5, 1.5), ylims = (0, 6))
    plot!([x2_test1[i], x2_test1[i]], [y2_test1[i], y2_test1[i]], markershape = :square, label = "player 2, OL", 
        xlims = (-0.5, 1.5), ylims = (0, 6))
    plot!([0], seriestype = "vline", color = "black", label = "")
    plot!([1], seriestype = "vline", color = "black", label = "") 
end

gif(anim1_test, "test_car_OLNE_costate_LineSearch.gif", fps = 10)

#-------------------------------------------------------------------------------------------------
max_GD_iteration_num = 10

θ_dim = length(θ)
sol = [zeros(θ_dim) for iter in 1:max_GD_iteration_num+1]
sol[1] = θ
loss = zeros(max_GD_iteration_num)
gradient = [zeros(θ_dim) for iter in 1:max_GD_iteration_num]
for iter in 1:max_GD_iteration_num
    sol[iter+1], loss[iter], gradient[iter] = inverse_game_gradient_descent(sol[iter], g, expert_traj2, x0, 20, 
                                                                            parameterized_cost, "FBNE_costate")
    if loss[iter]<0.1
        break
    end
end


g_test = GeneralGame(game_horizon, player_inputs, dynamics, parameterized_cost(sol[6]))
solver_test = iLQSolver(g_test, max_scale_backtrack=10, max_elwise_diff_step=Inf,max_elwise_diff_converged=0.05, equilibrium_type="FBNE_costate")
converged_test1, traj_test1, strategies_test1 = solve(g_test, solver_test, x0)


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

gif(anim1_test, "test_car_FBNE_costate_LineSearch.gif", fps = 10)



result=Optim.optimize(
    θ -> inverse_game_loss(θ, g, expert_traj1, x0, parameterized_cost, "OLNE_costate"),
    [1;1;1],
    method = Optim.BFGS(),
    autodiff = :forward,
    extended_trace = true,
    iterations = 10,
    store_trace=true,
    x_tol=1e-3
    )

