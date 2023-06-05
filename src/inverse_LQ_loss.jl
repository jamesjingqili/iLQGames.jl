function inverse_LQ_loss(x_expert, sol, game_horizon, player_inputs, dynamics, x0)
    @time costs = (FunctionPlayerCost((g, x, u, t) -> ( sol[1]*x[1]^2 + sol[1]*x[2]^2 + u[1]^2 )),
            FunctionPlayerCost((g, x, u, t) -> ( sol[2]*x[1]^2 + sol[2]*x[2]^2 + 4*u[2]^2)))

    @time g = GeneralGame(game_horizon, player_inputs, dynamics, costs)
    @time solver = iLQSolver(g, max_scale_backtrack=10, max_elwise_diff_step=Inf,max_n_iter = 3, equilibrium_type="FBNE")
    @time c, x, π = solve(g, solver, x0)
    return norm(x_expert.x - x.x, 2)
end