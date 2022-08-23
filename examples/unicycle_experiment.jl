using iLQGames
import iLQGames: dx
using Plots

# parametes: number of states, number of inputs, sampling time, horizon
nx, nu, ΔT, game_horizon = 4, 2, 0.1, 2

# setup the dynamics
struct LinearSystem <: ControlSystem{ΔT,nx,nu} end
# state: (px, py, phi, v)
dx(cs::LinearSystem, x, u, t) = SVector(0.5*u[2], 2*u[1], u[1], u[2])
dynamics = LinearSystem()

# player-1 wants the unicycle to stay close to the origin,
# player-2 wants to keep close to 1 m/s
costs = (FunctionPlayerCost((g, x, u, t) -> (1/2*x[1]^2 + 1/2*u[1]^2)),
         FunctionPlayerCost((g, x, u, t) -> (1/2*x[2]^2 + 1/2*u[2]^2)))

# indices of inputs that each player controls
player_inputs = (SVector(1), SVector(2))
# the horizon of the game
g = GeneralGame(game_horizon, player_inputs, dynamics, costs)

# get a solver, choose initial conditions and solve (in about 9 ms with AD)
solver1 = iLQSolver(g, max_n_iter=10, equilibrium_type="FBNE_costate")
x0 = SVector(1, 4, 2, 3)
converged1, trajectory1, strategies1 = solve(g, solver1, x0)


solver2 = iLQSolver(g, max_n_iter=10, equilibrium_type="FBNE_KKT")
converged2, trajectory2, strategies2 = solve(g, solver2, x0)


x1, y1 = [trajectory1.x[i][1] for i in 1:game_horizon], [trajectory1.x[i][2] for i in 1:game_horizon];
x2, y2 = [trajectory2.x[i][1] for i in 1:game_horizon], [trajectory2.x[i][2] for i in 1:game_horizon];

plot(x1, y1)

plot!(x2, y2)



OL, OL_KKT = deepcopy(current_strategy), deepcopy(current_strategy)
solve_lq_game_OLNE!(OL, lqg_approx)
solve_lq_game_OLNE_KKT!(OL_KKT, lqg_approx)


