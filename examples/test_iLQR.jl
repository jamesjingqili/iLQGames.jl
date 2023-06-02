using iLQGames
import iLQGames: dx
import BenchmarkTools
using Plots
using ForwardDiff

nx, nu, ΔT, game_horizon = 8, 4, 0.1, 60


struct total_dynamics <: ControlSystem{ΔT, nx, nu } end
dx(cs::total_dynamics, x, u, t) = SVector(x[4]cos(x[3]), x[4]sin(x[3]), u[1], u[2], 
                                    x[8]cos(x[7]), x[8]sin(x[7]), 0, 0, 
                                    )
dynamics2 = total_dynamics()

costs = (FunctionPlayerCost((g, x, u, t) -> ( 10*(x[1]-1)^2  + u[1]^2 + u[2]^2 )),
         FunctionPlayerCost((g, x, u, t) -> (  2*(x[5] - x[1])^2 + 2*(x[8]-1)^2 + u[3]^2 + u[4]^2 )))

player_inputs = (SVector(1,2), SVector(3,4))

g = GeneralGame(game_horizon, player_inputs, dynamics2, costs)
x0 = SVector(0, 0.5, pi/2, 1,       1, 0, pi/2, 1)
solver = iLQSolver(g, max_scale_backtrack=5,max_n_iter=10, max_elwise_diff_step=Inf, equilibrium_type="Stackelberg_KKT_dynamic_factorization")
c, expert_traj, strategies = solve(g, solver, x0)

