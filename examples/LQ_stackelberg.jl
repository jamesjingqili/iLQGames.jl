# copied from linear_system_example.jl
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


nx, nu, ΔT, game_horizon = 4, 4, 0.1, 5

struct LinearSystem <: ControlSystem{ΔT,nx,nu} end
dx(cs::LinearSystem, x, u, t) = SVector(u[1],u[2],u[3],u[4])
dynamics = LinearSystem()

costs = (FunctionPlayerCost((g, x, u, t) -> ( 2*(x[3])^2 + 2*(x[4])^2 + u[1]^2 + u[2]^2)),
         FunctionPlayerCost((g, x, u, t) -> ( 2*(x[1]-x[3])^2 + 2*(x[2]-x[4])^2 + u[3]^2 + u[4]^2)))

# indices of inputs that each player controls
player_inputs = (SVector(1,2), SVector(3,4))
# the horizon of the game
g = GeneralGame(game_horizon, player_inputs, dynamics, costs)
x0 = SVector(0.3, 1.1, 1,1/3)


solver3 = iLQSolver(g, 
    max_scale_backtrack=1,
    max_elwise_diff_converged = 1e-4, 
    max_elwise_diff_step = 1e-3, 
    max_n_iter = 1, 
    equilibrium_type="Stackelberg_KKT_dynamic_factorization")
@time c3, x3, π3 = solve(g, solver3, x0)










solver4 = iLQSolver(g, max_elwise_diff_converged = 1e-4, max_elwise_diff_step = 1e-3, max_n_iter = 10000,
    equilibrium_type="Stackelberg_KKT_dynamic_factorization")
@time c4, x4, π4 = solve(g, solver4, x0)

solver5 = iLQSolver(g, max_elwise_diff_converged = 1e-4, max_elwise_diff_step = 1e-3, max_n_iter = 100,
    equilibrium_type="Stackelberg_KKT_dynamic_factorization")
@time c5, x5, π5 = solve(g, solver5, x0)



solver1 = iLQSolver(g, max_scale_backtrack=10, max_elwise_diff_step=Inf, equilibrium_type="Stackelberg_KKT_dynamic_factorization")
@time c1, x1, π1 = solve(g, solver1, x0)



solver = iLQSolver(g, max_scale_backtrack=10, max_elwise_diff_step=Inf, equilibrium_type="Stackelberg_KKT")
@time c, x, π = solve(g, solver, x0)


solver2 = iLQSolver(g, max_scale_backtrack=10, max_elwise_diff_step=Inf, equilibrium_type="FBNE")
@time c2, x2, π2 = solve(g, solver2, x0)


solver0 = iLQSolver(g, max_elwise_diff_converged = 1e-4, max_elwise_diff_step = 1e-3, max_n_iter = 8000)
convergence, x_traj, strategies = solve(g, solver0, x0)




# test stationarity
lqg_approx= solver0._lq_mem

current_op= x_traj
last_op= x_traj

iLQGames.lqg_approximation!(lqg_approx, solver, g, current_op)
_,_,_ = iLQGames.solve_lq_game_Stackelberg_KKT_dynamic_factorization!(strategies, g, x0)
iLQGames.trajectory!(current_op, dynamics, π, last_op, x0)
copyto!(last_op, current_op)

# ------------- new code -----------------
using iLQGames
using iLQGames: LinearSystem
using Plots
using ForwardDiff
using iLQGames:
    SystemTrajectory
using Infiltrator
using Optim
using LinearAlgebra


nx, nu, ΔT, game_horizon = 4, 4, 0.1, 10

#struct LinearSystem <: ControlSystem{ΔT,nx,nu} end
#dx(cs::LinearSystem, x, u, t) = SVector(u[1],u[2],u[3],u[4])
#dynamics = LinearSystem()
dynamics = let
    dyn = LinearSystem{ΔT}(SMatrix{4,4}(I), ΔT * SMatrix{4,4}(I))
    LTISystem(dyn, nothing, nothing)
end

costs = (FunctionPlayerCost((g, x, u, t) -> (2 * (x[3])^2 + 2 * (x[4])^2 + u[1]^2 + u[2]^2)),
    FunctionPlayerCost((g, x, u, t) -> (2 * (x[1] - x[3])^2 + 2 * (x[2] - x[4])^2 + u[3]^2 + u[4]^2)))

# indices of inputs that each player controls
player_inputs = (SVector(1, 2), SVector(3, 4))
# the horizon of the game
g = GeneralGame(game_horizon, player_inputs, dynamics, costs)
x0 = SVector(0, 1, 1, 1 / 3)

solver0 = iLQSolver(g, equilibrium_type="Stackelberg_KKT_dynamic_factorization")
convergence, x_traj, strategies = solve(g, solver0, x0)

